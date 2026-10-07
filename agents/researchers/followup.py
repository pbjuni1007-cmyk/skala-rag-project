"""Initial market, stakeholder and domain research runner."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from typing import Any

from pydantic import BaseModel

from rag.budget import write_json
from rag.context import assessment_evidence
from rag.evidence import PERSPECTIVE_FACETS, validate_perspective
from rag.reassessment import reassess_facets
from rag.schemas import Assessment, PerspectiveQueries

from agents.researchers.prompts import (
    DEFAULT_WEB_QUERIES,
    PERSPECTIVE_PROMPTS,
    PERSPECTIVE_REASSESSMENT_PROMPT,
    PERSPECTIVE_REWRITE_PROMPT,
)


def _as_dict(value: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    raise TypeError(f"Expected a mapping or Pydantic model, got {type(value).__name__}")


def _get(value, key):
    if isinstance(value, Mapping):
        return value[key]
    return getattr(value, key)


def _stable_chunk(chunk):
    """Ignore the query-dependent retrieval score when checking artifact identity."""
    payload = _as_dict(chunk)
    payload.pop("score", None)
    return payload


def _merge_chunks(target, candidates, artifact_mismatch):
    for candidate in candidates:
        chunk = _as_dict(candidate)
        chunk_id = chunk["id"]
        existing = target.get(chunk_id)
        if existing is not None and _stable_chunk(existing) != _stable_chunk(chunk):
            raise artifact_mismatch("Conflicting duplicate source chunk")
        if existing is None:
            target[chunk_id] = chunk


def _cited_ids(assessments):
    return {
        reference["chunk_id"]
        for assessment in assessments.values()
        for claim in _as_dict(assessment).get("claims", [])
        for reference in claim.get("references", [])
    }


def _fallback_cited_chunks(corpus, tech_assessment):
    wanted = _cited_ids(tech_assessment)
    found = {}
    for chunk in corpus.chunks:
        chunk_id = _get(chunk, "id")
        if chunk_id in wanted:
            found[chunk_id] = _as_dict(chunk)
    missing = wanted - set(found)
    if missing:
        raise ValueError("Cited research chunks are absent from the corpus")
    return list(found.values())


def _perspective_check(chunks, view):
    def check(value):
        errors = validate_perspective(value, chunks, view)
        has_unknown = any(claim["kind"] == "unknown" for claim in value["claims"])
        if (value["status"] == "insufficient" or has_unknown) and not any(
            gap.strip() for gap in value["gaps"]
        ):
            errors.append("Insufficient or unknown claims require non-empty gaps")
        return errors

    return check


def _rewrite_check(missing):
    expected = set(missing)

    def check(value):
        queries = value["queries"]
        if len(queries) == len(missing) and {item["facet"] for item in queries} == expected:
            return []
        return ["Rewrite must contain one query for each missing facet"]

    return check


def _missing_facets(assessment):
    missing = sorted({claim["facet"] for claim in assessment["claims"] if claim["kind"] == "unknown"})
    if assessment["status"] == "insufficient" and not missing:
        missing = sorted({claim["facet"] for claim in assessment["claims"]})
    return missing


def run_followup(pipeline, request):
    """Run one initial follow-up view and return its keyed Assessment and chunks."""
    # Keep graph imports local so the Supervisor can import this runner without a cycle.
    from rag.graph import BASE, StructuredValidationError, token_context

    from agents.researchers.result import ArtifactMismatch, RetrievalFailure, balance_findings, contract_chunk

    view = _get(request, "view")
    if view not in PERSPECTIVE_FACETS:
        raise ValueError(f"Unsupported follow-up view: {view}")

    context = _get(request, "context")
    domain = _get(context, "domain")
    scenario = _get(context, "scenario")
    questions = list(_get(request, "questions"))
    feedback = list(_get(request, "feedback"))
    research_context = _get(request, "research_context")
    tech_assessment = {
        key: _as_dict(value)
        for key, value in _get(research_context, "assessments").items()
    }

    run_log = {
        "request_id": _get(request, "request_id"),
        "view": view,
        "queries": [],
        "research_chunk_ids": [],
        "inherited_chunk_ids": [],
        "web_chunk_ids": [],
        "expanded_chunk_ids": [],
        "missing_facets": [],
        "assessment_status": None,
    }
    balance_log = None

    try:
        corpus_by_id = {}
        for raw_chunk in pipeline.corpus.chunks:
            chunk = _as_dict(raw_chunk)
            chunk_id = chunk["id"]
            existing = corpus_by_id.get(chunk_id)
            if existing is not None and _stable_chunk(existing) != _stable_chunk(chunk):
                raise ArtifactMismatch("Corpus contains conflicting chunks with the same id")
            corpus_by_id.setdefault(chunk_id, chunk)

        for research_chunk in _get(research_context, "chunks"):
            research_payload = _as_dict(research_chunk)
            chunk_id = research_payload["id"]
            corpus_chunk = corpus_by_id.get(chunk_id)
            if corpus_chunk is None or _stable_chunk(contract_chunk(corpus_chunk)) != _stable_chunk(research_payload):
                raise ArtifactMismatch("Research context chunk does not match the corpus artifact")
            run_log["research_chunk_ids"].append(chunk_id)

        try:
            if hasattr(pipeline.corpus, "evidence_tokens"):
                inherited = assessment_evidence(pipeline.corpus, tech_assessment, pipeline.config)
            else:
                inherited = _fallback_cited_chunks(pipeline.corpus, tech_assessment)
        except Exception as exc:
            if isinstance(exc, RetrievalFailure):
                raise
            raise RetrievalFailure("Inherited research evidence could not be built") from exc

        candidate_lookup = {}
        _merge_chunks(candidate_lookup, inherited, ArtifactMismatch)
        run_log["inherited_chunk_ids"] = list(candidate_lookup)

        default_query = DEFAULT_WEB_QUERIES[view]
        question_query = " ".join(questions)
        run_log["queries"] = [default_query, question_query]
        try:
            default_hits = list(pipeline.corpus.web_search(default_query, 8) or [])
            question_hits = list(pipeline.corpus.web_search(question_query, 8) or [])
        except Exception as exc:
            raise RetrievalFailure(f"Web search failed for {view}") from exc

        web_candidates = {}
        _merge_chunks(web_candidates, default_hits, ArtifactMismatch)
        _merge_chunks(web_candidates, question_hits, ArtifactMismatch)
        unseen_web = {}
        for chunk_id, chunk in web_candidates.items():
            inherited_chunk = candidate_lookup.get(chunk_id)
            if inherited_chunk is not None:
                if _stable_chunk(inherited_chunk) != _stable_chunk(chunk):
                    raise ArtifactMismatch("Web search conflicts with inherited research evidence")
                continue
            unseen_web[chunk_id] = chunk

        token_counter = getattr(pipeline.corpus, "evidence_tokens", lambda text: len(text.split()))
        web_budget = pipeline.config.get("perspective_web_token_budget") or 2000
        try:
            bounded_web = token_context(list(unseen_web.values()), [], token_counter, web_budget, 0)
        except Exception as exc:
            raise RetrievalFailure(f"Web context construction failed for {view}") from exc
        _merge_chunks(candidate_lookup, bounded_web, ArtifactMismatch)
        model_chunks = list(candidate_lookup.values())
        run_log["web_chunk_ids"] = [chunk["id"] for chunk in bounded_web]

        payload = {
            "domain": domain,
            "scenario": scenario,
            "questions": questions,
            "feedback": feedback,
            "tech_assessment": deepcopy(tech_assessment),
            "web_query": default_query,
            "chunks": model_chunks,
            "source_metadata": pipeline.metadata(),
        }
        try:
            assessment = pipeline.structured(
                view,
                Assessment,
                PERSPECTIVE_PROMPTS[view],
                payload,
                _perspective_check(model_chunks, view),
            )
        except StructuredValidationError:
            run_log["assessment_status"] = "invalid_response"
            raise

        assessment = _as_dict(assessment)
        run_log["assessment_status"] = assessment["status"]
        missing = _missing_facets(assessment)
        run_log["missing_facets"] = missing

        if missing:
            try:
                rewritten = pipeline.structured(
                    view + "_rewrite",
                    PerspectiveQueries,
                    PERSPECTIVE_REWRITE_PROMPT,
                    {
                        "missing_facets": missing,
                        "gaps": assessment["gaps"],
                        "previous_query": default_query,
                    },
                    _rewrite_check(missing),
                )
            except StructuredValidationError:
                run_log["assessment_status"] = "invalid_response"
                raise

            expanded_candidates = {}
            for query in rewritten["queries"]:
                query_text = query["query"]
                run_log["queries"].append(query_text)
                try:
                    found = list(pipeline.corpus.web_search(query_text, 6, expanded_only=True) or [])
                except Exception as exc:
                    raise RetrievalFailure(f"Expanded web search failed for {view}") from exc
                _merge_chunks(expanded_candidates, found, ArtifactMismatch)

            for chunk_id, chunk in expanded_candidates.items():
                existing = candidate_lookup.get(chunk_id)
                if existing is not None:
                    if _stable_chunk(existing) != _stable_chunk(chunk):
                        raise ArtifactMismatch("Expanded web search conflicts with an existing chunk")
                    continue

            expanded_only = [
                chunk for chunk_id, chunk in expanded_candidates.items()
                if chunk_id not in candidate_lookup
            ]
            expanded_budget = pipeline.config.get("perspective_expanded_token_budget") or 2000
            try:
                bounded_expanded = token_context(expanded_only, [], token_counter, expanded_budget, 0)
            except Exception as exc:
                raise RetrievalFailure(f"Expanded web context construction failed for {view}") from exc
            _merge_chunks(candidate_lookup, bounded_expanded, ArtifactMismatch)
            model_chunks = list(candidate_lookup.values())
            run_log["expanded_chunk_ids"] = [chunk["id"] for chunk in bounded_expanded]

            preserved = [
                deepcopy(claim) for claim in assessment["claims"]
                if claim["facet"] not in missing
            ]

            def check_reassessment(value):
                errors = validate_perspective(value, model_chunks, view)
                if (value["status"] == "insufficient" or
                        any(claim["kind"] == "unknown" for claim in value["claims"])) and not any(
                    gap.strip() for gap in value["gaps"]
                ):
                    errors.append("Insufficient or unknown claims require non-empty gaps")
                reassessed_preserved = [
                    claim for claim in value["claims"] if claim["facet"] not in missing
                ]
                if reassessed_preserved != preserved:
                    errors.append("Preserve sufficient facets verbatim; reassess missing facets only")
                return errors

            tech_without_diagnostics = {
                technology: {
                    key: deepcopy(value)
                    for key, value in result.items()
                    if key != "retrieval_diagnostics"
                }
                for technology, result in tech_assessment.items()
            }
            try:
                assessment = reassess_facets(
                    pipeline,
                    view + "_reassessment",
                    PERSPECTIVE_PROMPTS[view] + PERSPECTIVE_REASSESSMENT_PROMPT,
                    {
                        "previous_assessment": assessment,
                        "missing_facets": missing,
                        "chunks": model_chunks,
                        "source_metadata": pipeline.metadata(),
                        "tech_assessment": tech_without_diagnostics,
                    },
                    check_reassessment,
                    BASE,
                )
            except StructuredValidationError:
                run_log["assessment_status"] = "invalid_response"
                raise
            assessment = _as_dict(assessment)
            run_log["assessment_status"] = assessment["status"]

        balance_input = {
            **assessment,
            "searched_chunks": [deepcopy(chunk) for chunk in candidate_lookup.values()],
        }
        balanced_assessment, diagnostics = balance_findings(view, balance_input)
        assessment = _as_dict(balanced_assessment)
        balance_log = {
            "request_id": _get(request, "request_id"),
            "view": view,
            "diagnostics": diagnostics,
            "gaps": deepcopy(assessment["gaps"]),
        }
        run_log["balance"] = deepcopy(diagnostics)
        run_log["cited_chunk_ids"] = sorted(_cited_ids({view: assessment}))
        run_log["candidate_chunk_ids"] = list(candidate_lookup)
        return {view: assessment}, candidate_lookup
    except Exception as exc:
        run_log["error_type"] = type(exc).__name__
        raise
    finally:
        write_json(pipeline.out / "retrieval" / f"{view}.json", run_log)
        if balance_log is not None:
            balance_path = pipeline.out / "retrieval" / "balance.json"
            existing = {"views": {}}
            if balance_path.exists():
                existing = json.loads(balance_path.read_text())
                if not isinstance(existing, dict) or not isinstance(existing.get("views"), dict):
                    existing = {"views": {}}
            existing["views"][view] = balance_log
            write_json(balance_path, existing)


__all__ = ["run_followup"]
