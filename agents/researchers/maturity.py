"""Initial technology-maturity research runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from itertools import zip_longest

from rag.budget import write_json
from rag.context import build_research_context
from rag.evidence import CORE_FACETS, validate_assessment, validate_retrieval_review
from rag.schemas import Assessment, Queries, RetrievalReview
from rag.tracing import submit

from agents.researchers.prompts import (
    RESEARCH_ASSESSMENT_PROMPT,
    RESEARCH_PLAN_PROMPT,
    RETRIEVAL_REVIEW_PROMPT,
    RETRIEVAL_REWRITE_PROMPT,
)


FACETS = ("mechanism", "limitation", "conditions", "maturity")
TECHNOLOGIES = ("KIVI", "InfiniGen")


def _get(value, key):
    if isinstance(value, dict):
        return value[key]
    return getattr(value, key)


def _assessment(status, gaps=()):
    return Assessment.model_validate({
        "status": status,
        "claims": [],
        "conflicts": [],
        "gaps": list(gaps),
    }).model_dump()


def _review_check(chunks, technology):
    return lambda result: validate_retrieval_review(result, chunks, technology)


def _assessment_check(chunks, technology):
    def check(result):
        errors = validate_assessment(result, chunks, technology, core=True)
        if (result["status"] != "ok" or len(result["claims"]) != 4 or
                {claim["facet"] for claim in result["claims"]} != CORE_FACETS):
            errors.append("Four complete core facets with status ok required")
        return errors

    return check


def _plan_check(result):
    expected = {(technology, facet) for technology in TECHNOLOGIES for facet in FACETS}
    queries = result["queries"]
    actual = {(query["technology"], query["facet"]) for query in queries}
    if len(queries) == 8 and actual == expected:
        return []
    return ["Exactly one query is required for each technology/facet pair"]


def _rewrite_check(result, technology):
    queries = result["queries"]
    expected = {(technology, facet) for facet in FACETS}
    actual = {(query["technology"], query["facet"]) for query in queries}
    if len(queries) == 4 and actual == expected:
        return []
    return ["Rewrite must preserve the four facets and technology"]


def _gap_text(item):
    missing = ", ".join(item["missing"])
    reason = item["reason"]
    return f"{item['facet']}: {missing} (사유: {reason})"


def _merge_chunks(lookup, chunks):
    for chunk in chunks:
        chunk_id = chunk["id"]
        existing = lookup.get(chunk_id)
        if existing is not None and existing != chunk:
            raise ValueError(f"Conflicting duplicate source chunk: {chunk_id}")
        lookup[chunk_id] = deepcopy(chunk)


def _balance(technology, assessment, chunks):
    from agents.researchers.result import balance_findings

    return balance_findings(
        "research",
        {
            "assessment": assessment,
            "technology": technology,
            "searched_chunks": chunks,
        },
    )


def _technology_research(pipeline, technology, initial_queries, feedback):
    """Search, review and assess one technology with at most one rewrite."""
    from rag.graph import StructuredValidationError, compact_chunks
    from agents.researchers.result import RetrievalFailure

    current_queries = deepcopy(initial_queries)
    diagnostics = []
    query_log = []

    for retrieval_attempt in range(2):
        groups = []
        round_log = {
            "attempt": retrieval_attempt,
            "queries": [],
            "hits": {},
        }
        diagnostics.append(round_log)
        try:
            for query in current_queries:
                round_log["queries"].append(deepcopy(query))
                query_log.append({**deepcopy(query), "attempt": retrieval_attempt})
                try:
                    with pipeline.search_lock:
                        hits = pipeline.corpus.search(
                            query["query"], technology, pipeline.config["top_k"]
                        )
                    hits = list(hits or [])
                    hit_ids = [hit["id"] for hit in hits]
                except Exception as exc:
                    raise RetrievalFailure(
                        f"Paper search failed for {technology} {query['facet']}"
                    ) from exc
                groups.append(hits)
                round_log["hits"][query["facet"]] = hit_ids

            try:
                ranked = [hit for row in zip_longest(*groups) for hit in row if hit]
                if hasattr(pipeline.corpus, "evidence_tokens"):
                    chunks = build_research_context(
                        pipeline.corpus, ranked, technology, pipeline.config
                    )
                else:
                    chunks = compact_chunks(ranked, 10500)
            except Exception as exc:
                raise RetrievalFailure(
                    f"Research context construction failed for {technology}"
                ) from exc

            round_log["candidate_chunk_ids"] = [chunk["id"] for chunk in chunks]
            if not ranked:
                query_by_facet = {query["facet"]: query for query in current_queries}
                gaps = [
                    f"검색 결과 없음: {facet} 질의 {query_by_facet[facet]['query']}"
                    for facet in FACETS
                ]
                assessment, balance = _balance(technology, _assessment("insufficient", gaps), chunks)
                return assessment, chunks, diagnostics, query_log, balance

            review = None
            reasons = None
            try:
                review = pipeline.structured(
                    f"retrieval_review_{technology.lower()}_{retrieval_attempt}",
                    RetrievalReview,
                    RETRIEVAL_REVIEW_PROMPT,
                    {
                        "technology": technology,
                        "queries": current_queries,
                        "chunks": chunks,
                        "source_metadata": pipeline.metadata(),
                        "feedback": feedback,
                    },
                    _review_check(chunks, technology),
                )
                round_log["review"] = review
                deficient = [item for item in review["items"] if not item["sufficient"]]
                if deficient:
                    reasons = [
                        {"facet": item["facet"], "reason": item["reason"], "missing": item["missing"]}
                        for item in deficient
                    ]
                else:
                    assessment = pipeline.structured(
                        f"research_{technology.lower()}_{retrieval_attempt}",
                        Assessment,
                        RESEARCH_ASSESSMENT_PROMPT,
                        {
                            "technology": technology,
                            "retrieval_attempt": retrieval_attempt,
                            "retrieval_review": review,
                            "chunks": chunks,
                            "source_metadata": pipeline.metadata(),
                            "feedback": feedback,
                        },
                        _assessment_check(chunks, technology),
                    )
                    round_log["status"] = "accepted"
                    assessment, balance = _balance(technology, assessment, chunks)
                    return assessment, chunks, diagnostics, query_log, balance
            except StructuredValidationError as exc:
                round_log.update(
                    status="invalid_response",
                    errors=deepcopy(getattr(exc, "errors", ["Invalid structured response"])),
                )
                reasons = deepcopy(getattr(exc, "errors", ["Invalid structured response"]))
                if retrieval_attempt:
                    raise

            if reasons is None:
                raise RuntimeError("Research review produced no decision")

            round_log["status"] = "insufficient_or_invalid"
            round_log["missing_facets"] = (
                [item["facet"] for item in review["items"] if not item["sufficient"]]
                if review is not None else []
            )
            round_log["errors"] = deepcopy(reasons)

            if retrieval_attempt:
                if review is None:
                    raise RuntimeError("Final retrieval review was invalid")
                assessment, balance = _balance(
                    technology,
                    _assessment("insufficient", [_gap_text(item) for item in deficient]),
                    chunks,
                )
                return assessment, chunks, diagnostics, query_log, balance

            rewrite = pipeline.structured(
                f"rewrite_{technology.lower()}",
                Queries,
                RETRIEVAL_REWRITE_PROMPT,
                {
                    "technology": technology,
                    "queries": current_queries,
                    "missing_reasons": reasons,
                    "previous_hits": groups,
                    "previous_queries": deepcopy(query_log),
                    "retrieval_review": review,
                    "feedback": feedback,
                },
                lambda result: _rewrite_check(result, technology),
            )
            current_queries = rewrite["queries"]
        except RetrievalFailure as exc:
            round_log.setdefault("status", "failed")
            exc.diagnostics = diagnostics
            raise
        except Exception as exc:
            round_log.setdefault("status", "failed")
            exc.diagnostics = diagnostics
            raise

    raise RuntimeError("Technology research exceeded its retrieval-round limit")


def run_maturity(pipeline, request):
    """Run the initial ``research`` view and return assessments plus a chunk lookup."""
    context = _get(request, "context")
    domain = _get(context, "domain")
    scenario = _get(context, "scenario")
    technologies = list(_get(context, "technologies"))
    questions = list(_get(request, "questions"))
    feedback = list(_get(request, "feedback"))

    query_log = []
    run_log = {
        "request_id": _get(request, "request_id"),
        "view": "research",
        "query_plan": None,
        "technologies": {},
        "technology_queries": {},
        "balance_diagnostics": {},
    }
    try:
        query_plan = pipeline.structured(
            "research_queries",
            Queries,
            RESEARCH_PLAN_PROMPT,
            {
                "domain": domain,
                "scenario": scenario,
                "technologies": technologies,
                "questions": questions,
            },
            _plan_check,
        )
        run_log["query_plan"] = query_plan

        per_technology = {
            technology: [
                query for query in query_plan["queries"]
                if query["technology"] == technology
            ]
            for technology in technologies
        }
        workers = min(2, max(1, pipeline.settings.integer("RAG_MAX_CONCURRENCY", 1)))
        outcomes = {}
        failures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                submit(
                    pool,
                    _technology_research,
                    pipeline,
                    technology,
                    per_technology[technology],
                    feedback,
                ): technology
                for technology in technologies
            }
            for future in as_completed(futures):
                technology = futures[future]
                try:
                    assessment, chunks, diagnostics, technology_queries, balance = future.result()
                    outcomes[technology] = (assessment, chunks, technology_queries)
                    run_log["technologies"][technology] = diagnostics
                    run_log["technology_queries"][technology] = technology_queries
                    run_log["balance_diagnostics"][technology] = balance
                except Exception as exc:
                    failures[technology] = exc
                    technology_diagnostics = getattr(exc, "diagnostics", [])
                    run_log["technologies"][technology] = technology_diagnostics
                    technology_queries = [
                        {**deepcopy(query), "attempt": item["attempt"]}
                        for item in technology_diagnostics
                        for query in item.get("queries", [])
                    ]
                    run_log["technology_queries"][technology] = technology_queries
                    query_log.extend(technology_queries)
                    run_log["technologies"].setdefault(technology, []).append(
                        {"status": "failed", "error_type": type(exc).__name__}
                    )

        # Futures are all drained before choosing an error; use contract technology order.
        for technology in technologies:
            if technology in failures:
                raise failures[technology]

        assessments = {}
        chunk_lookup = {}
        for technology in technologies:
            assessment, chunks, technology_queries = outcomes[technology]
            query_log.extend(technology_queries)
            key = f"research_{technology.lower()}"
            assessments[key] = assessment
            _merge_chunks(chunk_lookup, chunks)
        return assessments, chunk_lookup
    except Exception as exc:
        run_log["error_type"] = type(exc).__name__
        raise
    finally:
        run_log["queries"] = query_log
        write_json(pipeline.out / "retrieval" / "research.json", run_log)
        write_json(
            pipeline.out / "balance.json",
            {
                "view": "research",
                "technologies": {
                    technology: diagnostic["technologies"].get(technology, {})
                    for technology, diagnostic in run_log["balance_diagnostics"].items()
                },
            },
        )


__all__ = ["run_maturity"]
