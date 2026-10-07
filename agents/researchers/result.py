"""Conversions and citation artifact selection for research results."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from agents.researchers.contract import (
    Chunk,
    NodeError,
    ResearchRequest,
    ResearchResult,
    Source,
)
from rag.budget import BudgetExceeded
from rag.llm import APIError
from rag.request_budget import InputBudgetExceeded
from rag.schemas import Assessment


class RetrievalFailure(RuntimeError):
    """Raised when search or retrieval-context construction fails."""


class ArtifactMismatch(ValueError):
    """Raised when an artifact identity or cited-content invariant is violated."""


_ERROR_REASONS = {
    "retrieval_error": "검색 자료를 가져오지 못했습니다",
    "invalid_response": "구조화된 조사 결과를 검증하지 못했습니다",
    "api_error": "모델 호출에 실패했습니다",
    "budget_exceeded": "요청 예산 한도를 넘었습니다",
    "input_budget_exceeded": "입력 토큰 한도를 넘었습니다",
    "uncertain_request": "요청 완료 여부를 확인할 수 없습니다",
    "artifact_mismatch": "주장·청크·출처의 연결을 검증하지 못했습니다",
}


def _as_dict(value: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    """Return an independent mapping for either corpus dictionaries or models."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    raise TypeError(f"Expected a mapping or Pydantic model, got {type(value).__name__}")


def contract_chunk(chunk: Mapping[str, Any] | BaseModel) -> Chunk:
    """Convert a corpus chunk without changing source text or collection metadata."""
    payload = _as_dict(chunk)
    if payload.get("page") is None:
        payload["page"] = None
    else:
        # Corpus PDF chunks also carry a short section preview. The contract uses
        # physical page numbers for PDFs and reserves section for web snapshots.
        payload["section"] = None
    return Chunk.model_validate(payload)


def contract_source(source: Mapping[str, Any] | BaseModel) -> Source:
    """Convert corpus metadata to the public source shape with explicit fallbacks."""
    raw = _as_dict(source)
    accessed_at = raw.get("accessed_at")
    if not isinstance(accessed_at, str) or not accessed_at.strip():
        raise ValueError("Source is missing accessed_at collection time")

    authors = raw.get("authors")
    if not isinstance(authors, str) or not authors.strip():
        authors = "저자 미표기"

    version = raw.get("version")
    if not isinstance(version, str) or not version.strip():
        version = "unknown"

    # Keep provenance hashes while omitting local paths, cached body text, and
    # other corpus-internal fields from the model-facing source record.
    payload = {
        key: deepcopy(value)
        for key, value in raw.items()
        if "hash" in key.lower() or "sha" in key.lower()
    }
    payload.update(
        type=raw.get("type"),
        authors=authors,
        title=raw.get("title"),
        url=raw.get("url"),
        version=version,
        date=raw.get("date"),
        accessed_at=accessed_at,
    )
    return Source.model_validate(payload)


def _assessment_dict(assessment: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    return _as_dict(assessment)


def cited_artifacts(
    assessments: Mapping[str, Mapping[str, Any] | BaseModel],
    chunk_lookup: Mapping[str, Mapping[str, Any] | BaseModel],
    source_registry: Mapping[str, Mapping[str, Any] | BaseModel],
) -> tuple[list[Chunk], dict[str, Source]]:
    """Select each cited chunk and its source once, in first-citation order."""
    chunk_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    for assessment in assessments.values():
        for claim in _assessment_dict(assessment).get("claims", []):
            for reference in claim.get("references", []):
                chunk_id = reference["chunk_id"]
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    chunk_ids.append(chunk_id)

    chunks: list[Chunk] = []
    sources: dict[str, Source] = {}
    for chunk_id in chunk_ids:
        if chunk_id not in chunk_lookup:
            raise ValueError(f"Claim references missing chunk: {chunk_id}")
        chunk = contract_chunk(chunk_lookup[chunk_id])
        if chunk.id != chunk_id:
            raise ValueError(f"Chunk lookup key does not match chunk id: {chunk_id}")
        chunks.append(chunk)

        if chunk.source_id not in source_registry:
            raise ValueError(f"Chunk references missing source: {chunk.source_id}")
        if chunk.source_id not in sources:
            sources[chunk.source_id] = contract_source(source_registry[chunk.source_id])

    return chunks, sources


def _normalized(text: str) -> str:
    return " ".join(text.split())


def trace_errors(result: Mapping[str, Any] | BaseModel) -> list[str]:
    """Report broken or unused claim -> chunk -> source links in a result."""
    payload = _as_dict(result)
    chunks = [_as_dict(chunk) for chunk in payload.get("chunks", [])]
    sources = {
        source_id: _as_dict(source)
        for source_id, source in payload.get("sources", {}).items()
    }
    errors: list[str] = []

    chunks_by_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = chunk["id"]
        if chunk_id in chunks_by_id:
            errors.append(f"duplicate chunk id: {chunk_id}")
        else:
            chunks_by_id[chunk_id] = chunk

    cited_chunk_ids: set[str] = set()
    assessments = payload.get("assessments", {})
    for assessment in assessments.values():
        for claim_index, claim in enumerate(assessment.get("claims", [])):
            for reference_index, reference in enumerate(claim.get("references", [])):
                chunk_id = reference["chunk_id"]
                cited_chunk_ids.add(chunk_id)
                chunk = chunks_by_id.get(chunk_id)
                where = f"claim {claim_index} reference {reference_index} ({chunk_id})"
                if chunk is None:
                    errors.append(f"{where}: reference points to missing chunk")
                    continue

                quote = _normalized(reference["quote"])
                if len(quote) < 12:
                    errors.append(f"{where}: normalized quote is shorter than 12 characters")
                elif quote not in _normalized(chunk["text"]):
                    errors.append(f"{where}: quote not found in chunk")

    cited_source_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = chunk["id"]
        source_id = chunk["source_id"]
        if chunk_id not in cited_chunk_ids:
            errors.append(f"chunk {chunk_id}: chunk is not cited")
        if source_id not in sources:
            errors.append(f"chunk {chunk_id}: missing source {source_id}")
            continue

        source = sources[source_id]
        if chunk_id in cited_chunk_ids:
            cited_source_ids.add(source_id)

        if source.get("type") == "paper_pool":
            page = chunk.get("page")
            if type(page) is not int or page < 1:
                errors.append(f"chunk {chunk_id}: paper_pool chunk must have an integer page >= 1")
            if chunk.get("section") is not None:
                errors.append(f"chunk {chunk_id}: paper_pool chunk must have section=null")
        elif source.get("type") == "external_web":
            section = chunk.get("section")
            if chunk.get("page") is not None:
                errors.append(f"chunk {chunk_id}: external_web chunk must have page=null")
            if not isinstance(section, str) or not section.strip():
                errors.append(f"chunk {chunk_id}: external_web chunk must have a non-blank section")
        else:
            errors.append(f"chunk {chunk_id}: source has an unsupported type")

    for source_id in sources:
        if source_id not in cited_source_ids:
            errors.append(f"source {source_id}: unused source")

    return errors


def node_error(exc: Exception, stage: str) -> NodeError:
    """Map known execution failures to a safe, short contract error."""
    # Import lazily because rag.graph will import the research-agent boundary.
    from rag.graph import StructuredValidationError

    if isinstance(exc, RetrievalFailure):
        code = "retrieval_error"
    elif isinstance(exc, InputBudgetExceeded):
        code = "input_budget_exceeded"
    elif isinstance(exc, BudgetExceeded):
        code = "budget_exceeded"
    elif isinstance(exc, APIError):
        code = "uncertain_request" if "reservation retained" in str(exc).lower() else "api_error"
    elif isinstance(exc, (StructuredValidationError, ValidationError, json.JSONDecodeError)):
        code = "invalid_response"
    elif isinstance(exc, ArtifactMismatch):
        code = "artifact_mismatch"
    elif isinstance(exc, ValueError) and "Conflicting duplicate source chunk" in str(exc):
        code = "artifact_mismatch"
    elif isinstance(exc, ValueError):
        code = "invalid_response"
    else:
        raise exc

    stage_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage).strip())[:64] or "research"
    message = f"{stage_label}: {type(exc).__name__}: {_ERROR_REASONS[code]}"[:200]
    return NodeError(code=code, message=message, retryable=False)


def _request_model(request: Mapping[str, Any] | BaseModel) -> ResearchRequest:
    if isinstance(request, ResearchRequest):
        return request
    return ResearchRequest.model_validate(_as_dict(request))


def failed_result(
    request: Mapping[str, Any] | BaseModel,
    error: NodeError | Mapping[str, Any],
) -> ResearchResult:
    """Create the contract's empty failed shape while echoing its request envelope."""
    request_model = _request_model(request)
    error_model = error if isinstance(error, NodeError) else NodeError.model_validate(_as_dict(error))
    envelope = request_model.model_dump(
        mode="python",
        include={"contract_version", "run_id", "request_id", "attempt", "context"},
    )
    return ResearchResult.model_validate(
        {
            **envelope,
            "view": request_model.view,
            "status": "failed",
            "assessments": {},
            "chunks": [],
            "sources": {},
            "error": error_model,
        }
    )


def build_result(
    request: Mapping[str, Any] | BaseModel,
    assessments: Mapping[str, Mapping[str, Any] | BaseModel],
    chunk_lookup: Mapping[str, Mapping[str, Any] | BaseModel],
    source_registry: Mapping[str, Mapping[str, Any] | BaseModel],
) -> ResearchResult:
    """Assemble only cited artifacts and fail closed on any contract violation."""
    request_model = _request_model(request)
    try:
        assessment_models = {
            name: Assessment.model_validate(_assessment_dict(value))
            for name, value in assessments.items()
        }
        chunks, sources = cited_artifacts(assessment_models, chunk_lookup, source_registry)
        status = "ok" if all(value.status == "ok" for value in assessment_models.values()) else "insufficient"
        envelope = request_model.model_dump(
            mode="python",
            include={"contract_version", "run_id", "request_id", "attempt", "context"},
        )
        result = ResearchResult.model_validate(
            {
                **envelope,
                "view": request_model.view,
                "status": status,
                "assessments": assessment_models,
                "chunks": chunks,
                "sources": sources,
                "error": None,
            }
        )
        errors = trace_errors(result)
        if errors:
            raise ArtifactMismatch("; ".join(errors))
        return result
    except (KeyError, ValidationError, ValueError) as exc:
        # Keep the full diagnostic in caller-owned local logs; only the safe
        # classification crosses the ResearchResult boundary.
        error = node_error(ArtifactMismatch(str(exc)), "result_assembly")
        return failed_result(request_model, error)


def balance_findings(
    view: str,
    assessment: Mapping[str, Any] | BaseModel,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add deterministic caveat/counter-facet gaps and local source-balance counts."""
    counter_facets = {
        "research": "limitation",
        "market": "costs",
        "stakeholder": None,
        "domain": "risks",
    }
    if view not in counter_facets:
        raise ValueError(f"Unknown research view: {view}")

    local = _as_dict(assessment)
    local_chunks = []
    for key in ("chunks", "searched_chunks"):
        values = local.pop(key, None)
        if isinstance(values, list):
            local_chunks.extend(values)
    technology_hint = local.pop("technology", None)
    if "assessment" in local and "claims" not in local:
        assessment_value = local.pop("assessment")
        payload = _assessment_dict(assessment_value)
    else:
        payload = local

    value = Assessment.model_validate(payload)
    assessment_payload = value.model_dump(mode="python")
    claims = assessment_payload["claims"]
    claim_technologies = {
        claim["technology"]
        for claim in claims
        if claim["technology"] in {"KIVI", "InfiniGen", "both"}
    }
    if view == "research":
        if technology_hint in {"KIVI", "InfiniGen"}:
            technologies = [technology_hint]
        else:
            technologies = [
                tech for tech in ("KIVI", "InfiniGen")
                if tech in claim_technologies or "both" in claim_technologies
            ]
    else:
        technologies = ["KIVI", "InfiniGen"]

    findings = []
    counter_facet = counter_facets[view]
    for technology in technologies:
        tech_claims = [
            claim for claim in claims
            if claim["technology"] in {technology, "both"}
        ]
        if not any(
            isinstance(claim["caveats"], str) and claim["caveats"].strip()
            for claim in tech_claims
        ):
            findings.append(f"{technology}: 한계·반대 근거를 기재한 주장이 없음")

        if counter_facet:
            counter_claims = [claim for claim in tech_claims if claim["facet"] == counter_facet]
            if not counter_claims or all(claim["kind"] == "unknown" for claim in counter_claims):
                findings.append(f"{technology} {counter_facet}: 한계·위험 근거 미확인")

    assessment_payload["gaps"] = list(dict.fromkeys(assessment_payload["gaps"] + findings))
    if findings:
        assessment_payload["status"] = "insufficient"
    updated = Assessment.model_validate(assessment_payload).model_dump(mode="python")

    chunk_sources = {}
    for chunk in local_chunks:
        item = _as_dict(chunk)
        chunk_id, source_id = item.get("id"), item.get("source_id")
        if isinstance(chunk_id, str) and isinstance(source_id, str) and source_id.strip():
            chunk_sources[chunk_id] = source_id

    source_counts = {tech: {} for tech in technologies}
    for claim in claims:
        if claim["technology"] == "both":
            claim_technologies_for_counts = technologies
        elif claim["technology"] in source_counts:
            claim_technologies_for_counts = [claim["technology"]]
        else:
            claim_technologies_for_counts = []
        for reference in claim["references"]:
            source_id = chunk_sources.get(reference["chunk_id"])
            if source_id is None:
                continue
            for technology in claim_technologies_for_counts:
                counts = source_counts[technology]
                counts[source_id] = counts.get(source_id, 0) + 1

    diagnostics = {
        "view": view,
        "technologies": {},
    }
    for technology, counts in source_counts.items():
        total = sum(counts.values())
        diagnostics["technologies"][technology] = {
            "distinct_sources": len(counts),
            "max_source_share": max(counts.values(), default=0) / total if total else 0.0,
        }

    return updated, diagnostics
