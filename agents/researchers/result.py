"""Conversions and citation artifact selection for research results."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from pydantic import BaseModel

from agents.researchers.contract import Chunk, Source


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
