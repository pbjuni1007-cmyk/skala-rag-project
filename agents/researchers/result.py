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
