"""Public request boundary for the four research views."""

from __future__ import annotations

from copy import copy
import os
from pathlib import Path
import re
from typing import Any

from rag.budget import write_json

from agents.researchers.contract import parse_request
from agents.researchers.followup import run_followup
from agents.researchers.maturity import run_maturity
from agents.researchers.result import build_result, failed_result, node_error


_SAFE_REQUEST_ID = re.compile(r"[^A-Za-z0-9._-]")


def _safe_request_id(request_id: str) -> str:
    """Make a request id safe as one directory component."""
    value = _SAFE_REQUEST_ID.sub("_", request_id)
    if value in {"", ".", ".."}:
        return "_"
    return value


class ResearchAgent:
    """Run exactly one research view through an isolated pipeline output path."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def research(self, request: Any) -> dict[str, Any]:
        """Validate, dispatch, assemble, and persist one contract result."""
        parsed = parse_request(request)
        request_id = _safe_request_id(parsed.request_id)

        request_pipeline = copy(self.pipeline)
        request_dir = Path(self.pipeline.out) / "research" / parsed.view / request_id
        result_path = request_dir / "result.json"
        error_path = request_dir / "error.json"
        claim_path = request_dir / ".in_progress"

        if result_path.exists():
            raise ValueError("Research request already has a result")

        request_dir.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ValueError("Research request is already in progress or was interrupted") from exc
        else:
            os.close(descriptor)

        request_pipeline.out = request_dir
        result_saved = False
        try:
            runner = run_maturity if parsed.view == "research" else run_followup
            try:
                assessments, chunk_lookup = runner(request_pipeline, parsed)
                source_registry = getattr(request_pipeline.corpus, "sources", {})
                result = build_result(parsed, assessments, chunk_lookup, source_registry)
            except Exception as exc:
                error = node_error(exc, "research")
                result = failed_result(parsed, error)

            payload = result.model_dump(mode="json")
            if result.status == "failed":
                write_json(error_path, payload["error"])
            write_json(result_path, payload)
            result_saved = True
            return payload
        except Exception as exc:
            # Preserve only a safe class name for failures that cannot be mapped
            # into the contract. The request marker prevents accidental replay.
            if not result_saved:
                write_json(error_path, {"error_type": type(exc).__name__})
            raise
        finally:
            if result_saved:
                claim_path.unlink(missing_ok=True)


__all__ = ["ResearchAgent"]
