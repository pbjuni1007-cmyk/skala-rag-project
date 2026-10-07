"""Status, error mapping, and result assembly for the research-agent contract."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.researchers.contract import NodeError, ResearchRequest, ResearchResult
from agents.researchers.result import (
    ArtifactMismatch,
    RetrievalFailure,
    build_result,
    failed_result,
    node_error,
    trace_errors,
)
from rag.budget import BudgetExceeded
from rag.graph import StructuredValidationError
from rag.llm import APIError
from rag.request_budget import InputBudgetExceeded
from rag.schemas import Assessment


EXAMPLES = json.loads(Path("docs/agent-contract-examples.json").read_text(encoding="utf-8"))["examples"]


def payload(name):
    return deepcopy(EXAMPLES[name]["payload"])


def research_request():
    return ResearchRequest.model_validate(payload("research_request"))


def artifacts(name="research_ok"):
    value = payload(name)
    chunks = {chunk["id"]: chunk for chunk in value["chunks"]}
    return chunks, value["sources"]


def invalid_assessment_error():
    try:
        Assessment.model_validate({"status": "failed"})
    except ValidationError as exc:
        return exc
    raise AssertionError("fixture must produce a Pydantic ValidationError")


ERROR_CASES = [
    ("retrieval failure", lambda: RetrievalFailure("index unavailable"), "retrieval_error"),
    (
        "input budget is classified before ValueError",
        lambda: InputBudgetExceeded("assessment", 100, 50),
        "input_budget_exceeded",
    ),
    ("project budget", lambda: BudgetExceeded("budget limit"), "budget_exceeded"),
    (
        "uncertain reservation",
        lambda: APIError("request status unknown; reservation retained"),
        "uncertain_request",
    ),
    ("API error", lambda: APIError("generation failed"), "api_error"),
    (
        "structured validation",
        lambda: StructuredValidationError("assessment", ["invalid claim"]),
        "invalid_response",
    ),
    ("Pydantic validation", invalid_assessment_error, "invalid_response"),
    (
        "JSON decoding",
        lambda: json.JSONDecodeError("invalid JSON", "{", 0),
        "invalid_response",
    ),
    ("artifact mismatch exception", lambda: ArtifactMismatch("conflicting artifacts"), "artifact_mismatch"),
    (
        "conflicting duplicate source chunk",
        lambda: ValueError("Conflicting duplicate source chunk: chunk-1"),
        "artifact_mismatch",
    ),
    ("other value error", lambda: ValueError("assessment validation failed"), "invalid_response"),
]


@pytest.mark.parametrize("case, factory, code", ERROR_CASES, ids=[case[0] for case in ERROR_CASES])
def test_node_error_maps_known_failures_to_contract_codes(case, factory, code):
    exc = factory()

    error = node_error(exc, "assessment")

    assert isinstance(error, NodeError), case
    assert error.code == code
    assert "assessment" in error.message
    assert type(exc).__name__ in error.message
    assert len(error.message) <= 200
    assert error.retryable is False


def test_node_error_message_does_not_expose_key_or_source_text():
    secret_key = "sk-test-secret-key"
    source_quote = "PRIVATE-SOURCE-QUOTE-4815"
    exc = APIError(f"key {secret_key} rejected; response included {source_quote}; " + "x" * 400)

    error = node_error(exc, "generation")

    assert error.code == "api_error"
    assert "generation" in error.message
    assert "APIError" in error.message
    assert secret_key not in error.message
    assert source_quote not in error.message
    assert len(error.message) <= 200


def test_node_error_propagates_unclassified_exceptions():
    with pytest.raises(TypeError, match="implementation bug"):
        node_error(TypeError("implementation bug"), "assembly")


def test_build_result_assembles_ok_result_from_only_cited_artifacts():
    source_result = payload("research_ok")
    chunk_lookup, source_registry = artifacts()

    result = build_result(
        research_request(),
        source_result["assessments"],
        chunk_lookup,
        source_registry,
    )

    assert isinstance(result, ResearchResult)
    assert result.status == "ok"
    assert set(result.assessments) == {"research_kivi", "research_infinigen"}
    cited_ids = {
        reference["chunk_id"]
        for assessment in source_result["assessments"].values()
        for claim in assessment["claims"]
        for reference in claim["references"]
    }
    assert {chunk.id for chunk in result.chunks} == cited_ids
    assert {chunk.source_id for chunk in result.chunks} == set(result.sources)
    assert trace_errors(result) == []


def test_build_result_keeps_no_search_hits_as_insufficient_with_a_gap():
    source_result = payload("research_ok")
    assessments = deepcopy(source_result["assessments"])
    assessments["research_kivi"] = {
        "status": "insufficient",
        "claims": [],
        "conflicts": [],
        "gaps": ["검색 결과 없음: limitation 질의에 대한 검색 결과가 없음"],
    }
    chunk_lookup, source_registry = artifacts()

    result = build_result(research_request(), assessments, chunk_lookup, source_registry)

    assert isinstance(result, ResearchResult)
    assert result.status == "insufficient"
    assert result.assessments["research_kivi"].claims == []
    assert result.assessments["research_kivi"].gaps == assessments["research_kivi"]["gaps"]
    assert {chunk.technology for chunk in result.chunks} == {"InfiniGen"}
    assert trace_errors(result) == []


def test_failed_result_has_the_contract_failed_shape():
    request = research_request()
    error = NodeError(code="retrieval_error", message="search: RetrievalFailure: search unavailable", retryable=False)

    result = failed_result(request, error)

    assert isinstance(result, ResearchResult)
    assert result.status == "failed"
    assert result.assessments == {}
    assert result.chunks == []
    assert result.sources == {}
    assert result.error == error


def test_final_trace_violation_becomes_artifact_mismatch_failure():
    source_result = payload("research_ok")
    assessments = deepcopy(source_result["assessments"])
    assessments["research_kivi"]["claims"][0]["references"][0]["quote"] = "fabricated quotation absent from the supplied chunk"
    chunk_lookup, source_registry = artifacts()

    result = build_result(research_request(), assessments, chunk_lookup, source_registry)

    assert isinstance(result, ResearchResult)
    assert result.status == "failed"
    assert result.assessments == {}
    assert result.chunks == []
    assert result.sources == {}
    assert result.error is not None
    assert result.error.code == "artifact_mismatch"
