"""Offline checks against the published, fictional team integration examples."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents import contracts


EXAMPLES = json.loads(
    (Path(__file__).resolve().parents[1] / "docs/agent-contract-examples.json").read_text()
)["examples"]
COMMON = ("contract_version", "run_id", "request_id", "attempt", "context")


def example(name):
    return deepcopy(EXAMPLES[name]["payload"])


def response(name, request):
    """Echo the actual request identity without altering fictional evidence."""
    payload = example(name)
    payload.update({field: deepcopy(request[field]) for field in COMMON})
    return payload


def parse(name, payload):
    return getattr(contracts, EXAMPLES[name]["type"]).model_validate(payload)


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_all_published_examples_round_trip_without_losing_fields(name):
    payload = example(name)
    assert parse(name, payload).model_dump(mode="json") == payload


@pytest.mark.parametrize(
    "name,field",
    [
        ("research_request", "previous_result"),
        ("research_request", "research_context"),
        ("research_ok", "error"),
        ("write_request", "previous_report"),
        ("report_ok", "markdown"),
        ("report_ok", "report"),
        ("report_ok", "joined"),
        ("evaluation_pass", "error"),
        ("publish_failed", "markdown_ref"),
        ("publish_failed", "pdf_ref"),
        ("publish_failed", "pdf_pages"),
    ],
)
def test_nullable_fields_are_required_even_when_their_value_can_be_null(name, field):
    payload = example(name)
    del payload[field]
    with pytest.raises(ValidationError):
        parse(name, payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("contract_version", "agent-contract-v2"),
        ("run_id", ""),
        ("run_id", "   "),
        ("request_id", ""),
        ("attempt", 0),
        ("attempt", -1),
        ("attempt", True),
        ("attempt", 1.5),
        ("attempt", "1"),
    ],
)
def test_invalid_common_message_fields_are_rejected(field, value):
    payload = example("research_ok")
    payload[field] = value
    with pytest.raises(ValidationError):
        contracts.ResearchResult.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [("domain", ""), ("scenario", "  "), ("technologies", ["KIVI"]),
     ("technologies", ["KIVI", "KIVI"])],
)
def test_run_context_keeps_the_agreed_technology_pair_and_nonempty_scope(field, value):
    payload = example("research_request")["context"]
    payload[field] = value
    with pytest.raises(ValidationError):
        contracts.RunContext.model_validate(payload)


@pytest.mark.parametrize("damage", ["chunk", "source", "quote", "facet", "grounding"])
def test_research_rejects_unresolvable_or_private_evidence(damage):
    payload = example("research_ok")
    assessment = payload["assessments"]["research_kivi"]
    claim = assessment["claims"][0]
    if damage == "chunk":
        claim["references"][0]["chunk_id"] = "missing-chunk"
    elif damage == "source":
        del payload["sources"][payload["chunks"][0]["source_id"]]
    elif damage == "quote":
        claim["references"][0]["quote"] = "This sentence was never in the source."
    elif damage == "facet":
        assessment["claims"].pop()
    else:
        claim["grounding"] = {"subject": "private-contract-field"}
    with pytest.raises(ValidationError):
        contracts.ResearchResult.model_validate(payload)


def test_perspective_requires_all_facets_for_both_technologies():
    payload = example("market_ok")
    payload["assessments"]["market"]["claims"].pop()
    with pytest.raises(ValidationError):
        contracts.ResearchResult.model_validate(payload)


@pytest.mark.parametrize("damage", ["duplicate_chunk", "page_zero", "web_location", "version_null"])
def test_evidence_location_and_registry_are_not_silently_repaired(damage):
    payload = example("research_ok")
    if damage == "duplicate_chunk":
        duplicate = deepcopy(payload["chunks"][0])
        duplicate["text"] += " A different source payload."
        payload["chunks"].append(duplicate)
    elif damage == "page_zero":
        payload["chunks"][0]["page"] = 0
    elif damage == "web_location":
        payload["sources"][payload["chunks"][0]["source_id"]]["type"] = "external_web"
    else:
        payload["sources"][payload["chunks"][0]["source_id"]]["version"] = None
    with pytest.raises(ValidationError):
        contracts.ResearchResult.model_validate(payload)


def test_collection_metadata_survives_contract_serialization():
    payload = example("market_web_metadata")
    payload["chunks"][0]["text_sha256"] = "a" * 64
    next(iter(payload["sources"].values()))["raw_sha256"] = "b" * 64
    assert contracts.ResearchResult.model_validate(payload).model_dump(mode="json") == payload


@pytest.mark.parametrize("damage", ["status", "missing_gap", "error_in_ok", "failed_payload"])
def test_research_status_cannot_hide_missing_evidence_or_execution_failure(damage):
    payload = example("market_insufficient" if damage in {"status", "missing_gap"} else "market_ok")
    if damage == "status":
        payload["status"] = "ok"
    elif damage == "missing_gap":
        payload["assessments"]["market"]["gaps"] = []
    else:
        payload["error"] = example("market_failed")["error"]
        if damage == "failed_payload":
            payload["status"] = "failed"
    with pytest.raises(ValidationError):
        contracts.ResearchResult.model_validate(payload)


@pytest.mark.parametrize("damage", ["missing_view", "insufficient", "run_id", "context"])
def test_write_request_requires_current_successful_results_from_one_run(damage):
    payload = example("write_request")
    if damage == "missing_view":
        del payload["results"]["market"]
    elif damage == "insufficient":
        payload["results"]["market"] = example("market_insufficient")
    elif damage == "run_id":
        payload["results"]["market"]["run_id"] = "another-run"
    else:
        payload["results"]["market"]["context"]["scenario"] = "another scenario"
    with pytest.raises(ValidationError):
        contracts.WriteRequest.model_validate(payload)


@pytest.mark.parametrize("damage", ["missing_upstream", "wrong_view", "same_attempt", "different_run"])
def test_research_request_enforces_upstream_and_previous_result_identity(damage):
    payload = example("market_research_request")
    if damage == "missing_upstream":
        payload["research_context"] = None
    elif damage == "wrong_view":
        payload["previous_result"] = example("domain_ok")
    elif damage == "same_attempt":
        payload["previous_result"]["attempt"] = payload["attempt"]
    else:
        payload["research_context"]["run_id"] = "another-run"
    with pytest.raises(ValidationError):
        contracts.ResearchRequest.model_validate(payload)


@pytest.mark.parametrize("damage", ["claim_chunk", "claim_quote", "evidence_source", "evidence_id"])
def test_report_keeps_claims_and_evidence_resolvable(damage):
    payload = example("report_ok")
    claim = payload["joined"]["claims"]["research_kivi-1"]
    evidence = payload["joined"]["evidence"][claim["evidence_ids"][0]]
    if damage == "claim_chunk":
        claim["references"][0]["chunk_id"] = "missing-chunk"
    elif damage == "claim_quote":
        claim["references"][0]["quote"] = "A fabricated quote."
    elif damage == "evidence_source":
        evidence["source_id"] = "missing-source"
    else:
        claim["evidence_ids"] = ["missing-evidence"]
    with pytest.raises(ValidationError):
        contracts.ReportResult.model_validate(payload)


@pytest.mark.parametrize("damage", ["missing_check", "failed_check", "passed_with_repairs", "failed_without_repairs", "error_as_pass"])
def test_quality_verdict_requires_four_consistent_criteria(damage):
    payload = example("evaluation_pass")
    if damage == "missing_check":
        del payload["checks"]["coverage"]
    elif damage == "failed_check":
        payload["checks"]["coverage"]["passed"] = False
    elif damage == "passed_with_repairs":
        payload["repair_requests"] = example("evaluation_rework")["repair_requests"]
    elif damage == "failed_without_repairs":
        payload = example("evaluation_rework")
        payload["repair_requests"] = []
    else:
        payload = example("evaluation_error")
        payload["passed"] = True
    with pytest.raises(ValidationError):
        contracts.EvaluationResult.model_validate(payload)


@pytest.mark.parametrize("damage", ["stale_report", "different_run", "different_context", "quality_failure"])
def test_publication_requires_the_current_report_and_its_passing_verdict(damage):
    payload = example("publish_request")
    evaluation = payload["evaluation_result"]
    if damage == "stale_report":
        evaluation["report_request_id"] = "mock-contract-v1:writer:old"
    elif damage == "different_run":
        evaluation["run_id"] = "another-run"
    elif damage == "different_context":
        evaluation["context"]["domain"] = "another domain"
    else:
        payload["evaluation_result"] = example("evaluation_rework")
    with pytest.raises(ValidationError):
        contracts.PublishRequest.model_validate(payload)


@pytest.mark.parametrize("field,value", [("pdf_pages", 0), ("pdf_pages", 11),
                                        ("pdf_ref", None), ("human_review_pending", False)])
def test_publish_success_requires_bounded_outputs_and_human_review(field, value):
    payload = example("publish_ok")
    payload[field] = value
    with pytest.raises(ValidationError):
        contracts.PublishResult.model_validate(payload)
