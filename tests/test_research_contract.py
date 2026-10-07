"""Contract validation tests use only the published mock payloads."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from agents.researchers.contract import (
    ResearchRequest,
    ResearchResult,
    parse_request,
    required_assessment_keys,
)

EXAMPLES = json.loads(Path("docs/agent-contract-examples.json").read_text())["examples"]
CONTRACT_EXAMPLES = [
    (name, item["type"])
    for name, item in EXAMPLES.items()
    if item["type"] in {"ResearchRequest", "ResearchResult"}
]


def payload(name):
    return deepcopy(EXAMPLES[name]["payload"])


def set_path(value, path, replacement):
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = deepcopy(replacement)


def delete_path(value, path):
    target = value
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]


def insufficient_research_result():
    result = payload("research_ok")
    result["status"] = "insufficient"
    result["assessments"]["research_kivi"]["status"] = "insufficient"
    result["assessments"]["research_kivi"]["gaps"] = ["보완 근거가 필요하다."]
    return result


@pytest.mark.parametrize(("name", "kind"), CONTRACT_EXAMPLES, ids=[name for name, _ in CONTRACT_EXAMPLES])
def test_contract_examples_validate_and_round_trip(name, kind):
    value = payload(name)
    model = parse_request(value) if kind == "ResearchRequest" else ResearchResult.model_validate(value)

    assert model.model_dump(mode="json") == value
    if kind == "ResearchRequest":
        assert parse_request(model) == model


REQUEST_REJECTIONS = [
    ("unsupported contract version", "research_request", ("contract_version",), "agent-contract-v2"),
    ("blank run id", "research_request", ("run_id",), " \t"),
    ("blank request id", "research_request", ("request_id",), ""),
    ("zero attempt", "research_request", ("attempt",), 0),
    ("boolean attempt", "research_request", ("attempt",), True),
    ("unknown view", "research_request", ("view",), "evaluation"),
    ("wrong technology order", "research_request", ("context", "technologies"), ["InfiniGen", "KIVI"]),
    ("missing technology", "research_request", ("context", "technologies"), ["KIVI"]),
    ("blank domain", "research_request", ("context", "domain"), "  "),
    ("blank scenario", "research_request", ("context", "scenario"), ""),
    ("empty questions", "research_request", ("questions",), []),
    ("blank question", "research_request", ("questions",), ["   "]),
    ("research context on research view", "research_request", ("research_context",), payload("research_ok")),
    ("missing research context for other view", "market_request", ("research_context",), None),
    ("research context with wrong view", "market_request", ("research_context",), payload("market_ok")),
    (
        "research context with non-ok status",
        "market_request",
        ("research_context",),
        insufficient_research_result(),
    ),
    ("nested research run id mismatch", "market_request", ("research_context", "run_id"), "another-run"),
    (
        "nested research context mismatch",
        "market_request",
        ("research_context", "context", "domain"),
        "다른 업무",
    ),
    (
        "nested result version mismatch",
        "market_request",
        ("research_context", "contract_version"),
        "agent-contract-v2",
    ),
    ("previous result view mismatch", "market_research_request", ("view",), "domain"),
    (
        "previous result request id reused",
        "market_research_request",
        ("previous_result", "request_id"),
        "mock-contract-v1:market:2",
    ),
    ("previous result attempt is not lower", "market_research_request", ("attempt",), 1),
    (
        "previous result run id mismatch",
        "market_research_request",
        ("previous_result", "run_id"),
        "another-run",
    ),
    (
        "research context request id reused",
        "market_request",
        ("research_context", "request_id"),
        "mock-contract-v1:market:1",
    ),
    ("extra request field", "research_request", ("unexpected",), "value"),
]


@pytest.mark.parametrize(
    ("rule", "example", "path", "replacement"),
    REQUEST_REJECTIONS,
    ids=[case[0] for case in REQUEST_REJECTIONS],
)
def test_request_rejects_contract_violations(rule, example, path, replacement):
    value = payload(example)
    set_path(value, path, replacement)

    with pytest.raises(ValueError):
        parse_request(value)


RESULT_REJECTIONS = [
    ("unknown view", "market_ok", ("view",), "evaluation"),
    ("failed assessment is not empty", "market_failed", ("assessments",), payload("market_ok")["assessments"]),
    ("failed chunks are not empty", "market_failed", ("chunks",), payload("market_ok")["chunks"]),
    ("failed sources are not empty", "market_failed", ("sources",), payload("market_ok")["sources"]),
    ("failed error is missing", "market_failed", ("error",), None),
    ("ok has an error", "market_ok", ("error",), payload("market_failed")["error"]),
    (
        "ok contains an insufficient assessment",
        "market_ok",
        ("assessments", "market", "status"),
        "insufficient",
    ),
    ("insufficient has an error", "market_insufficient", ("error",), payload("market_failed")["error"]),
    ("insufficient has no insufficient assessment", "market_ok", ("status",), "insufficient"),
    ("insufficient assessment has no gaps", "market_insufficient", ("assessments", "market", "gaps"), []),
    (
        "insufficient assessment has blank gap",
        "market_insufficient",
        ("assessments", "market", "gaps"),
        ["  "],
    ),
    ("assessment keys do not match view", "market_ok", ("assessments", "unexpected"), payload("market_ok")["assessments"]["market"]),
    ("unknown NodeError code", "market_failed", ("error", "code"), "unknown_error"),
    ("Assessment carries retrieval diagnostics", "market_ok", ("assessments", "market", "retrieval_diagnostics"), []),
    ("extra result field", "market_ok", ("unexpected",), "value"),
]


@pytest.mark.parametrize(
    ("rule", "example", "path", "replacement"),
    RESULT_REJECTIONS,
    ids=[case[0] for case in RESULT_REJECTIONS],
)
def test_result_rejects_contract_violations(rule, example, path, replacement):
    value = payload(example)
    set_path(value, path, replacement)

    with pytest.raises(ValueError):
        ResearchResult.model_validate(value)


MISSING_FIELDS = [
    ("request", "research_request", ("previous_result",)),
    ("request", "market_request", ("research_context",)),
    ("result", "market_ok", ("error",)),
    ("result", "market_ok", ("chunks", 0, "page")),
    ("result", "market_ok", ("chunks", 0, "section")),
    ("result", "market_ok", ("sources", "mock-kivi", "date")),
]


@pytest.mark.parametrize(("kind", "example", "path"), MISSING_FIELDS)
def test_nullable_contract_fields_must_be_present(kind, example, path):
    value = payload(example)
    delete_path(value, path)

    with pytest.raises(ValueError):
        if kind == "request":
            parse_request(value)
        else:
            ResearchResult.model_validate(value)


def test_chunk_and_source_allow_collection_metadata():
    value = payload("market_ok")
    value["chunks"][0]["char_start"] = 10
    value["sources"]["mock-kivi"]["content_hash"] = "abc123"

    model = ResearchResult.model_validate(value)

    assert model.chunks[0].model_extra["char_start"] == 10
    assert model.sources["mock-kivi"].model_extra["content_hash"] == "abc123"
    assert model.model_dump(mode="json") == value


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        ("research", {"research_kivi", "research_infinigen"}),
        ("market", {"market"}),
        ("stakeholder", {"stakeholder"}),
        ("domain", {"domain"}),
    ],
)
def test_required_assessment_keys_per_view(view, expected):
    assert required_assessment_keys(view) == expected


def test_required_assessment_keys_rejects_unknown_view():
    with pytest.raises(ValueError):
        required_assessment_keys("evaluation")
