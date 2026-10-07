import json
from pathlib import Path

import pytest

from agents.researchers.contract import ResearchRequest, ResearchResult
from agents.researchers.result import trace_errors
from rag.evidence import validate_assessment, validate_perspective
from scripts.export_research_samples import _insert_chunk

FIXTURES = Path(__file__).parent / "fixtures" / "research"
VIEWS = ("research", "market", "stakeholder", "domain")


def load_result(view):
    return json.loads((FIXTURES / f"{view}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("view", VIEWS)
def test_fixture_validates_and_citations_trace(view):
    payload = load_result(view)
    result = ResearchResult.model_validate(payload)
    assert trace_errors(result) == []

    if view == "research":
        for technology, key in (("KIVI", "research_kivi"), ("InfiniGen", "research_infinigen")):
            assert validate_assessment(
                payload["assessments"][key], payload["chunks"], technology=technology, core=True
            ) == []
    else:
        assert validate_perspective(payload["assessments"][view], payload["chunks"], view) == []


def test_research_fixture_can_be_used_as_market_research_context():
    research = load_result("research")
    request = ResearchRequest.model_validate(
        {
            "contract_version": research["contract_version"],
            "run_id": research["run_id"],
            "request_id": f"{research['run_id']}:market:1",
            "attempt": 1,
            "context": research["context"],
            "view": "market",
            "questions": ["KIVI와 InfiniGen의 시장 적용성을 비교한다."],
            "feedback": [],
            "previous_result": None,
            "research_context": research,
        }
    )
    assert request.research_context.view == "research"
    assert request.research_context.status == "ok"

def _base_chunk():
    return {
        "id": "paper:p1:t0",
        "source_id": "paper",
        "technology": "KIVI",
        "text": "A preserved source chunk with enough text to identify it.",
        "page": 1,
        "section": "ignored by the PDF contract",
        "char_start": 0,
        "char_end": 56,
        "token_start": 0,
        "captions": ["Figure 1"],
        "raw_sha256": "hash-a",
        "score": 0.25,
    }


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("text", "A different source chunk with enough text to identify it."),
        ("raw_sha256", "hash-b"),
        ("captions", ["Figure 2"]),
        ("char_start", 3),
        ("token_start", 1),
        ("source_id", "other-paper"),
    ],
)
def test_duplicate_chunk_id_rejects_changed_content_or_collection_metadata(field, changed_value):
    original = _base_chunk()
    lookup = {}
    _insert_chunk(lookup, original)

    changed = {**original, field: changed_value}
    with pytest.raises(ValueError, match="Conflicting duplicate source chunk"):
        _insert_chunk(lookup, changed)


def test_duplicate_chunk_id_allows_query_score_difference():
    original = _base_chunk()
    lookup = {}
    _insert_chunk(lookup, original)
    _insert_chunk(lookup, {**original, "score": 0.9})
    assert lookup[original["id"]]["score"] == original["score"]
