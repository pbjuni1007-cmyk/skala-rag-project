"""Deterministic tests for the initial technology-maturity runner."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents.researchers.maturity import FACETS, run_maturity
from agents.researchers.result import RetrievalFailure, node_error
from rag.llm import APIError
from tests.test_pipeline_improvements import assessment, pipeline, review


TECHNOLOGIES = ("KIVI", "InfiniGen")


def plan():
    return {
        "queries": [
            {"technology": technology, "facet": facet, "query": f"{technology} {facet} evidence"}
            for technology in TECHNOLOGIES
            for facet in FACETS
        ]
    }


def queries_for(technology, suffix=""):
    return {
        "queries": [
            {
                "technology": technology,
                "facet": facet,
                "query": f"{technology} {facet} evidence{suffix}",
            }
            for facet in FACETS
        ]
    }


def review_for(technology, deficient=False):
    result = review(deficient)
    for item in result["items"]:
        item["chunk_ids"] = [technology]
    return result


class PurposeGateway:
    """Return scripted values by purpose so technology concurrency is deterministic."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate(self, purpose, instructions, content, schema):
        payload = json.loads(content)
        self.calls.append((purpose, instructions, payload, schema))
        value = self.responses[purpose]
        if isinstance(value, list):
            value = value.pop(0)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            value = value(purpose, payload)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def request():
    return SimpleNamespace(
        request_id="run-1:research:1",
        context=SimpleNamespace(
            technologies=list(TECHNOLOGIES),
            domain="기업 문서 검토",
            scenario="장문 사업 문서의 근거 확인",
        ),
        questions=["정확도와 메모리 부담을 함께 확인해 주세요"],
        feedback=[],
    )


def make_pipeline(tmp_path, responses):
    p = pipeline(tmp_path)
    p.gateway = PurposeGateway(responses)
    p.settings.integer = lambda name, default: 2
    return p


def successful_responses():
    return {
        "research_queries": plan(),
        "retrieval_review_kivi_0": review_for("KIVI"),
        "retrieval_review_infinigen_0": review_for("InfiniGen"),
        "research_kivi_0": assessment("KIVI"),
        "research_infinigen_0": assessment("InfiniGen"),
    }


def error_code(value):
    return value.code if hasattr(value, "code") else value["code"]


def test_run_maturity_returns_both_core_assessments_and_cited_chunk_lookup(tmp_path):
    p = make_pipeline(tmp_path, successful_responses())

    assessments, chunk_lookup = run_maturity(p, request())

    assert set(assessments) == {"research_kivi", "research_infinigen"}
    assert all(value["status"] == "ok" and len(value["claims"]) == 4 for value in assessments.values())
    assert set(chunk_lookup) == {"KIVI", "InfiniGen"}
    assert {call[0] for call in p.gateway.calls} == {
        "research_queries",
        "retrieval_review_kivi_0",
        "retrieval_review_infinigen_0",
        "research_kivi_0",
        "research_infinigen_0",
    }

    plan_call = next(call for call in p.gateway.calls if call[0] == "research_queries")
    assert plan_call[2]["domain"] == request().context.domain
    assert plan_call[2]["scenario"] == request().context.scenario
    assert plan_call[2]["questions"] == request().questions
    stored = json.loads((tmp_path / "retrieval/research.json").read_text())
    assert stored["query_plan"] == plan()
    assert set(stored["technology_queries"]) == set(TECHNOLOGIES)
    balance = json.loads((tmp_path / "balance.json").read_text())
    assert balance["view"] == "research"
    assert balance["technologies"] == {
        "KIVI": {"distinct_sources": 1, "max_source_share": 1.0},
        "InfiniGen": {"distinct_sources": 1, "max_source_share": 1.0},
    }


def test_deficient_reviews_rewrite_once_then_return_insufficient_with_reasons(tmp_path):
    responses = successful_responses()
    responses.update({
        "retrieval_review_kivi_0": review_for("KIVI", deficient=True),
        "rewrite_kivi": queries_for("KIVI", " revised"),
        "retrieval_review_kivi_1": review_for("KIVI", deficient=True),
    })
    p = make_pipeline(tmp_path, responses)

    assessments, _ = run_maturity(p, request())

    assert assessments["research_kivi"]["status"] == "insufficient"
    assert assessments["research_kivi"]["claims"] == []
    assert assessments["research_kivi"]["conflicts"] == []
    assert any("conditions" in gap and "A100 experimental setup section" in gap for gap in assessments["research_kivi"]["gaps"])
    assert not any(call[0].startswith("research_kivi_") for call in p.gateway.calls)
    assert len([call for call in p.gateway.calls if call[0].startswith("retrieval_review_kivi_")]) == 2
    assert len([call for call in p.gateway.calls if call[0] == "rewrite_kivi"]) == 1
    searched = [query for query, technology, _ in p.corpus.searches if technology == "KIVI"]
    assert any(query.endswith("revised") for query in searched)


def test_zero_hits_return_insufficient_without_review_or_assessment_calls(tmp_path):
    p = make_pipeline(tmp_path, {"research_queries": plan()})
    p.corpus.search = lambda query, technology, top_k: []

    assessments, chunk_lookup = run_maturity(p, request())

    assert all(value["status"] == "insufficient" for value in assessments.values())
    assert all(value["claims"] == [] and len(value["gaps"]) == 6 for value in assessments.values())
    assert all(any(gap.startswith("검색 결과 없음:") for gap in value["gaps"]) for value in assessments.values())
    assert all(any("한계·위험 근거 미확인" in gap for gap in value["gaps"]) for value in assessments.values())
    assert chunk_lookup == {}
    assert [call[0] for call in p.gateway.calls] == ["research_queries"]
    balance = json.loads((tmp_path / "balance.json").read_text())
    assert balance["technologies"] == {
        "KIVI": {"distinct_sources": 0, "max_source_share": 0.0},
        "InfiniGen": {"distinct_sources": 0, "max_source_share": 0.0},
    }


def test_search_error_is_retrieval_failure_and_both_workers_are_drained(tmp_path):
    p = make_pipeline(tmp_path, {"research_queries": plan()})
    searched = []

    def fail_search(query, technology, top_k):
        searched.append(technology)
        raise OSError("local index unavailable")

    p.corpus.search = fail_search

    with pytest.raises(RetrievalFailure):
        run_maturity(p, request())

    assert set(searched) == set(TECHNOLOGIES)
    stored = json.loads((tmp_path / "retrieval/research.json").read_text())
    assert stored["error_type"] == "RetrievalFailure"
    assert all(
        any(entry.get("status") == "failed" for entry in stored["technologies"][technology])
        for technology in TECHNOLOGIES
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [("provider unavailable", "api_error"), ("reservation retained after timeout", "uncertain_request")],
)
def test_api_errors_propagate_and_map_to_contract_codes(tmp_path, message, expected):
    p = make_pipeline(tmp_path, {"research_queries": Exception("unused")})
    p.gateway.responses["research_queries"] = APIError(message)

    with pytest.raises(APIError) as exc:
        run_maturity(p, request())

    assert error_code(node_error(exc.value, "research")) == expected
    assert [call[0] for call in p.gateway.calls] == ["research_queries"]


def test_invalid_assessment_after_rewrite_is_failed_and_calls_stay_bounded(tmp_path):
    responses = successful_responses()
    responses.update({
        "research_kivi_0": ["{}"],
        "research_kivi_0_repair": ["{}"],
        "rewrite_kivi": queries_for("KIVI", " revised"),
        "retrieval_review_kivi_1": review_for("KIVI"),
        "research_kivi_1": ["{}"],
        "research_kivi_1_repair": ["{}"],
    })
    p = make_pipeline(tmp_path, responses)

    with pytest.raises(ValueError) as exc:
        run_maturity(p, request())

    assert error_code(node_error(exc.value, "research")) == "invalid_response"
    purposes = [call[0] for call in p.gateway.calls]
    assert purposes.count("retrieval_review_kivi_0") == 1
    assert purposes.count("retrieval_review_kivi_1") == 1
    assert purposes.count("rewrite_kivi") == 1
    assert purposes.count("research_kivi_0") == 1
    assert purposes.count("research_kivi_1") == 1
    assert purposes.count("research_kivi_0_repair") == 1
    assert purposes.count("research_kivi_1_repair") == 1
