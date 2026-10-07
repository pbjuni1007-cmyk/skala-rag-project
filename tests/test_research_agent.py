"""Deterministic integration tests for the ResearchAgent boundary."""

import json

import pytest

from agents.researchers.agent import ResearchAgent, _safe_request_id
from agents.researchers.contract import ResearchResult
from agents.researchers.result import trace_errors
from tests.test_pipeline_improvements import (
    assessment,
    perspective_assessment,
    pipeline as base_pipeline,
)
from tests.test_research_maturity import PurposeGateway, successful_responses


CONTEXT = {
    "technologies": ["KIVI", "InfiniGen"],
    "domain": "문서 검토",
    "scenario": "업무 가정",
}
VIEWS = ("market", "stakeholder", "domain")


def _source(source_id):
    return {
        "type": "paper_pool",
        "authors": "Example authors",
        "title": f"Source {source_id}",
        "url": f"https://example.test/{source_id}",
        "version": "v1",
        "date": "2026-01-01",
        "accessed_at": "2026-10-07T00:00:00+00:00",
    }


def _request(view, request_id, research_context=None):
    return {
        "contract_version": "agent-contract-v1",
        "run_id": "run-1",
        "request_id": request_id,
        "attempt": 1,
        "context": CONTEXT,
        "view": view,
        "questions": ["문서 검토에 어떤 근거가 필요한가?"],
        "feedback": [],
        "previous_result": None,
        "research_context": research_context,
    }


def _pipeline(tmp_path, responses=None):
    pipeline = base_pipeline(tmp_path)
    pipeline.corpus.sources = {
        technology: _source(technology)
        for technology in CONTEXT["technologies"]
    }
    pipeline.corpus.source_metadata = lambda: {
        source_id: {
            key: value[key]
            for key in ("title", "url", "version", "date")
        }
        for source_id, value in pipeline.corpus.sources.items()
    }
    pipeline.corpus.web_search_calls = []

    def web_search(query, top_k, **kwargs):
        pipeline.corpus.web_search_calls.append((query, top_k, kwargs))
        return []

    pipeline.corpus.web_search = web_search
    if responses is None:
        responses = successful_responses()
    responses.update({view: perspective_assessment(view) for view in VIEWS})
    pipeline.gateway = PurposeGateway(responses)
    return pipeline


def test_all_views_run_through_real_runners_and_keep_request_local_logs(tmp_path):
    pipeline = _pipeline(tmp_path)
    agent = ResearchAgent(pipeline)

    research = agent.research(_request("research", "research-1"))
    assert ResearchResult.model_validate(research).status == "ok"
    assert set(research["assessments"]) == {"research_kivi", "research_infinigen"}
    assert trace_errors(research) == []

    for view in VIEWS:
        result = agent.research(_request(view, f"{view}-1", research))
        assert ResearchResult.model_validate(result).status == "ok"
        assert result["request_id"] == f"{view}-1"
        assert result["context"] == CONTEXT
        assert set(result["assessments"]) == {view}
        assert trace_errors(result) == []

        request_dir = tmp_path / "research" / view / f"{view}-1"
        assert (request_dir / "result.json").is_file()
        assert (request_dir / "retrieval" / f"{view}.json").is_file()
        assert (request_dir / "retrieval" / "balance.json").is_file()

    research_dir = tmp_path / "research" / "research" / "research-1"
    assert (research_dir / "result.json").is_file()
    assert (research_dir / "retrieval" / "research.json").is_file()
    assert pipeline.out == tmp_path


def test_invalid_request_stops_before_any_runner_gateway_or_search(tmp_path):
    pipeline = _pipeline(tmp_path)
    agent = ResearchAgent(pipeline)

    with pytest.raises(ValueError):
        agent.research(_request("market", "bad-1", None))

    assert pipeline.gateway.calls == []
    assert pipeline.corpus.searches == []
    assert pipeline.corpus.web_search_calls == []
    assert not (tmp_path / "research").exists()


def test_request_id_is_sanitized_and_completed_attempt_cannot_be_reused(tmp_path, monkeypatch):
    import agents.researchers.agent as agent_module

    pipeline = _pipeline(tmp_path, responses={})
    calls = []

    def fake_maturity(local_pipeline, request):
        calls.append("research")
        return {
            "research_kivi": assessment("KIVI", facets=()),
            "research_infinigen": assessment("InfiniGen", facets=()),
        }, {}

    monkeypatch.setattr(agent_module, "run_maturity", fake_maturity)
    request = _request("research", "one/../two")
    result = ResearchAgent(pipeline).research(request)
    request_dir = tmp_path / "research" / "research" / "one_.._two"

    assert result["request_id"] == "one/../two"
    assert _safe_request_id(request["request_id"]) == "one_.._two"
    assert (request_dir / "result.json").is_file()
    with pytest.raises(ValueError, match="already has a result"):
        ResearchAgent(pipeline).research(request)
    assert calls == ["research"]


def test_runner_failure_returns_failed_result_and_writes_error_json(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.corpus.search = lambda *args: (_ for _ in ()).throw(OSError("fake index failure"))
    result = ResearchAgent(pipeline).research(_request("research", "failure-1"))

    assert result["status"] == "failed"
    assert result["error"]["code"] == "retrieval_error"
    request_dir = tmp_path / "research" / "research" / "failure-1"
    assert json.loads((request_dir / "error.json").read_text()) == result["error"]
    assert json.loads((request_dir / "result.json").read_text()) == result
    assert (request_dir / "retrieval" / "research.json").is_file()


def test_invalid_core_citation_preserves_other_technology_in_insufficient_v1_result(tmp_path):
    responses = successful_responses()
    bad = assessment('KIVI')
    quote = 'A quotation absent from every supplied source.'
    bad['claims'][0]['references'][0]['quote'] = quote
    responses.update(research_kivi_0=bad, research_kivi_0_repair_0={
        'patches': [{'target_id': 'claims:0:reference:0', 'quote': quote}],
    })
    pipeline = _pipeline(tmp_path, responses)
    result = ResearchAgent(pipeline).research(_request('research', 'citation-failure-1'))

    assert ResearchResult.model_validate(result).status == 'insufficient'
    assert result['error'] is None
    assert result['assessments']['research_kivi']['claims'] == []
    assert any('구조·인용 검증' in gap for gap in result['assessments']['research_kivi']['gaps'])
    assert result['assessments']['research_infinigen'] == assessment('InfiniGen')
    assert [chunk['id'] for chunk in result['chunks']] == ['InfiniGen']
    assert set(result['sources']) == {'InfiniGen'}
    assert trace_errors(result) == []
