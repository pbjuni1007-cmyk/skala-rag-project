"""Real research/Pipeline/Supervisor boundaries with only external I/O scripted."""

from copy import deepcopy
import json

import pytest

from agents.contracts import COMMON_FIELDS, ResearchResult, SupervisorDecision, VIEWS
from agents.researchers.agent import ResearchAgent
from agents.state import Limits
from agents.store import RunStore
from agents.supervisor import Nodes, Supervisor
from rag.budget import BudgetExceeded
from rag.llm import APIError
from rag.request_budget import InputBudgetExceeded
from tests.test_pipeline_improvements import QUOTE
from tests.test_research_agent import CONTEXT, _pipeline, _request, _source
from tests.test_research_maturity import queries_for


RUN_ID = "research-integration"
IDENTITY = "offline-research-pipeline-supervisor-v1"


def _decide(request):
    action = request["allowed_actions"][0]
    code = (
        "evidence_ready" if action == "writer" else
        "evidence_gap" if action in request["results"] else
        "initial_research" if action == "research" else "missing_view"
    )
    return SupervisorDecision(
        **{field: deepcopy(request[field]) for field in COMMON_FIELDS},
        next_action=action,
        evidence_sufficient=action == "writer",
        reason_code=code,
        reason="오프라인 조사 연결 검사에서 현재 결과와 허용 작업을 확인했습니다.",
        feedback=[],
    )


def _unexpected_output(_request):
    raise AssertionError("This research integration test stops before output nodes")


def _supervisor(tmp_path, *, scores=None, decide=_decide, limits=None):
    pipeline = _pipeline(tmp_path / "pipeline")
    # Exercise the production full-chunk context path, with corpus I/O replaced.
    for chunk in pipeline.corpus.chunks:
        chunk.update(section=chunk["text"][:160], content_sha256="a" * 64)
    pipeline.corpus.evidence_tokens = lambda text: len(text.split())
    pipeline.corpus.adjacent_candidates = lambda seeds: []
    if scores == "paper":
        search = pipeline.corpus.search
        pipeline.corpus.search = lambda query, technology, top_k: [
            {**chunk, "score": len(query) / 100}
            for chunk in search(query, technology, top_k)
        ]
    if scores == "web":
        pipeline.corpus.sources["web"] = {**_source("web"), "type": "external_web"}
        for view in VIEWS[1:]:
            pipeline.gateway.responses[view]["claims"][0]["references"] = [
                {"chunk_id": "web", "quote": QUOTE}
            ]

        def web_search(query, top_k, **kwargs):
            pipeline.corpus.web_search_calls.append((query, top_k, kwargs))
            return [{"id": "web", "source_id": "web", "technology": "both",
                     "text": QUOTE, "page": None, "section": "Frozen snapshot, paragraph 1",
                     "score": len(pipeline.corpus.web_search_calls)}]

        pipeline.corpus.web_search = web_search
    supervisor = Supervisor(
        Nodes(ResearchAgent(pipeline).research, _unexpected_output,
              _unexpected_output, _unexpected_output),
        decide, RunStore(tmp_path / "supervisor"), limits=limits,
    )
    return supervisor, pipeline


def _run(supervisor, **kwargs):
    return supervisor.run(RUN_ID, deepcopy(CONTEXT), IDENTITY, **kwargs)


@pytest.mark.parametrize("scores", [None, "paper", "web"])
def test_actual_research_results_reach_writer_gate_with_stable_evidence(tmp_path, scores):
    supervisor, pipeline = _supervisor(tmp_path, scores=scores)
    original_chunks = deepcopy(pipeline.corpus.chunks)
    original_sources = deepcopy(pipeline.corpus.sources)

    state = _run(supervisor, pause_after=9)

    assert state["status"] == "running", state["last_error"]
    assert state["next_action"] == "writer"
    assert state["evidence_sufficient"] is True
    assert state["attempts"] == dict.fromkeys(VIEWS, 1)
    assert state["pending"] is None
    for view, ref in state["results"].items():
        result = ResearchResult.model_validate(supervisor.store.get(ref))
        assert result.view == view and result.status == "ok"
        assert result.contract_version == "agent-contract-v1"
        assert all("score" not in chunk.model_dump() for chunk in result.chunks)
        papers = [chunk for chunk in result.chunks if chunk.page is not None]
        assert [chunk.model_dump() for chunk in papers] == [
            {**chunk, "section": None} for chunk in original_chunks
        ]
    if scores:
        assert any("score" in chunk for _, _, payload, _ in pipeline.gateway.calls
                   for chunk in payload.get("chunks", []))
    assert pipeline.corpus.chunks == original_chunks
    assert pipeline.corpus.sources == original_sources


@pytest.mark.parametrize("field,value", [
    ("text", QUOTE + " Changed source body."),
    ("page", 2),
    ("source_id", "other-source"),
    ("source_title", "Changed source metadata"),
    ("content_sha256", "b" * 64),
])
def test_actual_followup_rejects_same_id_with_changed_evidence(tmp_path, field, value):
    supervisor, pipeline = _supervisor(tmp_path)
    initial = _run(supervisor, pause_after=2)
    original = supervisor.store.get(initial["results"]["research"])
    if field == "source_title":
        pipeline.corpus.sources["KIVI"]["title"] = value
    else:
        pipeline.corpus.chunks[0][field] = value

    state = _run(supervisor, resume=True)

    assert state["status"] == "incomplete"
    assert state["last_error"]["code"] == "artifact_mismatch"
    assert state["report"] is None and state["publication"] is None
    assert supervisor.store.get(initial["results"]["research"]) == original
    if field != "source_title":
        failed = ResearchResult.model_validate(supervisor.store.get(state["results"]["market"]))
        assert failed.status == "failed" and failed.error.code == "artifact_mismatch"


def test_actual_followup_rejects_same_web_id_with_changed_section(tmp_path):
    supervisor, pipeline = _supervisor(tmp_path, scores="web")
    initial = _run(supervisor, pause_after=4)
    original = supervisor.store.get(initial["results"]["market"])
    search = pipeline.corpus.web_search
    pipeline.corpus.web_search = lambda query, top_k, **kwargs: [
        {**chunk, "section": "Different snapshot paragraph"}
        for chunk in search(query, top_k, **kwargs)
    ]

    state = _run(supervisor, resume=True)

    assert state["status"] == "incomplete"
    assert state["last_error"]["code"] == "artifact_mismatch"
    assert supervisor.store.get(initial["results"]["market"]) == original


def test_actual_empty_search_is_insufficient_and_stops_at_worker_limit(tmp_path):
    supervisor, pipeline = _supervisor(tmp_path, limits=Limits(worker_attempts=1))
    pipeline.corpus.search = lambda *args: []

    state = _run(supervisor)

    result = ResearchResult.model_validate(supervisor.store.get(state["results"]["research"]))
    assert result.status == "insufficient"
    assert all(assessment.gaps for assessment in result.assessments.values())
    assert result.chunks == [] and result.error is None
    assert state["status"] == "incomplete"
    assert state["last_error"]["code"] == "limit_exceeded"
    assert state["attempts"] == {"research": 1}
    assert [call[0] for call in pipeline.gateway.calls] == ["research_queries"]


def test_actual_partial_research_feedback_rechecks_only_missing_technology(tmp_path):
    supervisor, pipeline = _supervisor(tmp_path, scores="paper")
    search = pipeline.corpus.search
    pipeline.corpus.search = lambda query, technology, top_k: (
        [] if technology == "KIVI" and "feedback" not in query
        else search(query, technology, top_k)
    )
    pipeline.gateway.responses["rewrite_kivi_feedback"] = queries_for("KIVI", " feedback")
    initial = _run(supervisor, pause_after=2)
    previous = supervisor.store.get(initial["results"]["research"])
    assert ResearchResult.model_validate(previous).status == "insufficient"
    assert previous["assessments"]["research_kivi"]["claims"] == []
    assert previous["assessments"]["research_infinigen"]["status"] == "ok"

    state = _run(supervisor, resume=True, pause_after=2)

    current = ResearchResult.model_validate(supervisor.store.get(state["results"]["research"]))
    assert current.status == "ok" and current.attempt == 2
    assert current.request_id != previous["request_id"]
    assert current.assessments["research_infinigen"].model_dump() == previous["assessments"]["research_infinigen"]
    purposes = [call[0] for call in pipeline.gateway.calls]
    assert purposes.count("research_infinigen_0") == 1
    assert "rewrite_infinigen_feedback" not in purposes
    rewrite = next(payload for purpose, _, payload, _ in pipeline.gateway.calls
                   if purpose == "rewrite_kivi_feedback")
    assert rewrite["previous_assessment"] == previous["assessments"]["research_kivi"]
    assert rewrite["questions"] and rewrite["feedback"]
    assert set(previous["assessments"]["research_kivi"]["gaps"]) <= set(rewrite["feedback"])
    assert state["feedback"] == {}


@pytest.mark.parametrize("target", ["research", "market"])
def test_actual_requested_research_rework_preserves_target_and_invalidates_dependents(tmp_path, target):
    supervisor, pipeline = _supervisor(tmp_path, scores="paper")
    initial = _run(supervisor, pause_after=8)
    assert set(initial["results"]) == set(VIEWS)
    previous = supervisor.store.get(initial["results"][target])
    feedback = "적용 조건을 보완하세요. | claim_ids=['market-1'] | gap_ids=['gap-1']"

    def rework(request):
        assert target in request["allowed_actions"]
        assert request["results"][target] == previous
        return SupervisorDecision(
            **{field: deepcopy(request[field]) for field in COMMON_FIELDS},
            next_action=target, evidence_sufficient=False, reason_code="evidence_gap",
            reason="현재 관점의 근거 조건을 보완해야 합니다.", feedback=[feedback],
        )

    supervisor.decide = rework
    if target == "research":
        for technology in CONTEXT["technologies"]:
            pipeline.gateway.responses[f"rewrite_{technology.lower()}_feedback"] = queries_for(
                technology, " feedback"
            )
        rewrite_purpose = "rewrite_kivi_feedback"
    else:
        rewrite_purpose = "market_rewrite"
        pipeline.gateway.responses[rewrite_purpose] = {
            "queries": [{"facet": "costs", "query": "operating cost evidence"}]
        }
        revised = deepcopy(previous["assessments"]["market"])
        for claim in revised["claims"]:
            claim["text"] = "지정된 비용 조건을 보완한 분석" if claim["facet"] == "costs" else "변경하면 안 되는 기존 주장"
        pipeline.gateway.responses["market_reassessment"] = revised

    state = _run(supervisor, resume=True, pause_after=2)

    assert state["status"] == "running", state["last_error"]
    current = supervisor.store.get(state["results"][target])
    assert ResearchResult.model_validate(current).status == "ok"
    assert current["attempt"] == 2 and current["request_id"] != previous["request_id"]
    assert state["evidence_sufficient"] is False
    assert state["report"] is None and state["evaluation"] is None and state["publication"] is None
    rewrite = next(payload for purpose, _, payload, _ in pipeline.gateway.calls
                   if purpose == rewrite_purpose)
    assert feedback in rewrite["feedback"]
    if target == "research":
        assert set(state["results"]) == {"research"}
        assert set(state["summaries"]) == {"research"}
    else:
        assert all(state["results"][view] == initial["results"][view]
                   for view in VIEWS if view != target)
        original_claims = previous["assessments"][target]["claims"]
        current_claims = current["assessments"][target]["claims"]
        assert [c for c in current_claims if c["facet"] != "costs"] == [
            c for c in original_claims if c["facet"] != "costs"
        ]
        assert all(c["text"] == "지정된 비용 조건을 보완한 분석"
                   for c in current_claims if c["facet"] == "costs")
    assert supervisor.store.get(initial["results"][target]) == previous


@pytest.mark.parametrize("error,code", [
    (APIError("generation failed"), "api_error"),
    (APIError("request uncertain; reservation retained"), "uncertain_request"),
    (BudgetExceeded("local test budget exhausted"), "budget_exceeded"),
    (InputBudgetExceeded("research_queries", 100, 50), "input_budget_exceeded"),
    (OSError("offline index unavailable"), "retrieval_error"),
])
def test_actual_execution_failures_remain_failed_through_supervisor(tmp_path, error, code):
    supervisor, pipeline = _supervisor(tmp_path)
    if code == "retrieval_error":
        def unavailable(*args):
            raise error
        pipeline.corpus.search = unavailable
    else:
        pipeline.gateway.responses["research_queries"] = error

    state = _run(supervisor)

    result = ResearchResult.model_validate(supervisor.store.get(state["results"]["research"]))
    assert result.status == "failed" and result.error.code == code
    assert result.assessments == {} and result.chunks == [] and result.sources == {}
    assert state["status"] == ("needs_attention" if code == "uncertain_request" else "incomplete")
    assert state["last_error"]["code"] == code
    assert state["attempts"] == {"research": 1}
    assert state["report"] is None and state["publication"] is None
    assert bool(state["pending"]) == (code == "uncertain_request")
    assert [call[0] for call in pipeline.gateway.calls] == ["research_queries"]


@pytest.mark.parametrize("boundary", ["model", "snapshot"])
def test_interrupted_actual_research_is_not_reissued_on_resume(tmp_path, monkeypatch, boundary):
    supervisor, pipeline = _supervisor(tmp_path)
    checkpoint = supervisor.store.checkpoint
    if boundary == "model":
        def interrupted(*args):
            raise KeyboardInterrupt("simulated process interruption at model boundary")
        pipeline.gateway.responses["research_queries"] = interrupted
    else:
        def interrupt_completed_research(state):
            if "research" in state["results"] and state["pending"] is None:
                raise KeyboardInterrupt("simulated interruption before completed snapshot")
            checkpoint(state)
        monkeypatch.setattr(supervisor.store, "checkpoint", interrupt_completed_research)

    with pytest.raises(KeyboardInterrupt):
        _run(supervisor)

    snapshot = json.loads((supervisor.store.root / "snapshot.json").read_text())
    assert snapshot["pending"]["role"] == "research"
    request_id = snapshot["pending"]["request_id"]
    request_dir = pipeline.out / "research" / "research" / request_id
    assert (request_dir / ".in_progress").exists() == (boundary == "model")
    assert (request_dir / "result.json").exists() == (boundary == "snapshot")
    calls = deepcopy(pipeline.gateway.calls)
    monkeypatch.setattr(supervisor.store, "checkpoint", checkpoint)

    state = _run(supervisor, resume=True)

    assert state["status"] == "needs_attention"
    assert state["last_error"]["code"] == "uncertain_request"
    assert state["pending"] == snapshot["pending"]
    assert pipeline.gateway.calls == calls
    request = {**_request("research", request_id), "run_id": RUN_ID}
    with pytest.raises(ValueError, match="already in progress|already has a result"):
        ResearchAgent(pipeline).research(request)
    assert pipeline.gateway.calls == calls
