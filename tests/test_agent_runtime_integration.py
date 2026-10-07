"""Issue #4 end-to-end checks with published fictional responses, never model evidence."""
import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from agents.state import Limits
from agents.supervisor import Nodes
from rag.agent_runtime import load_run_context, run_team_agent
from rag.render import publish
from rag.settings import Settings
from test_team_contracts import response
from test_team_supervisor import AdaptiveDecider, FakeNodes, quality_rework


CONFIG = Path(__file__).resolve().parents[1] / "config/run.yaml"
RUN_ID = "offline-issue-4"


def execute(tmp_path, *, overrides=None, limits=None, decider=None):
    root = tmp_path / RUN_ID
    fake = FakeNodes(root, overrides=overrides)
    callbacks = fake.nodes()
    settings = Settings({"LANGSMITH_TRACING": "false"})
    events = []
    nodes = Nodes(callbacks.research, callbacks.write_report, callbacks.evaluate_report,
                  lambda request: publish(request, root, settings))
    state = run_team_agent(
        config_path=CONFIG, run_id=RUN_ID, identity="mock-node-code-data-config-v1",
        nodes=nodes, decide=decider or AdaptiveDecider(), settings=settings, output_root=tmp_path,
        limits=limits, on_event=events.append,
    )
    return state, fake, events, root


def assert_real_publication(state, root):
    assert state["status"] == "completed"
    result = json.loads((root / state["publication"]["path"]).read_text(encoding="utf-8"))
    assert result["status"] == "ok" and result["human_review_pending"] is True
    markdown = root / result["markdown_ref"]["path"]
    pdf = root / result["pdf_ref"]["path"]
    report = json.loads((root / state["report"]["path"]).read_text(encoding="utf-8"))
    assert markdown.read_text(encoding="utf-8") == report["markdown"]
    assert pdf.is_file() and 1 <= len(PdfReader(pdf).pages) == result["pdf_pages"] <= 10
    assert "SUMMARY" in PdfReader(pdf).pages[0].extract_text()
    assert "REFERENCE" in PdfReader(pdf).pages[-1].extract_text()
    return result


def test_config_context_is_validated_before_agent_execution(tmp_path):
    assert load_run_context(CONFIG)["technologies"] == ["KIVI", "InfiniGen"]
    bad = tmp_path / "bad.yaml"
    bad.write_text("technologies: [KIVI]\ndomain: test\nscenario: test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="technology pair"):
        load_run_context(bad)


def test_complete_graph_publishes_real_pdf_and_links_local_decisions(tmp_path):
    state, fake, events, root = execute(tmp_path)
    assert_real_publication(state, root)
    assert [role for role, _ in fake.calls] == ["research", "domain", "market", "stakeholder", "writer", "evaluator"]
    decisions = [event for event in events if event["node"] == "supervisor" and event.get("reason_code")]
    assert decisions and all(event["run_id"] == RUN_ID and event["request_id"] for event in decisions)
    assert any(event["next_action"] == "writer" and event["reason_code"] == "evidence_ready" for event in decisions)
    assert any(event["next_action"] == "publish" and event["reason_code"] == "quality_passed" for event in decisions)
    saved = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event.get("reason") for event in saved if event["node"] == "supervisor")
    assert all("reason" not in event for event in decisions)


def test_insufficient_market_reinvestigates_then_publishes(tmp_path):
    def insufficient_once(request, payload):
        return response("market_insufficient", request) if request["attempt"] == 1 else payload

    state, fake, events, root = execute(tmp_path, overrides={"market": insufficient_once})
    assert_real_publication(state, root)
    assert [request["attempt"] for request in fake.requests("market")] == [1, 2]
    assert any(event.get("reason_code") == "evidence_gap" and event.get("next_action") == "market"
               for event in events)


def test_failed_evaluation_rewrites_and_reevaluates_before_pdf(tmp_path):
    policy = AdaptiveDecider()

    def decide(request):
        result = policy(request)
        if result.next_action == "writer" and request["feedback"].get("writer"):
            return result.model_copy(update={"reason_code": "quality_rework"})
        return result

    state, fake, events, root = execute(tmp_path, overrides={"evaluator": quality_rework},
                                        decider=decide)
    assert_real_publication(state, root)
    assert len(fake.requests("writer")) == len(fake.requests("evaluator")) == 2
    assert any(event.get("reason_code") == "quality_rework" and event.get("next_action") == "writer"
               for event in events)


@pytest.mark.parametrize("failure", ["failed", "limit", "uncertain"])
def test_failure_and_retry_limit_stop_without_pdf(tmp_path, failure):
    if failure == "failed":
        override = lambda request, payload: response("market_failed", request)
    elif failure == "uncertain":
        def override(request, payload):
            failed = response("market_failed", request)
            failed["error"] = {"code": "uncertain_request", "message": "Completion unknown", "retryable": False}
            return failed
    else:
        override = lambda request, payload: response("market_insufficient", request)
    state, fake, _, root = execute(tmp_path, overrides={"market": override},
                                   limits=Limits(worker_attempts=2))
    assert state["status"] == ("needs_attention" if failure == "uncertain" else "incomplete")
    assert state["publication"] is None
    assert not list(root.rglob("*.pdf"))
    assert not fake.requests("writer")


def test_unsafe_run_id_is_rejected_before_creating_output(tmp_path):
    fake = FakeNodes(tmp_path)
    with pytest.raises(ValueError, match="safe output directory"):
        run_team_agent(config_path=CONFIG, run_id="../outside", identity="id", nodes=fake.nodes(),
                       decide=AdaptiveDecider(), settings=Settings({}), output_root=tmp_path)
    assert not (tmp_path / "outside").exists()
