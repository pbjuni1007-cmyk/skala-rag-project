"""Real team components with only model HTTP and corpus I/O replaced by fixtures."""
from copy import deepcopy
from pathlib import Path
import json

import pytest
import yaml

from agents.contracts import COMMON_FIELDS
from agents.state import Limits
from rag import agent_runtime
from rag.budget import Budget
from rag.llm import Gateway
from test_gateway import completed, settings
from test_pipeline_improvements import report_fixture
from test_research_agent import _pipeline
from test_report_evaluator import judge_result


@pytest.fixture
def case(tmp_path, settings, monkeypatch):
    pipeline = _pipeline(tmp_path / "fixtures")
    corpus = pipeline.corpus
    for chunk in corpus.chunks:
        chunk["section"] = "PDF page preview"
    corpus.retrieval_identity = lambda: {"fixture": "offline-v1"}
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(pipeline.config, allow_unicode=True), encoding="utf-8")
    responses = deepcopy(pipeline.gateway.responses)
    report = report_fixture(tmp_path / "report-fixture")[3]
    responses["synthesis_report"] = {key: value for key, value in report.items() if key != "gap_decisions"}
    responses["synthesis_gaps_0"] = {"gap_decisions": report["gap_decisions"]}
    responses["report_quality_judge"] = judge_result()
    settings.values.update(RAG_MAX_API_CALLS_PER_RUN="40", LANGSMITH_TRACING="false")
    calls = []

    def request(gateway, path, payload=None):
        if path == "responses/input_tokens":
            return {"input_tokens": 100}
        assert path == "responses", path
        purpose = payload["text"]["format"]["name"]
        content = json.loads(payload["input"])
        calls.append((purpose, content))
        if purpose.startswith("supervisor_"):
            action = content["allowed_actions"][0]
            value = {**{key: content[key] for key in COMMON_FIELDS}, "next_action": action,
                     "evidence_sufficient": action == "writer", "feedback": [],
                     "reason": "가상 근거를 사용한 오프라인 경로 검증",
                     "reason_code": "evidence_ready" if action == "writer" else
                     "evidence_gap" if action in content["summaries"] else
                     "initial_research" if action == "research" else "missing_view"}
        else:
            value = responses[purpose]
            if isinstance(value, list):
                value = value.pop(0)
            if isinstance(value, BaseException):
                raise value
            if callable(value):
                value = value(content)
        return completed(json.dumps(value, ensure_ascii=False))

    monkeypatch.setattr(Gateway, "request", request)
    return {"settings": settings, "corpus": corpus, "config_path": config,
            "index": {"index_hash": "offline-index", "embedding": {"revision": "mock-v1"}},
            "retrieval_check": {"status": "passed", "fingerprint": "offline-retrieval"},
            "output_root": tmp_path / "outputs", "run_id": "offline-real-components",
            "responses": responses, "calls": calls}


def execute(case, **overrides):
    arguments = {key: value for key, value in case.items() if key not in {"responses", "calls"}}
    return agent_runtime.execute_agent(**(arguments | overrides))


def test_real_team_chain_shares_run_budget_and_publishes(case):
    result = execute(case)
    state = result["state"]
    assert state["status"] == "completed", state["last_error"]
    assert result["human_review_pending"] is True
    root = Path(result["run"])
    assert (root / "manifest.json").is_file() and (root / "graph.mmd").is_file()
    assert state["attempts"] == {"research": 1, "market": 1, "stakeholder": 1,
                                  "domain": 1, "writer": 1, "evaluator": 1, "publish": 1}
    purposes = [purpose for purpose, _ in case["calls"]]
    assert any(purpose.startswith("supervisor_") for purpose in purposes)
    assert {"research_queries", "market", "stakeholder", "domain", "synthesis_report",
            "report_quality_judge"} <= set(purposes)
    ledger = json.loads(Path(case["settings"].get("BUDGET_LEDGER_PATH")).read_text())
    assert len(ledger["entries"]) == len(case["calls"])
    assert {entry["run_id"] for entry in ledger["entries"]} == {state["run_id"]}
    assert all(entry["state"] == "settled" for entry in ledger["entries"])
    assert list(root.rglob("*.pdf"))


@pytest.mark.parametrize("target", ["writer", "market"])
def test_real_quality_rework_uses_current_evaluation_and_feedback(case, target):
    claim_id = "synthesis-1" if target == "writer" else "market-1"
    criterion = "neutrality" if target == "writer" else "groundedness"
    def revise(content):
        claim = next(claim for claim in content["claims"] if claim["id"] == claim_id)
        return judge_result({criterion: {"status": "revise", "rationale": "조건 표현 수정 필요",
            "findings": [{"criterion": criterion, "target": claim_id, "claim_ids": [claim_id],
                          "evidence_ids": [item["evidence_id"] for item in claim["evidence"]], "issue": "조건 표현 확인",
                          "revision_request": "적용 조건을 명확히 쓰세요."}]}})
    case["responses"]["report_quality_judge"] = [revise, judge_result()]
    if target == "market":
        case["responses"]["market_rewrite"] = {"queries": [{"facet": "costs", "query": "operating cost evidence"}]}
        case["responses"]["market_reassessment"] = deepcopy(case["responses"]["market"])
    result = execute(case)
    assert result["state"]["status"] == "completed", result["state"]["last_error"]
    assert result["state"]["attempts"]["writer"] == result["state"]["attempts"]["evaluator"] == 2
    drafts = [payload for purpose, payload in case["calls"] if purpose == "synthesis_report"]
    if target == "writer":
        assert "revision_requests" in drafts[1]
    else:
        assert result["state"]["attempts"]["market"] == 2
        assert result["state"]["attempts"]["research"] == result["state"]["attempts"]["domain"] == 1
        feedback = next(payload["feedback"] for purpose, payload in case["calls"] if purpose == "market_rewrite")
        assert any(claim_id in note for note in feedback)


@pytest.mark.parametrize("failure", ["budget", "timeout", "step_limit"])
def test_real_chain_stops_without_publication(case, failure):
    if failure == "budget":
        case["settings"].values["RAG_MAX_API_CALLS_PER_RUN"] = "1"
    elif failure == "timeout":
        case["responses"]["research_queries"] = TimeoutError("offline uncertainty")
    limits = Limits(supervisor_steps=1) if failure == "step_limit" else None
    result = execute(case, limits=limits)
    assert result["state"]["status"] == ("needs_attention" if failure == "timeout" else "incomplete")
    assert result["state"]["publication"] is None
    assert not list(Path(result["run"]).rglob("*.pdf"))
    before = len(case["calls"])
    resumed = execute(case, resume=Path(result["run"]), limits=limits)
    for field in ("status", "run_id", "attempts", "pending", "results", "publication"):
        assert resumed["state"][field] == result["state"][field]
    assert resumed["state"]["last_error"]["code"] == result["state"]["last_error"]["code"]
    assert len(case["calls"]) == before


def test_paused_run_continues_same_id_without_replaying_research(case):
    paused = execute(case, pause_after=2)
    assert paused["state"]["status"] == "running" and paused["state"]["results"].keys() == {"research"}
    case["corpus"].sources["KIVI"]["accessed_at"] = "2026-10-08T00:00:00Z"
    result = execute(case, resume=Path(paused["run"]))
    assert result["state"]["status"] == "completed", result["state"]["last_error"]
    assert result["state"]["run_id"] == paused["state"]["run_id"]
    assert sum(purpose == "research_queries" for purpose, _ in case["calls"]) == 1
    root = Path(result["run"])
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    count = len(case["calls"])
    assert execute(case, resume=root)["state"] == result["state"]
    assert len(case["calls"]) == count
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("changed", ["nested_code", "raw_source", "call_cap", "ledger_reset", "ledger_missing", "ledger_truncated", "tracing", "font", "font_contents"])
def test_resume_rejects_changed_identity_before_any_model_call(case, changed, monkeypatch):
    if changed == "font_contents":
        font = case["output_root"].parent / "custom.ttf"
        font.write_bytes(b"offline test font placeholder")
        case["settings"].values["REPORT_FONT_PATH"] = str(font)
    paused = execute(case, pause_after=2)
    if changed == "nested_code":
        original = agent_runtime.code_identity
        monkeypatch.setattr(agent_runtime, "code_identity", lambda: original() | {"agents/researchers/prompts.py": "changed"})
    elif changed == "raw_source":
        case["corpus"].sources["KIVI"]["raw_sha256"] = "changed"
    elif changed == "call_cap":
        case["settings"].values["RAG_MAX_API_CALLS_PER_RUN"] = "41"
    elif changed == "tracing":
        case["settings"].values.update(LANGSMITH_TRACING="true", LANGSMITH_ENDPOINT="https://example.invalid", LANGSMITH_PROJECT="other")
    elif changed == "font":
        case["settings"].values["REPORT_FONT_PATH"] = "assets/fonts/NanumGothic-Bold.ttf"
    elif changed == "font_contents":
        font.write_bytes(b"changed font bytes")
    else:
        path = Path(case["settings"].get("BUDGET_LEDGER_PATH"))
        if changed == "ledger_truncated":
            ledger = json.loads(path.read_text())
            ledger["entries"] = []
            path.write_text(json.dumps(ledger), encoding="utf-8")
        elif changed == "ledger_missing":
            path.unlink()
        else:
            path.write_text('{"schema":1,"entries":[]}', encoding="utf-8")
    count = len(case["calls"])
    with pytest.raises(ValueError, match="[Rr]esume|[Ll]edger"):
        execute(case, resume=Path(paused["run"]))
    assert len(case["calls"]) == count


def test_budget_binding_preserves_old_entries_and_rejects_reset(settings):
    budget = Budget(settings)
    reservation = budget.reserve("older-run", "research", 24000, 4096)
    identity = budget.bind_identity()
    assert Budget(settings).bind_identity(identity) == identity
    ledger = json.loads(budget.path.read_text())
    assert [entry["id"] for entry in ledger["entries"]] == [reservation]
    budget.path.unlink()
    with pytest.raises(ValueError, match="[Ll]edger"):
        budget.reserve("new-run", "research", 24000, 4096)


@pytest.mark.parametrize("role", ["writer", "evaluator", "supervisor"])
def test_uncertain_generation_is_not_reissued_on_resume(case, role, monkeypatch):
    purpose = {"writer": "synthesis_report", "evaluator": "report_quality_judge"}.get(role)
    if purpose:
        case["responses"][purpose] = TimeoutError("offline uncertainty")
    else:
        original = Gateway.request
        def timeout(gateway, path, payload=None):
            if path == "responses" and payload["text"]["format"]["name"].startswith("supervisor_"):
                raise TimeoutError("offline uncertainty")
            return original(gateway, path, payload)
        monkeypatch.setattr(Gateway, "request", timeout)
    result = execute(case)
    assert result["state"]["status"] == "needs_attention"
    assert result["state"]["last_error"]["code"] == "uncertain_request"
    ledger = Path(case["settings"].get("BUDGET_LEDGER_PATH"))
    count = len(json.loads(ledger.read_text())["entries"])
    resumed = execute(case, resume=Path(result["run"]))
    assert resumed["state"]["status"] == "needs_attention"
    assert len(json.loads(ledger.read_text())["entries"]) == count


@pytest.mark.parametrize("arguments", [
    ["--agent", "--resume", "old"], ["--agent", "--reuse-calls", "old"],
    ["--agent", "--prepare"], ["--agent-resume", "old"],
    ["--agent", "--agent-resume", "old", "--refresh-web"],
])
def test_cli_rejects_mixed_execution_modes_before_settings(arguments, monkeypatch):
    import app
    monkeypatch.setattr("sys.argv", ["app.py", *arguments])
    monkeypatch.setattr(app.Settings, "load", lambda: pytest.fail("Invalid mode must fail before credential loading"))
    with pytest.raises(SystemExit) as error:
        app.main()
    assert error.value.code == 2


def test_cli_agent_mode_dispatches_real_team_chain(case, monkeypatch, capsys):
    import app
    case["settings"].values["RAG_OUTPUT_DIR"] = str(case["output_root"])
    monkeypatch.setattr("sys.argv", ["app.py", "--agent", "--config", str(case["config_path"])])
    monkeypatch.setattr(app.Settings, "load", lambda: case["settings"])
    case["corpus"].prepare_sources = lambda refresh: None
    case["corpus"].build = lambda: case["index"]
    monkeypatch.setattr("scripts.revalidate_retrieval.ensure_retrieval_regression", lambda *args, **kwargs: case["retrieval_check"])
    monkeypatch.setattr("rag.corpus.Corpus", lambda settings, config: case["corpus"])
    assert app.main() == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["status"] == "completed" and result["human_review_pending"] is True
    assert json.loads((Path(result["run"]) / "manifest.json").read_text())["mode"] == "agent"
