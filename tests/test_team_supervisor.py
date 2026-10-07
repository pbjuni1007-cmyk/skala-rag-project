"""Exercise the real graph with JSON callbacks; no model or renderer is used."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from langgraph.graph.state import CompiledStateGraph

from agents import contracts
from agents.decision import GatewayDecider
from agents.state import Limits, VIEWS
from agents.store import RunStore
from agents.supervisor import Nodes, Supervisor
from test_team_contracts import COMMON, example, response


CONTEXT = example("research_request")["context"]
RUN_ID = "offline-team-run"
IDENTITY = "offline-code-data-config-v1"


def decision(request, action, *, sufficient=None, feedback=None):
    if action == "writer":
        code = "evidence_ready"
    elif action == "research" and action not in request["results"]:
        code = "initial_research"
    elif action in request["results"]:
        code = "evidence_gap"
    else:
        code = "missing_view"
    return contracts.SupervisorDecision.model_validate({
        **{field: deepcopy(request[field]) for field in COMMON},
        "next_action": action,
        "evidence_sufficient": action == "writer" if sufficient is None else sufficient,
        "reason_code": code,
        "reason": "가상 근거를 확인한 오프라인 연결 검증 결정",
        "feedback": feedback or [],
    })


class AdaptiveDecider:
    def __init__(self, order=("domain", "market", "stakeholder"), approve=True):
        self.order = ("research", *order)
        self.approve = approve
        self.requests = []

    def __call__(self, request):
        self.requests.append(deepcopy(request))
        results = request["results"]
        for view in self.order:
            if view in results and results[view]["status"] == "insufficient":
                gaps = [gap for assessment in results[view]["assessments"].values()
                        for gap in assessment["gaps"]]
                return decision(request, view, feedback=gaps)
        for view in self.order:
            if view not in results:
                return decision(request, view)
        if "writer" not in request["allowed_actions"]:
            target = next(view for view in self.order if view in request["allowed_actions"])
            return decision(request, target)
        return decision(request, "writer", sufficient=self.approve)


class FakeNodes:
    """Published mock payloads plus real temporary output-file hashes."""
    def __init__(self, root, *, overrides=None, models=False):
        self.root = Path(root)
        self.overrides = overrides or {}
        self.models = models
        self.calls = []

    def requests(self, role):
        return [request for actual, request in self.calls if actual == role]

    def call(self, role, request):
        self.calls.append((role, deepcopy(request)))
        if role in VIEWS:
            payload = response(f"{role}_ok", request)
            model = contracts.ResearchResult
        elif role == "writer":
            payload = response("report_ok", request)
            model = contracts.ReportResult
        elif role == "evaluator":
            payload = response("evaluation_pass", request)
            payload["report_request_id"] = request["report_result"]["request_id"]
            model = contracts.EvaluationResult
        else:
            payload = response("publish_ok", request)
            # These bytes test persistence only, not PDF validity or visual quality.
            files = {
                "markdown_ref": ("report/mock.md", request["report_result"]["markdown"].encode()),
                "pdf_ref": ("report/mock.pdf", b"Fictional publisher fixture; not a rendered PDF.\n"),
            }
            for field, (name, content) in files.items():
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                payload[field] = {"path": name, "sha256": hashlib.sha256(content).hexdigest()}
            model = contracts.PublishResult
        if role in self.overrides:
            payload = self.overrides[role](request, payload)
        return model.model_validate(payload) if self.models else payload

    def nodes(self):
        return Nodes(
            research=lambda request: self.call(request["view"], request),
            write_report=lambda request: self.call("writer", request),
            evaluate_report=lambda request: self.call("evaluator", request),
            publish=lambda request: self.call("publish", request),
        )


def make_supervisor(tmp_path, *, decider=None, overrides=None, limits=None, models=False, on_event=None):
    fake = FakeNodes(tmp_path, overrides=overrides, models=models)
    policy = decider if decider is not None else AdaptiveDecider()
    supervisor = Supervisor(fake.nodes(), policy, RunStore(tmp_path), limits=limits, on_event=on_event)
    return supervisor, fake, policy


def run(supervisor, **kwargs):
    return supervisor.run(RUN_ID, deepcopy(CONTEXT), IDENTITY, **kwargs)


def quality_rework(request, payload, target="writer"):
    if request["attempt"] > 1:
        return payload
    result = response("evaluation_rework", request)
    result["report_request_id"] = request["report_result"]["request_id"]
    result["repair_requests"][0]["target"] = target
    return result


def assert_not_published(state, fake):
    assert state["status"] in {"incomplete", "needs_attention"}
    assert not fake.requests("publish")


@pytest.mark.parametrize("order", [("domain", "market", "stakeholder"),
                                   ("stakeholder", "domain", "market")])
@pytest.mark.parametrize("models", [False, True])
def test_actual_conditional_graph_follows_different_valid_decisions(tmp_path, order, models):
    supervisor, fake, policy = make_supervisor(tmp_path, decider=AdaptiveDecider(order), models=models)
    compiled = supervisor.compile()
    assert isinstance(compiled, CompiledStateGraph)
    assert any(edge.conditional for edge in compiled.get_graph().edges)

    state = run(supervisor)

    assert state["status"] == "completed"
    assert [role for role, _ in fake.calls] == ["research", *order, "writer", "evaluator", "publish"]
    assert state["pending"] is None
    assert state["step_count"] >= len(fake.calls)
    assert len(policy.requests) >= 5
    latest = policy.requests[-1]
    assert set(latest["results"]) == set(VIEWS)
    assert latest["results"]["research"]["chunks"][0]["text"] == example("research_ok")["chunks"][0]["text"]
    assert all(request["context"] == CONTEXT for _, request in fake.calls)
    ids = [request["request_id"] for _, request in fake.calls]
    assert len(ids) == len(set(ids))
    publication = fake.requests("publish")[0]
    assert publication["evaluation_result"]["report_request_id"] == publication["report_result"]["request_id"]


def test_every_subordinate_returns_through_supervisor_including_publication(tmp_path):
    supervisor, _, _ = make_supervisor(tmp_path)
    edges = supervisor.compile().get_graph().edges
    for role in (*VIEWS, "writer", "evaluator", "publish"):
        assert {edge.target for edge in edges if edge.source == role} == {"supervisor"}


def test_insufficient_evidence_targets_the_same_view_with_previous_result_and_reason(tmp_path):
    def insufficient_once(request, payload):
        return response("market_insufficient", request) if request["attempt"] == 1 else payload

    supervisor, fake, _ = make_supervisor(tmp_path, overrides={"market": insufficient_once})
    state = run(supervisor)

    assert state["status"] == "completed"
    first, retry = fake.requests("market")
    assert [first["attempt"], retry["attempt"]] == [1, 2]
    assert retry["previous_result"]["request_id"] == first["request_id"]
    assert retry["previous_result"]["status"] == "insufficient"
    assert retry["research_context"]["view"] == "research"
    gap = example("market_insufficient")["assessments"]["market"]["gaps"][0]
    assert gap in " ".join(retry["feedback"])
    roles = [role for role, _ in fake.calls]
    assert roles.index("writer") > max(index for index, role in enumerate(roles) if role == "market")
    assert state["attempts"]["market"] == 2


def test_all_ok_evidence_still_requires_explicit_supervisor_approval(tmp_path):
    supervisor, fake, _ = make_supervisor(tmp_path, decider=AdaptiveDecider(approve=False))
    state = run(supervisor)
    assert_not_published(state, fake)
    assert not fake.requests("writer")


@pytest.mark.parametrize("action", ["writer", "evaluator", "publish"])
def test_decider_cannot_skip_research_or_quality_gates(tmp_path, action):
    supervisor, fake, _ = make_supervisor(tmp_path, decider=lambda request: decision(request, action, sufficient=True))
    state = run(supervisor)
    assert_not_published(state, fake)
    assert not fake.calls


def test_quality_rewrite_preserves_feedback_and_evaluates_new_report_before_publish(tmp_path):
    supervisor, fake, _ = make_supervisor(tmp_path, overrides={"evaluator": quality_rework})
    state = run(supervisor)

    assert state["status"] == "completed"
    first, second = fake.requests("writer")
    assert [first["attempt"], second["attempt"]] == [1, 2]
    assert first["request_id"] != second["request_id"]
    assert second["previous_report"]["request_id"] == first["request_id"]
    repair = example("evaluation_rework")["repair_requests"][0]
    assert repair["reason"] in " ".join(second["feedback"])
    assert repair["claim_ids"][0] in " ".join(second["feedback"])
    evaluations = fake.requests("evaluator")
    assert [request["report_result"]["request_id"] for request in evaluations] == [first["request_id"], second["request_id"]]
    assert fake.requests("publish")[0]["evaluation_result"]["report_request_id"] == second["request_id"]


def test_quality_research_repair_targets_named_view_and_invalidates_report(tmp_path):
    supervisor, fake, _ = make_supervisor(
        tmp_path, overrides={"evaluator": lambda request, payload: quality_rework(request, payload, "market")}
    )
    state = run(supervisor)

    assert state["status"] == "completed"
    assert len(fake.requests("market")) == 2
    assert all(len(fake.requests(view)) == 1 for view in ("research", "domain", "stakeholder"))
    retry = fake.requests("market")[1]
    assert "synthesis-1" in " ".join(retry["feedback"])
    assert retry["previous_result"]["attempt"] == 1
    assert fake.requests("writer")[1]["previous_report"] is None
    assert len(fake.requests("evaluator")) == 2


def test_refreshing_technical_research_invalidates_all_downstream_views_without_resetting_attempts(tmp_path):
    supervisor, fake, _ = make_supervisor(
        tmp_path, overrides={"evaluator": lambda request, payload: quality_rework(request, payload, "research")}
    )
    state = run(supervisor)

    assert state["status"] == "completed"
    for view in VIEWS:
        first, second = fake.requests(view)
        assert [first["attempt"], second["attempt"]] == [1, 2]
        assert state["attempts"][view] == 2
        if view != "research":
            assert second["previous_result"] is None
            assert second["research_context"]["request_id"] == fake.requests("research")[1]["request_id"]
    assert fake.requests("research")[1]["previous_result"]["attempt"] == 1
    assert fake.requests("research")[1]["research_context"] is None
    assert fake.requests("writer")[1]["previous_report"] is None
    assert len(fake.requests("evaluator")) == 2


def test_multiple_repair_targets_keep_full_reasons_and_ids_across_intermediate_invalidations(tmp_path):
    repairs = [
        {"target": target, "reason": ("보완해야 하는 구체적인 조건 " * 60) + f"끝-{target}",
         "claim_ids": [claim_id], "gap_ids": []}
        for target, claim_id in [("domain", "domain-1"), ("market", "market-1"), ("writer", "synthesis-1")]
    ]

    def multiple_repairs(request, payload):
        result = quality_rework(request, payload)
        if request["attempt"] == 1:
            result["repair_requests"] = repairs
        return result

    supervisor, fake, _ = make_supervisor(tmp_path, overrides={"evaluator": multiple_repairs})
    state = run(supervisor)
    assert state["status"] == "completed"
    for repair in repairs:
        retry = fake.requests(repair["target"])[1]
        feedback = " ".join(retry["feedback"])
        assert repair["reason"] in feedback
        assert repair["claim_ids"][0] in feedback
    assert len(fake.requests("research")) == 1
    assert len(fake.requests("evaluator")) == 2


@pytest.mark.parametrize("damage", ["source", "original_claim", "synthesis_quote", "unknown_summary_id"])
def test_writer_cannot_replace_evidence_or_bypass_structural_checks_with_a_positive_evaluation(tmp_path, damage):
    def dishonest_writer(request, payload):
        if damage == "source":
            next(iter(payload["sources"].values()))["title"] = "A source the researchers did not supply"
        elif damage == "original_claim":
            payload["joined"]["claims"]["market-1"]["text"] = "An unsupported new factual conclusion"
        elif damage == "synthesis_quote":
            payload["joined"]["evidence"]["E-synthesis-1-1"]["quote"] = "A quote absent from every source"
        else:
            payload["report"]["summary_claim_ids"].append("missing-claim")
        return payload

    supervisor, fake, _ = make_supervisor(tmp_path, overrides={"writer": dishonest_writer})
    state = run(supervisor)
    assert_not_published(state, fake)
    assert len(fake.requests("writer")) == 1


@pytest.mark.parametrize("damage", ["stale_report", "unknown_claim", "unknown_gap"])
def test_evaluation_cannot_refer_to_a_different_report_or_unknown_ids(tmp_path, damage):
    def invalid_evaluation(request, payload):
        if damage == "stale_report":
            payload["report_request_id"] = "another-report-request"
        else:
            payload = quality_rework(request, payload)
            key = "claim_ids" if damage == "unknown_claim" else "gap_ids"
            payload["repair_requests"][0][key] = ["unknown-id"]
        return payload

    supervisor, fake, _ = make_supervisor(tmp_path, overrides={"evaluator": invalid_evaluation})
    state = run(supervisor)
    assert_not_published(state, fake)
    assert len(fake.requests("writer")) == 1


@pytest.mark.parametrize("field", ["run_id", "request_id", "attempt", "context"])
def test_response_must_echo_the_actual_request_identity(tmp_path, field):
    def wrong_identity(request, payload):
        if field == "attempt":
            payload[field] += 1
        elif field == "context":
            payload[field]["scenario"] = "a different scenario"
        else:
            payload[field] = "a-different-identity"
        return payload

    supervisor, fake, _ = make_supervisor(tmp_path, overrides={"research": wrong_identity})
    state = run(supervisor)
    assert_not_published(state, fake)
    assert len(fake.calls) == 1


@pytest.mark.parametrize("damage", ["chunk", "source"])
def test_same_evidence_id_with_different_content_never_silently_overwrites(tmp_path, damage):
    def conflicting_evidence(request, payload):
        if damage == "chunk":
            payload["chunks"][0]["text"] += " Changed document content."
        else:
            next(iter(payload["sources"].values()))["title"] = "Another source with the same ID"
        return payload

    supervisor, fake, _ = make_supervisor(tmp_path, overrides={"market": conflicting_evidence})
    state = run(supervisor)
    assert_not_published(state, fake)
    assert state["last_error"]["code"] == "artifact_mismatch"
    assert not fake.requests("writer")


@pytest.mark.parametrize("role", ["research", "writer", "evaluator", "publish"])
@pytest.mark.parametrize("code", ["api_error", "uncertain_request"])
def test_explicit_node_failure_is_terminal_without_automatic_resend(tmp_path, role, code):
    def fail(request, payload):
        if role == "research":
            result = response("market_failed", request)
            result["view"] = request["view"]
        elif role == "writer":
            result = {**payload, "status": "failed", "markdown": None, "report": None,
                      "joined": None, "chunks": [], "sources": {}}
        else:
            result = response("evaluation_error" if role == "evaluator" else "publish_failed", request)
            if role == "evaluator":
                result["report_request_id"] = request["report_result"]["request_id"]
        result["error"] = {"code": code, "message": "Offline failure", "retryable": False}
        return result

    supervisor, fake, _ = make_supervisor(tmp_path, overrides={role: fail})
    state = run(supervisor)
    assert state["status"] in {"incomplete", "needs_attention"}
    assert len(fake.requests(role)) == 1
    assert state["last_error"]["code"] == code
    if code == "uncertain_request":
        assert state["status"] == "needs_attention"
        assert state["pending"]["role"] == role
        before = len(fake.calls)
        assert run(supervisor, resume=True)["status"] == "needs_attention"
        assert len(fake.calls) == before
    if role != "publish":
        assert not fake.requests("publish")


@pytest.mark.parametrize("code", ["budget_exceeded", "input_budget_exceeded", "uncertain_request"])
def test_cost_and_uncertain_errors_are_not_retried_or_reported_as_success(tmp_path, code):
    def fail(request, payload):
        result = response("market_failed", request)
        result["view"] = request["view"]
        result["error"] = {"code": code, "message": "Offline boundary failure", "retryable": False}
        return result

    supervisor, fake, _ = make_supervisor(tmp_path, overrides={"research": fail})
    state = run(supervisor)
    assert_not_published(state, fake)
    assert len(fake.calls) == 1
    assert state["last_error"]["code"] == code


@pytest.mark.parametrize("kind", ["worker", "writer", "supervisor"])
def test_attempt_and_step_limits_end_incomplete_without_extra_calls(tmp_path, kind):
    if kind == "worker":
        limits = Limits(worker_attempts=1)
        overrides = {"market": lambda request, payload: response("market_insufficient", request)}
    elif kind == "writer":
        limits = Limits(writer_attempts=1)
        overrides = {"evaluator": quality_rework}
    else:
        limits = Limits(supervisor_steps=1)
        overrides = {}
    supervisor, fake, _ = make_supervisor(tmp_path, overrides=overrides, limits=limits)
    state = run(supervisor)
    assert state["status"] == "incomplete"
    assert not fake.requests("publish")
    assert len(fake.requests("market")) <= 1
    assert len(fake.requests("writer")) <= 1
    if kind == "supervisor":
        assert state["step_count"] <= limits.supervisor_steps


@pytest.mark.parametrize("damage", ["missing", "digest", "markdown"])
def test_publication_cannot_complete_with_missing_or_changed_output_files(tmp_path, damage):
    def invalid_publication(request, payload):
        if damage == "missing":
            payload["pdf_ref"]["path"] = "report/nonexistent.pdf"
        elif damage == "digest":
            payload["pdf_ref"]["sha256"] = "0" * 64
        else:
            replacement = b"A different report than the evaluator reviewed"
            (tmp_path / payload["markdown_ref"]["path"]).write_bytes(replacement)
            payload["markdown_ref"]["sha256"] = hashlib.sha256(replacement).hexdigest()
        return payload

    supervisor, fake, _ = make_supervisor(tmp_path, overrides={"publish": invalid_publication})
    state = run(supervisor)
    assert state["status"] in {"incomplete", "needs_attention"}
    assert len(fake.requests("publish")) == 1


@pytest.mark.parametrize("hook_fails", [False, True])
def test_optional_observability_hook_gets_only_metadata_and_cannot_break_the_run(tmp_path, hook_fails):
    records = []

    def observe(record):
        records.append(record)
        if hook_fails:
            raise RuntimeError("Offline telemetry is unavailable")

    supervisor, _, _ = make_supervisor(tmp_path, on_event=observe)
    assert run(supervisor)["status"] == "completed"
    assert records
    allowed = {"run_id", "node", "request_id", "attempt", "step_count", "status",
               "next_action", "reason_code", "error_code"}
    assert all(set(record) <= allowed for record in records)
    assert example("research_ok")["chunks"][0]["text"] not in json.dumps(records)
    if hook_fails:
        assert "telemetry_error" in (tmp_path / "events.jsonl").read_text()


def gateway_request():
    request = {field: example("supervisor_research")[field] for field in COMMON}
    return {**request, "allowed_actions": ["market", "stop"], "summaries": {},
            "attempts": {"research": 1}, "feedback": {},
            "results": {"research": example("research_ok")}}


def test_gateway_decider_passes_current_evidence_and_published_schema_to_the_existing_gateway():
    calls = []

    class FakeGateway:
        def generate(self, purpose, instructions, content, schema):
            calls.append((purpose, instructions, json.loads(content), schema))
            return json.dumps(example("supervisor_research"))

    request = gateway_request()
    result = GatewayDecider(FakeGateway())(request)
    assert isinstance(result, contracts.SupervisorDecision)
    assert len(calls) == 1
    purpose, instructions, content, schema = calls[0]
    assert request["request_id"] in purpose
    assert instructions.strip()
    assert content == request
    assert schema == contracts.SupervisorDecision.model_json_schema()


@pytest.mark.parametrize("field", ["run_id", "request_id", "attempt", "context"])
def test_gateway_decider_rejects_a_response_for_a_different_request(field):
    class FakeGateway:
        def generate(self, *args):
            payload = example("supervisor_research")
            if field == "attempt":
                payload[field] += 1
            elif field == "context":
                payload[field]["domain"] = "another domain"
            else:
                payload[field] = "another-request"
            return json.dumps(payload)

    with pytest.raises(ValueError):
        GatewayDecider(FakeGateway())(gateway_request())


def test_gateway_budget_failure_is_not_retried_by_the_supervisor(tmp_path):
    from rag.budget import BudgetExceeded

    calls = []

    class OverBudgetGateway:
        def generate(self, *args):
            calls.append(args)
            raise BudgetExceeded("offline reserved budget exceeded")

    supervisor, fake, _ = make_supervisor(tmp_path, decider=GatewayDecider(OverBudgetGateway()))
    state = run(supervisor)
    assert_not_published(state, fake)
    assert not fake.calls
    assert len(calls) == 1
    assert state["last_error"]["code"] == "budget_exceeded"
