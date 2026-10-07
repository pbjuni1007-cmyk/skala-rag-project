"""State-driven coordination for the team contract; subordinates never call each other."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from agents.contracts import (
    COMMON_FIELDS, EvaluationRequest, EvaluationResult, PublishRequest, PublishResult,
    ReportResult, ResearchRequest, ResearchResult, RunContext, SupervisorDecision,
    WriteRequest, VIEWS,
)
from agents.state import AgentState, Limits
from agents.store import RunStore, encoded


@dataclass(frozen=True)
class Nodes:
    research: Callable
    write_report: Callable
    evaluate_report: Callable
    publish: Callable


def merge_registry(results, name):
    merged = {}
    for result in results:
        items = ((c["id"], c) for c in result[name]) if name == "chunks" else result[name].items()
        for key, value in items:
            if key in merged and merged[key] != value:
                raise ValueError("artifact_mismatch: conflicting " + name + " ID")
            merged[key] = value
    return merged


def summarize(result):
    assessments = list(result["assessments"].values())
    claims = [claim for assessment in assessments for claim in assessment["claims"]]
    return {
        "status": result["status"], "request_id": result["request_id"],
        "claim_count": len(claims),
        "evidence_count": len({r["chunk_id"] for c in claims for r in c["references"]}),
        "gaps": [(f"[{name}] " + gap)[:300]
                 for name, assessment in result["assessments"].items() for gap in assessment["gaps"]][:8],
        "findings": [(f"[{c['technology']} / {c['facet']}] " + c["text"])[:220] for c in claims][:8],
    }


class Supervisor:
    def __init__(self, nodes: Nodes, decide, store: RunStore, limits=None, *, on_event=None):
        self.nodes, self.decide, self.store = nodes, decide, store
        self.limits = limits or Limits()
        # Optional #4 metadata hook; full decisions are written only to local artifacts.
        self.on_event = on_event

    def initial(self, run_id, context, identity):
        return AgentState(
            contract_version="agent-contract-v1", run_id=run_id, context=context, identity=identity,
            status="running", next_action="supervisor", step_count=0, attempts={}, pending=None,
            last_error=None, feedback={}, evidence_sufficient=False, summaries={}, results={},
            report=None, evaluation=None, publication=None, repair_source=None,
        )

    def _identity(self, run_id, context, identity):
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Provide the subordinate code/data/config fingerprint")
        root = Path(__file__).resolve().parents[1]
        files = [*sorted((root / "agents").glob("*.py")), root / "prompts/supervisor.txt", root / "uv.lock"]
        code = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        return hashlib.sha256(encoded({
            "run_id": run_id, "context": context, "caller": identity,
            "limits": self.limits.model_dump(), "code": code,
        })).hexdigest()

    def _common(self, state, role, attempt):
        return {
            "contract_version": state["contract_version"], "run_id": state["run_id"],
            "request_id": f"{state['identity'][:16]}-{role}-{attempt}",
            "attempt": attempt, "context": deepcopy(state["context"]),
        }

    def _event(self, state, role, **values):
        metadata = {
            "run_id": state["run_id"], "node": role,
            "request_id": state["pending"]["request_id"] if state["pending"] else None,
            "attempt": state["pending"]["attempt"] if state["pending"] else None,
            "step_count": state["step_count"], "status": state["status"],
            **values,
        }
        self.store.event(**metadata)
        if self.on_event is not None:
            # Reasons/feedback may contain source text, so the external hook gets only codes.
            try:
                self.on_event({k: v for k, v in metadata.items()
                               if k in {"run_id", "node", "request_id", "attempt", "step_count",
                                        "status", "next_action", "reason_code", "error_code", "evidence_sufficient"}})
            except Exception:
                self.store.event(run_id=state["run_id"], node=role, status="telemetry_error")

    def _stop(self, code, message, *, uncertain=False):
        return {"status": "needs_attention" if uncertain else "incomplete", "next_action": "stop",
                "last_error": {"code": code, "message": message, "retryable": False}}

    def choices(self, state):
        if state["status"] != "running" or state["last_error"]:
            return [], "fatal_error"
        if state["step_count"] >= self.limits.supervisor_steps:
            return [], "limit_exceeded"
        if state["evaluation"]:
            verdict = self.store.get(state["evaluation"])
            if verdict["status"] == "ok" and verdict["passed"]:
                return ["publish"], "quality_passed"
        if state["report"] and state["evaluation"] is None:
            return ["evaluator"], "report_ready"
        ready = {v for v, s in state["summaries"].items() if s["status"] == "ok"}
        if "research" not in ready or "research" in state["feedback"]:
            targets = ["research"]
        else:
            targets = [v for v in VIEWS if v in state["feedback"]]
            targets = targets or [v for v in VIEWS if v not in ready]
        if targets:
            if any(state["attempts"].get(v, 0) >= self.limits.worker_attempts for v in targets):
                return [], "limit_exceeded"
            if state["repair_source"]:
                reason = "quality_rework"
            elif any(v in state["results"] for v in targets):
                reason = "evidence_gap"
            else:
                reason = "initial_research" if targets == ["research"] else "missing_view"
            return targets, reason
        if state["attempts"].get("writer", 0) >= self.limits.writer_attempts:
            return [], "limit_exceeded"
        if state["feedback"].get("writer"):
            return ["writer"], "quality_rework"
        return ["writer", *[v for v in VIEWS if state["attempts"].get(v, 0) < self.limits.worker_attempts]], "evidence_ready"

    def _call(self, callback, request, result_type):
        raw = callback(deepcopy(request))
        result = result_type.model_validate(raw).model_dump(mode="json")
        if any(result[k] != request[k] for k in COMMON_FIELDS):
            raise ValueError("Response does not match its request identity")
        return result

    def supervise(self, state):
        if state["status"] != "running":
            return {"next_action": "stop"}
        allowed, code = self.choices(state)
        if not allowed:
            update = self._stop(code, "작업 상한 또는 실행 오류로 완료할 수 없습니다.")
            self._event(state, "supervisor", next_action="stop", reason_code=code)
            return update
        common = self._common(state, "supervisor", state["step_count"] + 1)
        if allowed in (["evaluator"], ["publish"]):
            decision = SupervisorDecision(
                **common, next_action=allowed[0], evidence_sufficient=True,
                reason_code=code, reason="현재 보고서의 평가·출력 선행 조건을 확인했습니다.", feedback=[],
            ).model_dump(mode="json")
        else:
            request = {
                **common, "allowed_actions": [*allowed, "stop"], "summaries": state["summaries"],
                "attempts": state["attempts"], "feedback": state["feedback"],
                "results": {v: self.store.get(ref) for v, ref in state["results"].items()},
            }
            decision = self._call(self.decide, request, SupervisorDecision)
        action = decision["next_action"]
        if action not in [*allowed, "stop"]:
            raise ValueError("Supervisor selected an ineligible action")
        if action == "writer" and not decision["evidence_sufficient"]:
            raise ValueError("Writing requires the Supervisor's explicit sufficiency judgment")
        self.store.put("supervisor", common["attempt"], decision)
        self._event(state, "supervisor", next_action=action,
                    reason_code=decision["reason_code"], reason=decision["reason"],
                    evidence_sufficient=decision["evidence_sufficient"])
        feedback = deepcopy(state["feedback"])
        if decision["feedback"]:
            feedback[action] = [s[:500] for s in decision["feedback"][:8]]
        if action in VIEWS and state["attempts"].get(action, 0):
            feedback.setdefault(action, [decision["reason"][:500]])
        update = {"next_action": action, "step_count": common["attempt"], "feedback": feedback,
                  "evidence_sufficient": decision["evidence_sufficient"]}
        if action == "stop":
            update.update(self._stop("evidence_gap", "Supervisor가 근거 부족으로 중단했습니다."))
        return update

    def _feedback(self, state, role):
        # Complete repair reasons and IDs remain in the immutable evaluator artifact,
        # even after refreshing a view invalidates the current report/evaluation.
        notes = list(state["feedback"].get(role, []))
        if state["repair_source"] and role in state["feedback"]:
            for repair in self.store.get(state["repair_source"])["repair_requests"]:
                if repair["target"] == role:
                    notes.append(f"{repair['reason']} | claim_ids={repair['claim_ids']} | gap_ids={repair['gap_ids']}")
        if role in state["results"] and role in state["feedback"]:
            for assessment in self.store.get(state["results"][role])["assessments"].values():
                notes.extend(assessment["gaps"])
        return list(dict.fromkeys(notes))

    def worker(self, view, state):
        attempt = state["attempts"][view]
        request = ResearchRequest(
            **self._common(state, view, attempt), view=view,
            questions=[f"{state['context']['domain']} 시나리오에서 두 기술의 {view} 관점 근거·조건·한계는 무엇인가?"],
            feedback=self._feedback(state, view),
            previous_result=self.store.get(state["results"][view]) if view in state["results"] else None,
            research_context=None if view == "research" else self.store.get(state["results"]["research"]),
        ).model_dump(mode="json")
        result = self._call(self.nodes.research, request, ResearchResult)
        if result["view"] != view:
            raise ValueError("Research result returned the wrong view")
        remaining = {} if view == "research" else {v: ref for v, ref in state["results"].items() if v != view}
        inputs = [self.store.get(ref) for ref in remaining.values()] + [result]
        for registry in ("chunks", "sources"):
            merge_registry(inputs, registry)
        ref = self.store.put(view, attempt, result)
        feedback = deepcopy(state["feedback"])
        feedback.pop(view, None)
        if result["status"] == "insufficient":
            feedback[view] = ["필수 근거를 보완하고 기존 gaps 각각에 답하세요."]
        summaries = {} if view == "research" else dict(state["summaries"])
        summaries[view] = summarize(result)
        update = {
            "results": {**remaining, view: ref}, "summaries": summaries, "feedback": feedback,
            "report": None, "evaluation": None, "publication": None,
            "evidence_sufficient": False, "next_action": "supervisor",
        }
        if result["status"] == "failed":
            update.update(self._stop(result["error"]["code"], result["error"]["message"],
                                     uncertain=result["error"]["code"] == "uncertain_request"))
        return update

    def writer(self, state):
        if not state["evidence_sufficient"]:
            raise ValueError("Writing requires an explicit sufficiency judgment")
        results = {v: self.store.get(ref) for v, ref in state["results"].items()}
        request = WriteRequest(
            **self._common(state, "writer", state["attempts"]["writer"]),
            results=results, feedback=self._feedback(state, "writer"),
            previous_report=self.store.get(state["report"]) if state["report"] else None,
        ).model_dump(mode="json")
        result = self._call(self.nodes.write_report, request, ReportResult)
        if result["status"] == "ok":
            self._check_report_inputs(result, results)
        ref = self.store.put("writer", state["attempts"]["writer"], result)
        update = {"report": ref, "evaluation": None, "publication": None, "repair_source": None,
                  "feedback": {}, "next_action": "supervisor"}
        if result["status"] == "failed":
            update.update(self._stop(result["error"]["code"], result["error"]["message"],
                                     uncertain=result["error"]["code"] == "uncertain_request"))
        return update

    def _check_report_inputs(self, result, results):
        from rag.conflicts import conflict_record_errors
        from rag.evidence import collect_evidence
        assessments = {k: a for r in results.values() for k, a in r["assessments"].items()}
        chunks = merge_registry(results.values(), "chunks")
        sources = merge_registry(results.values(), "sources")
        if result["joined"]["assessments"] != assessments:
            raise ValueError("Report assessments changed after research")
        for c in result["chunks"]:
            if chunks.get(c["id"]) != c:
                raise ValueError("Report introduced or changed source text")
        for key, source in result["sources"].items():
            if sources.get(key) != source:
                raise ValueError("Report introduced or changed a source")
        old_refs = {(r["chunk_id"], r["quote"]) for a in assessments.values()
                    for c in a["claims"] for r in c["references"]}
        for claim in result["report"]["synthesis_claims"]:
            if any((r["chunk_id"], r["quote"]) not in old_refs for r in claim["references"]):
                raise ValueError("Synthesis introduced new evidence")
        synthesis = {"claims": result["report"]["synthesis_claims"]}
        expected_claims, expected_evidence = collect_evidence(
            {**assessments, "synthesis": synthesis}, list(chunks.values()))
        if result["joined"]["claims"] != expected_claims or result["joined"]["evidence"] != expected_evidence:
            raise ValueError("Report changed claim/evidence content or IDs")
        for key in ("gaps", "conflicts"):
            expected = [value for a in assessments.values() for value in a[key]]
            if result["joined"][key] != expected:
                raise ValueError("Report changed original " + key)
        gaps = [{"id": f"gap-{name}-{i}", "perspective": name, "text": gap}
                for name, assessment in assessments.items() for i, gap in enumerate(assessment["gaps"], 1)]
        if result["joined"]["gap_records"] != gaps:
            raise ValueError("Report changed original gap records")
        if conflict_record_errors(result["joined"]["conflict_records"], expected_claims, assessments):
            raise ValueError("Report changed conflict provenance")

    def evaluate(self, state):
        report = self.store.get(state["report"])
        request = EvaluationRequest(
            **self._common(state, "evaluator", state["attempts"]["evaluator"]),
            report_result=report,
        ).model_dump(mode="json")
        result = self._call(self.nodes.evaluate_report, request, EvaluationResult)
        if result["report_request_id"] != report["request_id"]:
            raise ValueError("Evaluation belongs to a different report")
        claim_ids = set(report["joined"]["claims"])
        gap_ids = {g["id"] for g in report["joined"]["gap_records"]}
        for item in [*result["checks"].values(), *result["repair_requests"]]:
            if not set(item["claim_ids"]) <= claim_ids or not set(item["gap_ids"]) <= gap_ids:
                raise ValueError("Evaluation references unknown claim/gap IDs")
        if result["passed"]:
            from rag.evidence import report_errors
            if report_errors(report["report"], report["joined"]["claims"],
                             report["chunks"], report["joined"]["gap_records"]):
                raise ValueError("Positive evaluation cannot override structural report errors")
        ref = self.store.put("evaluator", state["attempts"]["evaluator"], result)
        feedback = defaultdict(list)
        for repair in result["repair_requests"]:
            feedback[repair["target"]].append(repair["reason"][:300])
        update = {"evaluation": ref, "repair_source": ref if result["repair_requests"] else None,
                  "feedback": {k: v[:8] for k, v in feedback.items()}, "next_action": "supervisor"}
        if result["status"] == "failed":
            update.update(self._stop(result["error"]["code"], result["error"]["message"],
                                     uncertain=result["error"]["code"] == "uncertain_request"))
        return update

    def publish(self, state):
        request = PublishRequest(
            **self._common(state, "publish", state["attempts"]["publish"]),
            report_result=self.store.get(state["report"]),
            evaluation_result=self.store.get(state["evaluation"]),
        ).model_dump(mode="json")
        result = self._call(self.nodes.publish, request, PublishResult)
        if result["status"] == "ok":
            markdown = self.store.verify(result["markdown_ref"])
            self.store.verify(result["pdf_ref"])
            if markdown.decode("utf-8") != request["report_result"]["markdown"]:
                raise ValueError("Published Markdown differs from the evaluated report")
        ref = self.store.put("publish", state["attempts"]["publish"], result)
        if result["status"] == "failed":
            return {"publication": ref, **self._stop(
                result["error"]["code"], result["error"]["message"],
                uncertain=result["error"]["code"] == "uncertain_request")}
        return {"publication": ref, "status": "completed", "next_action": "stop"}

    def node(self, role, function):
        def run(state):
            current = deepcopy(state)
            if role == "supervisor" and current["status"] != "running":
                # Final acknowledgement is not another decision or paid operation.
                # Keep the original uncertain request visible for recovery.
                current["next_action"] = "stop"
                self.store.checkpoint(current)
                self._event(current, role, next_action="stop")
                return current
            attempt = current["step_count"] + 1 if role == "supervisor" else current["attempts"].get(role, 0) + 1
            if role != "supervisor":
                current["attempts"][role] = attempt
            common = self._common(current, role, attempt)
            current["pending"] = {"role": role, "request_id": common["request_id"], "attempt": attempt}
            self.store.checkpoint(current)
            try:
                update = function(current)
            except Exception as exc:
                from rag.llm import APIError
                codes = {"APIError": "api_error", "BudgetExceeded": "budget_exceeded",
                         "InputBudgetExceeded": "input_budget_exceeded"}
                code = "artifact_mismatch" if str(exc).startswith("artifact_mismatch:") else codes.get(type(exc).__name__, "invalid_response")
                if isinstance(exc, APIError):
                    code = exc.code
                update = self._stop(code, f"{role} 처리 실패 ({type(exc).__name__})", uncertain=code == "uncertain_request")
            current.update(update)
            # Preserve a pending marker when the provider explicitly cannot confirm completion.
            if current["status"] != "needs_attention":
                current["pending"] = None
            self.store.checkpoint(current)
            self._event({**current, "pending": {"role": role, "request_id": common["request_id"], "attempt": attempt}},
                        role, next_action=current["next_action"],
                        error_code=(current["last_error"] or {}).get("code"))
            return current
        return run

    def compile(self):
        graph = StateGraph(AgentState)
        graph.add_node("supervisor", self.node("supervisor", self.supervise))
        for view in VIEWS:
            graph.add_node(view, self.node(view, lambda state, name=view: self.worker(name, state)))
        for role, function in (("writer", self.writer), ("evaluator", self.evaluate), ("publish", self.publish)):
            graph.add_node(role, self.node(role, function))
        routes = {role: role for role in (*VIEWS, "supervisor", "writer", "evaluator", "publish")}
        routes["stop"] = END
        graph.add_conditional_edges(START, lambda state: state["next_action"], routes)
        graph.add_conditional_edges("supervisor", lambda state: state["next_action"], routes)
        for role in (*VIEWS, "writer", "evaluator", "publish"):
            graph.add_edge(role, "supervisor")
        return graph.compile()

    def run(self, run_id, context, identity, *, resume=False, pause_after=None):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be nonempty")
        if pause_after is not None and (type(pause_after) is not int or pause_after < 1):
            raise ValueError("pause_after must be a positive completed-node count")
        context = RunContext.model_validate(context).model_dump(mode="json")
        bound = self._identity(run_id, context, identity)
        # Native graph tracing would expose complete state/inputs. #4 can subscribe to
        # the explicit metadata hook instead, within its approved trace configuration.
        with self.store.locked(), tracing_context(enabled=False, parent=False):
            if resume:
                state = self.store.load(bound)
                if state["run_id"] != run_id or state["context"] != context:
                    raise ValueError("Snapshot run/context changed")
                if state["pending"]:
                    state.update(self._stop("uncertain_request", "중단된 호출의 완료 여부를 확인해야 합니다.", uncertain=True))
                    self.store.checkpoint(state)
                    return state
                if state["status"] != "running":
                    return state
            else:
                if (self.store.root / "snapshot.json").exists():
                    raise ValueError("Run exists; resume it or choose a new run directory")
                state = self.initial(run_id, context, bound)
                self.store.checkpoint(state)
            graph = self.compile()
            events = graph.stream(state, {"recursion_limit": 2 * self.limits.supervisor_steps + 8}, stream_mode="values")
            try:
                next(events)  # Initial state is not an executed node.
                for count, _ in enumerate(events, 1):
                    if pause_after is not None and count >= pause_after:
                        break
            finally:
                events.close()
            return self.store.load(bound)
