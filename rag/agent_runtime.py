"""Issue #4 entry point for a team Supervisor with injected team nodes."""
from pathlib import Path
import re

import yaml

from agents.contracts import RunContext
from agents.store import RunStore
from agents.supervisor import Nodes, Supervisor
from rag.tracing import trace_agent_call, trace_decision, trace_run


def load_run_context(path):
    """Read the agreed run scope from the team config, without model calls."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping")
    return RunContext.model_validate({key: config.get(key) for key in
                                      ("technologies", "domain", "scenario")}).model_dump(mode="json")


def run_team_agent(*, config_path, run_id, identity, nodes, decide, settings,
                   output_root, resume=False, limits=None, on_event=None):
    """Execute the real graph; callbacks supply #2 and #3 implementations.

    The returned state contains only control data and artifact references. Complete
    decisions and reasons remain in RunStore, while LangSmith gets safe codes.
    """
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Agent run_id must be a safe output directory name")
    context = load_run_context(config_path)
    store = RunStore(Path(output_root) / run_id)

    wrapped = Nodes(
        research=lambda request: trace_agent_call(request["view"], request, nodes.research),
        write_report=lambda request: trace_agent_call("writer", request, nodes.write_report),
        evaluate_report=lambda request: trace_agent_call("evaluator", request, nodes.evaluate_report),
        publish=nodes.publish,
    )

    def record_event(event):
        if event.get("node") == "supervisor" and event.get("reason_code"):
            trace_decision(event)
        if on_event is not None:
            on_event(event)

    supervisor = Supervisor(wrapped, decide, store, limits=limits, on_event=record_event)
    with trace_run(settings, run_id) as metadata:
        state = supervisor.run(run_id, context, identity, resume=resume)
        metadata["status"] = state["status"]
    return state
