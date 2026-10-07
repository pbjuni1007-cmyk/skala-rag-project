"""Team Supervisor assembly, immutable execution inputs and checkpoint recovery."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

import yaml

from agents.contracts import RunContext
from agents.store import RunStore
from agents.supervisor import Nodes, Supervisor
from rag.budget import Budget, write_json
from rag.tracing import trace_agent_call, trace_decision, trace_run


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_identity():
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in ("app.py", "uv.lock", "pyproject.toml")]
    for directory, pattern in (("agents", "*.py"), ("rag", "*.py"), ("prompts", "*.txt"),
                               ("config", "*.yaml"), ("assets/fonts", "*.ttf")):
        files.extend((root / directory).rglob(pattern))
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def execution_identity(settings, config, corpus, index, budget):
    effective = {**settings.public(), **{key: str(settings.integer(key, default)) for key, default in (
        ("RAG_MAX_API_CALLS_PER_RUN", 40), ("OPENAI_TIMEOUT_SECONDS", 60))},
        "OPENAI_MAX_RETRIES": settings.get("OPENAI_MAX_RETRIES", "2")}
    for key in ("REPORT_CAMPUS", "REPORT_CLASS", "REPORT_CONTRIBUTORS"):
        effective[key] = settings.get(key)
    for key, default in (("LANGSMITH_TRACING", "false"), ("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
                         ("LANGSMITH_PROJECT", "skala-rag-project"), ("LANGSMITH_WORKSPACE_ID", "")):
        effective[key] = settings.get(key, default)
    for key, default in (("REPORT_FONT_PATH", "assets/fonts/NanumGothic-Regular.ttf"),
                         ("REPORT_FONT_BOLD_PATH", "assets/fonts/NanumGothic-Bold.ttf")):
        path = Path(settings.get(key, default)).resolve()
        effective[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    index_files = {}
    if hasattr(corpus, "index"):
        for name in ("index.json", "vectors.npy", "embedding.json"):
            path = Path(corpus.index) / name
            index_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"code": code_identity(), "config": config, "settings": effective, "index": index,
            "index_files": index_files, "chunks_sha256": digest(corpus.chunks),
            "retrieval": corpus.retrieval_identity(), "budget": budget,
            "sources": {key: {field: value for field, value in source.items()
                               if field not in {"accessed_at", "local_path", "raw_path"}}
                        for key, source in corpus.sources.items()}}


def load_run_context(path):
    """Read the agreed run scope from the team config, without model calls."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping")
    return RunContext.model_validate({key: config.get(key) for key in
                                      ("technologies", "domain", "scenario")}).model_dump(mode="json")


def run_team_agent(*, config_path, run_id, identity, nodes, decide, settings,
                   output_root, resume=False, limits=None, on_event=None, pause_after=None):
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
    if resume:
        saved = store.load(supervisor._identity(run_id, context, identity))
        # No trace upload, model call, regeneration or publication for terminal or
        # ambiguous checkpoints. Supervisor still verifies the lock and artifacts.
        if saved["pending"] or saved["status"] != "running":
            return supervisor.run(run_id, context, identity, resume=True)
    else:
        if (store.root / "snapshot.json").exists():
            raise ValueError("Run exists; resume it or choose a new run directory")
        (store.root / "graph.mmd").write_text(supervisor.compile().get_graph().draw_mermaid(), encoding="utf-8")
    with trace_run(settings, run_id) as metadata:
        state = supervisor.run(run_id, context, identity, resume=resume, pause_after=pause_after)
        metadata["status"] = state["status"]
    return state


def execute_agent(*, config_path, settings, corpus, index, retrieval_check, output_root,
                  run_id=None, resume=None, limits=None, pause_after=None):
    """Compose real team roles around one Gateway and one cumulative budget.

    ``resume`` restores the same control checkpoint. It is intentionally separate
    from the legacy RAG call cache, which starts a new execution.
    """
    from agents.decision import GatewayDecider
    from agents.researchers.agent import ResearchAgent
    from rag.graph import Pipeline
    from rag.llm import Gateway
    from rag.render import publish

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    load_run_context(config_path)
    output_root = Path(output_root).resolve()
    previous = None
    if resume is not None:
        out = Path(resume).resolve()
        if out.parent != output_root or (run_id is not None and run_id != out.name):
            raise ValueError("Resume must use the original run under the configured output directory")
        run_id = out.name
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("mode") != "agent" or previous.get("run_id") != run_id:
            raise ValueError("Resume requires an Agent checkpoint, not a RAG call cache")
    else:
        run_id = run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6])
        out = output_root / run_id
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Agent run_id must be a safe output directory name")
    if resume is None and out.exists():
        raise ValueError("Run directory exists; resume it or choose another run_id")

    budget = Budget(settings)
    bound_budget = budget.bind_identity(previous["identity"]["budget"] if previous else None)
    identity = execution_identity(settings, config, corpus, index, bound_budget)
    fingerprint = digest(identity)
    if previous:
        sources = json.loads((out / "sources.json").read_text(encoding="utf-8"))
        if (digest(previous["identity"]) != previous["fingerprint"] or
                previous["fingerprint"] != fingerprint or digest(sources) != previous["sources_sha256"]):
            raise ValueError("Resume rejected: code, inputs, sources, settings or dependencies changed")
        # prepare_sources observes PDFs again. Keep the original collection metadata
        # so completed and resumed views share exactly the same source registry.
        corpus.sources = sources
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "sources.json", corpus.sources)
        write_json(out / "manifest.json", {
            "mode": "agent", "run_id": run_id, "fingerprint": fingerprint, "identity": identity,
            "sources_sha256": digest(corpus.sources), "index": index,
            "retrieval_validation": retrieval_check,
        })
    budget.bind_history(out / "budget-history.json", resume=resume is not None)
    gateway = Gateway(settings, run_id, out)
    gateway.budget = budget
    pipeline = Pipeline(settings, config, corpus, gateway, out)
    nodes = Nodes(ResearchAgent(pipeline).research, pipeline.write_report_node,
                  pipeline.evaluate_report_node, lambda request: publish(request, out, settings))
    state = run_team_agent(config_path=config_path, run_id=run_id, identity=fingerprint, nodes=nodes,
                           decide=GatewayDecider(gateway), settings=settings, output_root=output_root,
                           resume=resume is not None, limits=limits, pause_after=pause_after)
    summary = {"run": str(out), "status": state["status"], "budget": budget.summary(),
               "human_review_pending": state["status"] == "completed"}
    if resume is None or not (out / "execution.json").exists() or state["status"] == "running":
        write_json(out / "execution.json", summary)
    elif json.loads((out / "execution.json").read_text()).get("status") != state["status"]:
        write_json(out / "execution.json", summary)
    return {**summary, "state": state}
