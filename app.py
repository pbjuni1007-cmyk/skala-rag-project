from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
import uuid
import yaml

from rag.budget import write_json, Budget
from rag.settings import Settings


def report_contract_hash(path=Path("config/report-contract.yaml")):
    """Bind saved runs to the public report behavior contract."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_call_reuse(path, identity, requirements_hash, index=None, current_sources=None):
    """Allow code-only changes; every generation still requires an exact request hash."""
    previous = json.loads((path / "manifest.json").read_text())
    sources = json.loads((path / "sources.json").read_text())
    prior_identity = {"config": previous["config"], "settings": previous["settings"],
        "index_hash": previous["index"]["index_hash"], "lock_sha256": previous["lock_sha256"],
        "sources": {k: {n: v.get(n) for n in ("sha256", "text_sha256", "version", "url", "status")} for k, v in sources.items()}}
    if "retrieval" in previous:
        prior_identity["retrieval"] = previous["retrieval"]
    if "report_contract_sha256" in previous:
        prior_identity["report_contract_sha256"] = previous["report_contract_sha256"]
    original = {**prior_identity, "code_sha256": previous["code_sha256"]}
    if hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest() != previous["fingerprint"]:
        raise ValueError("Call reuse rejected: prior manifest and sources do not match")
    if index is not None and previous["index"]["embedding"] != index["embedding"]:
        raise ValueError("Call reuse rejected: resolved embedding changed")
    if current_sources is not None and {k: v.get("raw_sha256") for k, v in sources.items()} != {k: v.get("raw_sha256") for k, v in current_sources.items()}:
        raise ValueError("Call reuse rejected: raw source snapshot changed")
    current = {k: v for k, v in identity.items() if k != "code_sha256"}
    if prior_identity != current or previous["requirements_sha256"] != requirements_hash:
        raise ValueError("Call reuse rejected: inputs, settings, sources, requirements or dependencies changed")


def main():
    parser = argparse.ArgumentParser(description="KIVI/InfiniGen evidence-based report agent")
    parser.add_argument("--config", default="config/run.yaml")
    parser.add_argument("--prepare", action="store_true", help="Download sources and build local embeddings, no GPT calls")
    parser.add_argument("--render", type=Path, help="Rewrite Markdown from a saved, validated run, no GPT calls")
    parser.add_argument("--refresh-web", action="store_true")
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--reuse-calls", type=Path, help="Rerun current graph; reuse only exact successful requests when non-code inputs match")
    recovery.add_argument("--resume", type=Path, help="Reuse exact-input successful calls from a compatible saved run")
    args = parser.parse_args()
    settings = Settings.load()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.render:
        from rag.render import render_report
        state = json.loads((args.render / "state.json").read_text())
        if not state.get("validation_result", {}).get("passed"):
            raise ValueError("Cannot render a run that has not passed citation validation")
        result = render_report(args.render, state["report"], state["joined"], state["source_registry"], settings, state["run_config"]["config"])
        print(json.dumps(result, ensure_ascii=False))
        return 0
    from rag.corpus import Corpus
    corpus = Corpus(settings, config)
    pages = corpus.prepare_sources(args.refresh_web)
    index = corpus.build()
    if args.prepare:
        print(json.dumps(index, ensure_ascii=False))
        return 0
    from rag.llm import Gateway
    from rag.graph import Pipeline
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = Path(settings.get("RAG_OUTPUT_DIR", "outputs")) / run_id
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path("rag").glob("*.py"))) + Path("app.py").read_bytes()).hexdigest()
    identity = {"config": config, "settings": settings.public(), "index_hash": index["index_hash"], "code_sha256": code_hash,
                "sources": {k: {n: v.get(n) for n in ("sha256", "text_sha256", "version", "url", "status")} for k, v in corpus.sources.items()},
                "retrieval": corpus.retrieval_identity(),
                "report_contract_sha256": report_contract_hash(),
                "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.resume and json.loads((args.resume / "manifest.json").read_text()).get("fingerprint") != fingerprint:
        raise ValueError("Resume rejected: code, inputs, sources, settings or dependencies changed")
    if args.reuse_calls:
        check_call_reuse(args.reuse_calls, identity, identity["report_contract_sha256"], index, corpus.sources)
    gateway = Gateway(settings, run_id, out, args.resume or args.reuse_calls)
    manifest = {"run_id": run_id, "fingerprint": fingerprint, "resumed_from": str(args.resume) if args.resume else None,
                "reused_calls_from": str(args.reuse_calls) if args.reuse_calls else None,
                "config": config, "settings": settings.public(), "index": index,
                "retrieval": identity["retrieval"],
                "requirements_sha256": identity["report_contract_sha256"],
                "report_contract_sha256": identity["report_contract_sha256"],
                "requirements_path": "config/report-contract.yaml",
                "code_sha256": code_hash, "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
                "api": gateway.lookup(), "budget_before": gateway.budget.summary()}
    write_json(out / "manifest.json", manifest)
    write_json(out / "sources.json", corpus.sources)
    pipeline = Pipeline(settings, config, corpus, gateway, out)
    graph = pipeline.compile()
    (out / "graph.mmd").write_text(graph.get_graph().draw_mermaid())
    print(f"run={out}", flush=True)
    state = graph.invoke({"run_config": manifest, "source_registry": corpus.sources},
                         {"max_concurrency": settings.integer("RAG_MAX_CONCURRENCY", 1), "recursion_limit": 30})
    state["budget_after"] = gateway.budget.summary()
    write_json(out / "state.json", state)
    print(json.dumps({"run": str(out), "status": state["run_status"], "outputs": state.get("output_paths"),
                      "budget": state["budget_after"]}, ensure_ascii=False))
    return 0 if state["run_status"] == "human_review_pending" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        # No response body, headers, environment or credential-bearing exception repr.
        print(f"Execution stopped: {type(exc).__name__}. Inspect saved node receipts.", file=sys.stderr)
        sys.exit(2)
