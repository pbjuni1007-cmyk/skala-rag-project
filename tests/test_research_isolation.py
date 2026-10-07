"""Pending AST and dispatch-isolation tests for research runners."""

import ast
from pathlib import Path
from types import SimpleNamespace

from agents.researchers.agent import ResearchAgent


RESEARCHERS = Path.cwd() / "agents" / "researchers"
CONTEXT = {"technologies": ["KIVI", "InfiniGen"], "domain": "문서 검토", "scenario": "업무 가정"}


def _request(view, request_id, research_context=None):
    return {
        "contract_version": "agent-contract-v1",
        "run_id": "run-1",
        "request_id": request_id,
        "attempt": 1,
        "context": CONTEXT,
        "view": view,
        "questions": ["분석에 필요한 자료는 무엇인가?"],
        "feedback": [],
        "previous_result": None,
        "research_context": research_context,
    }


def _ok_assessment():
    return {"status": "ok", "claims": [], "conflicts": [], "gaps": []}


def test_module_scope_imports_do_not_create_agent_or_rag_graph_cycles():
    failures = []
    for path in RESEARCHERS.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    if imported == "rag.graph":
                        failures.append(f"{path.name}: module-level import rag.graph")
                    if imported.startswith("agents.") and not imported.startswith("agents.researchers"):
                        failures.append(f"{path.name}: module-level import {imported}")
                continue
            if not module:
                continue
            if module == "rag.graph":
                failures.append(f"{path.name}: module-level import rag.graph")
            if module.startswith("agents.") and not module.startswith("agents.researchers"):
                failures.append(f"{path.name}: module-level import {module}")
    assert failures == []


def test_maturity_and_followup_do_not_import_each_other():
    for filename, forbidden in (("maturity.py", "followup"), ("followup.py", "maturity")):
        tree = ast.parse((RESEARCHERS / filename).read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
        assert not any(name.endswith("." + forbidden) or name == forbidden for name in imported)


def test_one_runner_is_dispatched_for_each_view_and_market_skips_maturity(tmp_path, monkeypatch):
    import agents.researchers.agent as agent_module

    pipeline = SimpleNamespace(out=tmp_path, corpus=SimpleNamespace(sources={}))
    calls = []

    def maturity(local_pipeline, request):
        calls.append("maturity")
        return {
            "research_kivi": _ok_assessment(),
            "research_infinigen": _ok_assessment(),
        }, {}

    def followup(local_pipeline, request):
        calls.append("followup:" + request.view)
        return {request.view: _ok_assessment()}, {}

    monkeypatch.setattr(agent_module, "run_maturity", maturity)
    monkeypatch.setattr(agent_module, "run_followup", followup)
    agent = ResearchAgent(pipeline)
    research = agent.research(_request("research", "research-1"))
    assert calls == ["maturity"]

    calls.clear()
    market = agent.research(_request("market", "market-1", research))
    assert market["view"] == "market"
    assert calls == ["followup:market"]

    calls.clear()
    for view in ("stakeholder", "domain"):
        agent.research(_request(view, f"{view}-1", research))
    assert calls == ["followup:stakeholder", "followup:domain"]
