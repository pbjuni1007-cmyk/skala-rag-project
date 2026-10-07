"""Deterministic tests for the initial follow-up research runner."""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from agents.researchers.followup import run_followup
from agents.researchers.prompts import DEFAULT_WEB_QUERIES
from agents.researchers.result import contract_chunk, cited_artifacts
from rag.evidence import PERSPECTIVE_FACETS
from tests.test_pipeline_improvements import (
    TECHS,
    assessment,
    pipeline,
    perspective_assessment,
)


WEB_QUOTE = "The registered snapshot describes a bounded public deployment example."
WEB_TEXT = WEB_QUOTE + " It does not establish adoption by a particular organization."
QUESTIONS = ["기업 문서 검토의 적용 조건과 검증 부담을 확인해 주세요"]


def _source(source_id, source_type):
    return {
        "type": source_type,
        "authors": "Example authors",
        "title": f"Source {source_id}",
        "url": f"https://example.test/{source_id}",
        "version": "v1",
        "date": "2026-01-01",
        "accessed_at": "2026-10-07T00:00:00+00:00",
    }


def _web_chunk(chunk_id, text=WEB_TEXT):
    return {
        "id": chunk_id,
        "source_id": "web",
        "technology": "both",
        "text": text,
        "page": None,
        "section": "snapshot block 1, character 0",
    }


def _followup_assessment(view, *, unknown_facet=None, web_id="web-default"):
    result = perspective_assessment(view)
    if unknown_facet is not None:
        claim_to_change = next(
            item for item in result["claims"] if item["facet"] == unknown_facet
        )
        claim_to_change.update(kind="unknown", references=[])
        result["status"] = "insufficient"
        result["gaps"] = [f"{unknown_facet}: 검색 범위에서 근거 미확인"]
    else:
        result["claims"][0]["references"] = [{"chunk_id": web_id, "quote": WEB_QUOTE}]
    return result


def _make_pipeline(tmp_path, view, responses):
    p = pipeline(tmp_path, responses)
    p.config["perspective_web_token_budget"] = 1000
    p.config["perspective_expanded_token_budget"] = 1000
    p.corpus.sources = {
        "KIVI": _source("KIVI", "paper_pool"),
        "InfiniGen": _source("InfiniGen", "paper_pool"),
        "web": _source("web", "external_web"),
    }
    p.corpus.source_metadata = lambda: {
        source_id: {key: value for key, value in source.items() if key in {"title", "url", "version", "date"}}
        for source_id, source in p.corpus.sources.items()
    }
    web_default = _web_chunk("web-default")
    web_question = _web_chunk("web-question", "Question search found a separate public snapshot passage.")
    expanded = _web_chunk("web-expanded", "Expanded search found another bounded public snapshot passage.")
    search_calls = []

    def web_search(query, top_k, **kwargs):
        search_calls.append((query, top_k, kwargs))
        if kwargs.get("expanded_only"):
            return [expanded]
        if query == DEFAULT_WEB_QUERIES[view]:
            return [web_default]
        return [web_question]

    p.corpus.web_search = web_search
    p.corpus.web_search_calls = search_calls
    research_context = SimpleNamespace(
        assessments={technology: assessment(technology) for technology in TECHS},
        chunks=[contract_chunk(chunk) for chunk in p.corpus.chunks],
    )
    request = SimpleNamespace(
        request_id=f"run-1:{view}:1",
        view=view,
        context=SimpleNamespace(
            technologies=list(TECHS),
            domain="기업 문서 검토",
            scenario="장문 사업 문서의 근거 확인",
        ),
        questions=list(QUESTIONS),
        feedback=[],
        research_context=research_context,
    )
    return p, request


@pytest.mark.parametrize("view", ["market", "stakeholder", "domain"])
def test_initial_followup_searches_default_and_questions_and_keeps_only_cited_result_chunks(tmp_path, view):
    response = _followup_assessment(view)
    p, request = _make_pipeline(tmp_path, view, [response])

    assessments, chunk_lookup = run_followup(p, request)
    result = assessments[view]

    assert result["status"] == "ok"
    assert len(result["claims"]) == 6
    assert [(query, top_k, options) for query, top_k, options in p.corpus.web_search_calls] == [
        (DEFAULT_WEB_QUERIES[view], 8, {}),
        (" ".join(QUESTIONS), 8, {}),
    ]
    purpose, _, payload, _ = p.gateway.calls[0]
    assert purpose == view
    assert payload["questions"] == QUESTIONS
    assert payload["feedback"] == []
    assert payload["web_query"] == DEFAULT_WEB_QUERIES[view]
    assert {chunk["id"] for chunk in payload["chunks"]} == {
        "KIVI", "InfiniGen", "web-default", "web-question"
    }

    cited_chunks, cited_sources = cited_artifacts(
        {view: result}, chunk_lookup, p.corpus.sources
    )
    cited_ids = {
        reference["chunk_id"]
        for item in result["claims"]
        for reference in item["references"]
    }
    assert {chunk.id for chunk in cited_chunks} == cited_ids
    assert cited_ids == {"KIVI", "InfiniGen", "web-default"}
    assert "web-question" not in cited_ids
    assert set(cited_sources) == {"KIVI", "InfiniGen", "web"}
    assert set(chunk_lookup) == {"KIVI", "InfiniGen", "web-default", "web-question"}
    balance_log = json.loads((tmp_path / "retrieval" / "balance.json").read_text())
    assert balance_log["views"][view]["diagnostics"]["technologies"]["KIVI"] == {
        "distinct_sources": 2,
        "max_source_share": pytest.approx(2 / 3),
    }
    assert balance_log["views"][view]["diagnostics"]["technologies"]["InfiniGen"] == {
        "distinct_sources": 1,
        "max_source_share": 1.0,
    }


@pytest.mark.parametrize("view", ["market", "stakeholder", "domain"])
def test_unknown_facet_is_researched_once_and_preserved_as_insufficient_with_gaps(tmp_path, view):
    missing = sorted(PERSPECTIVE_FACETS[view])[0]
    initial = _followup_assessment(view, unknown_facet=missing)
    rewrite = {"queries": [{"facet": missing, "query": "official evidence for missing facet"}]}
    reassessed = deepcopy(initial)
    p, request = _make_pipeline(tmp_path, view, [initial, rewrite, reassessed])

    assessments, chunk_lookup = run_followup(p, request)
    result = assessments[view]

    assert result["status"] == "insufficient"
    assert any(missing in gap for gap in result["gaps"])
    assert [call[0] for call in p.gateway.calls] == [
        view, view + "_rewrite", view + "_reassessment"
    ]
    assert p.corpus.web_search_calls[-1] == (
        "official evidence for missing facet", 6, {"expanded_only": True}
    )
    assert "web-expanded" in chunk_lookup
    assert len(result["claims"]) == 6
    assert [
        claim for claim in result["claims"] if claim["facet"] != missing
    ] == [claim for claim in initial["claims"] if claim["facet"] != missing]


@pytest.mark.parametrize("view", ["market", "stakeholder", "domain"])
def test_web_search_failure_is_classified_as_retrieval_failure(tmp_path, view):
    from agents.researchers.result import RetrievalFailure

    p, request = _make_pipeline(tmp_path, view, [])

    def fail(*args, **kwargs):
        raise RuntimeError("fake web search failure")

    p.corpus.web_search = fail
    with pytest.raises(RetrievalFailure):
        run_followup(p, request)
    log = json.loads((tmp_path / "retrieval" / f"{view}.json").read_text())
    assert log["error_type"] == "RetrievalFailure"


@pytest.mark.parametrize("view", ["market", "stakeholder", "domain"])
def test_research_context_chunk_mismatch_is_an_artifact_error(tmp_path, view):
    from agents.researchers.result import ArtifactMismatch

    p, request = _make_pipeline(tmp_path, view, [])
    request.research_context.chunks[0].text = "changed corpus content"

    with pytest.raises(ArtifactMismatch):
        run_followup(p, request)


@pytest.mark.parametrize("failure_call", [1, 3], ids=["initial-context", "expanded-context"])
def test_token_context_failure_is_classified_as_retrieval_failure(tmp_path, failure_call):
    from agents.researchers.result import RetrievalFailure

    view = "market"
    missing = sorted(PERSPECTIVE_FACETS[view])[0]
    initial = _followup_assessment(view, unknown_facet=missing)
    rewrite = {"queries": [{"facet": missing, "query": "official evidence for missing facet"}]}
    responses = [initial, rewrite] if failure_call == 3 else []
    p, request = _make_pipeline(tmp_path, view, responses)
    calls = 0

    def count_tokens(text):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("fake token counter failure")
        return len(text.split())

    p.corpus.evidence_tokens = count_tokens
    with pytest.raises(RetrievalFailure):
        run_followup(p, request)
