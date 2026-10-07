"""Feedback-driven re-research tests for Issue #13."""

from copy import deepcopy
import json
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agents.researchers.followup import run_followup
from agents.researchers.maturity import run_maturity
from agents.researchers.prompts import DEFAULT_WEB_QUERIES
from rag.evidence import CORE_FACETS, PERSPECTIVE_FACETS


QUOTE = "The experiment uses a documented laboratory GPU configuration."
TECHS = ("KIVI", "InfiniGen")
CONTEXT = {
    "technologies": list(TECHS),
    "domain": "문서 검토",
    "scenario": "업무 가정",
}
FEEDBACK = ["실험 조건과 적용 한계를 보강해 주세요."]
QUESTIONS = ["시장 도입 조건과 대안을 확인해 주세요."]


def paper_chunk(technology):
    return {
        "id": technology,
        "technology": technology,
        "source_id": technology,
        "page": 1,
        "section": None,
        "text": QUOTE + " The source also describes separate batch and accuracy experiments.",
    }


def web_chunk(chunk_id="previous-web"):
    return {
        "id": chunk_id,
        "technology": "both",
        "source_id": chunk_id,
        "page": None,
        "section": "Frozen snapshot / market evidence",
        "text": QUOTE + " The snapshot describes adoption constraints and operating costs.",
    }


def claim(technology, facet, chunk_id=None, *, kind="source_fact", text=None):
    references = [] if chunk_id is None else [{"chunk_id": chunk_id, "quote": QUOTE}]
    claim_text = text or f"근거 기반 {technology} {facet} 분석"
    conditions = "공통 장비 A100; 별도 정확도 실험과 배치 조건은 병합하지 않음"
    caveats = "기업 운용 검증은 확인되지 않음"
    if facet == "maturity":
        kind = "team_inference"
        claim_text = "공개 실험실 근거로 추정한 TRL 4"
    if kind == "unknown":
        claim_text = f"{technology} {facet} 근거 미확인"
    return {
        "technology": technology,
        "facet": facet,
        "text": claim_text,
        "kind": kind,
        "references": references,
        "conditions": conditions,
        "caveats": caveats,
    }


def tech_assessment(technology, *, status="ok"):
    return {
        "status": status,
        "claims": [claim(technology, facet, technology) for facet in sorted(CORE_FACETS)],
        "conflicts": [],
        "gaps": ["조건 근거 보완 필요"] if status == "insufficient" else [],
    }


def perspective_assessment(view, *, unknown_facets=(), paper_ids=None, web_id=None):
    paper_ids = paper_ids or {technology: technology for technology in TECHS}
    claims = []
    for technology in TECHS:
        for facet in sorted(PERSPECTIVE_FACETS[view]):
            if facet in unknown_facets:
                claims.append(claim(technology, facet, kind="unknown"))
            else:
                chunk_id = web_id if facet == "costs" and web_id else paper_ids[technology]
                claims.append(claim(technology, facet, chunk_id))
    has_unknown = any(item["kind"] == "unknown" for item in claims)
    return {
        "status": "insufficient" if has_unknown else "ok",
        "claims": claims,
        "conflicts": [],
        "gaps": ["adoption 근거 미확인"] if has_unknown else [],
    }


def core_plan(suffix=""):
    facets = ("mechanism", "limitation", "conditions", "maturity")
    return {
        "queries": [
            {"technology": technology, "facet": facet, "query": f"{technology} {facet}{suffix}"}
            for technology in TECHS
            for facet in facets
        ]
    }


def feedback_queries(technology, suffix=" revised"):
    return {
        "queries": [
            {"technology": technology, "facet": facet, "query": f"{technology} {facet}{suffix}"}
            for facet in ("mechanism", "limitation", "conditions", "maturity")
        ]
    }


def retrieval_review(technology):
    return {
        "items": [
            {
                "facet": facet,
                "sufficient": True,
                "chunk_ids": [technology],
                "reason": "해당 facet의 원문 근거 확인",
                "missing": [],
            }
            for facet in ("mechanism", "limitation", "conditions", "maturity")
        ]
    }


def _contract_paper(raw):
    value = deepcopy(raw)
    value["section"] = None
    return value


def research_result(*, status="ok", assessments=None, chunks=None, request_id="research-1", view="research"):
    if assessments is None:
        assessments = {
            "research_kivi": tech_assessment("KIVI"),
            "research_infinigen": tech_assessment("InfiniGen"),
        }
    if chunks is None:
        chunks = [_contract_paper(paper_chunk(technology)) for technology in TECHS]
    return {
        "contract_version": "agent-contract-v1",
        "run_id": "run-1",
        "request_id": request_id,
        "attempt": 1,
        "context": deepcopy(CONTEXT),
        "view": view,
        "status": status,
        "assessments": deepcopy(assessments),
        "chunks": deepcopy(chunks),
        "sources": {},
        "error": None,
    }


class FakeCorpus:
    def __init__(self, chunks, *, web_responses=None):
        self.chunks = deepcopy(chunks)
        self.search_calls = []
        self.web_calls = []
        self.web_responses = web_responses or {}

    def search(self, query, technology, top_k):
        self.search_calls.append((query, technology, top_k))
        return [chunk for chunk in self.chunks if chunk.get("technology") == technology]

    def web_search(self, query, top_k, expanded_only=False):
        self.web_calls.append((query, top_k, expanded_only))
        return deepcopy(self.web_responses.get(query, []))


class FakePipeline:
    def __init__(self, out, corpus, responses):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.corpus = corpus
        self.responses = deepcopy(responses)
        self.config = {"top_k": 3, "perspective_web_token_budget": 2000}
        self.settings = SimpleNamespace(integer=lambda _name, default: default)
        self.search_lock = Lock()
        self.calls = []

    def structured(self, purpose, schema, instructions, content, check=None):
        self.calls.append({
            "purpose": purpose,
            "schema": schema,
            "instructions": instructions,
            "payload": deepcopy(content),
        })
        if purpose not in self.responses:
            raise AssertionError(f"No fake response configured for {purpose}")
        response = self.responses[purpose]
        if isinstance(response, list):
            response = response.pop(0)
        return deepcopy(response)

    def metadata(self):
        return {}

    def _preflight(self, *_args, **_kwargs):
        # Force reassess_facets through its ordinary single structured call.
        return None


def _request(view, *, previous_result, research_context=None, feedback=None):
    return {
        "contract_version": "agent-contract-v1",
        "run_id": "run-1",
        "request_id": f"{view}-2",
        "attempt": 2,
        "context": deepcopy(CONTEXT),
        "view": view,
        "questions": deepcopy(QUESTIONS),
        "feedback": deepcopy(FEEDBACK if feedback is None else feedback),
        "previous_result": deepcopy(previous_result),
        "research_context": deepcopy(research_context),
    }


def _assessment_for_rewrite(previous, *, new_web_id="new-web"):
    value = deepcopy(previous)
    value["status"] = "ok"
    value["gaps"] = []
    value["claims"] = [
        claim(technology, facet, new_web_id, text=f"feedback 반영 {technology} {facet}")
        if facet == "adoption"
        else deepcopy(old_claim)
        for old_claim in previous["claims"]
        for technology, facet in [(old_claim["technology"], old_claim["facet"])]
    ]
    return value


def test_research_feedback_targets_non_ok_technology_and_preserves_other_assessment(tmp_path):
    raw_chunks = [paper_chunk(technology) for technology in TECHS]
    old_assessments = {
        "research_kivi": tech_assessment("KIVI", status="insufficient"),
        "research_infinigen": tech_assessment("InfiniGen"),
    }
    previous = research_result(status="insufficient", assessments=old_assessments, request_id="research-1")
    current_out = tmp_path / "research" / "research" / "research-2"
    prior_out = current_out.parent / previous["request_id"] / "retrieval"
    prior_out.mkdir(parents=True)
    prior_queries = [{"technology": "KIVI", "facet": "conditions", "query": "old conditions query"}]
    (prior_out / "research.json").write_text(json.dumps({
        "technology_queries": {"KIVI": prior_queries, "InfiniGen": []}
    }), encoding="utf-8")
    responses = {
        "rewrite_kivi_feedback": feedback_queries("KIVI", " feedback revised"),
        "retrieval_review_kivi_0": retrieval_review("KIVI"),
        "research_kivi_0": tech_assessment("KIVI"),
    }
    pipeline = FakePipeline(current_out, FakeCorpus(raw_chunks), responses)

    assessments, chunk_lookup = run_maturity(pipeline, _request("research", previous_result=previous))

    purposes = [call["purpose"] for call in pipeline.calls]
    assert purposes == ["rewrite_kivi_feedback", "retrieval_review_kivi_0", "research_kivi_0"]
    rewrite = pipeline.calls[0]["payload"]
    assert rewrite["feedback"] == FEEDBACK
    assert rewrite["questions"] == QUESTIONS
    assert rewrite["previous_assessment"] == old_assessments["research_kivi"]
    assert rewrite["previous_queries"] == prior_queries
    rewrite_schema = pipeline.calls[0]["schema"]
    rewrite_schema.model_validate(feedback_queries("KIVI"))
    with pytest.raises(ValidationError):
        rewrite_schema.model_validate(core_plan())
    assert rewrite_schema.model_json_schema()["properties"]["queries"]["maxItems"] == 4
    assert {technology for _, technology, _ in pipeline.corpus.search_calls} == {"KIVI"}
    assert {query for query, _, _ in pipeline.corpus.search_calls} == {
        item["query"] for item in feedback_queries("KIVI", " feedback revised")["queries"]
    }
    assert pipeline.calls[1]["payload"]["feedback"] == FEEDBACK
    assert pipeline.calls[2]["payload"]["feedback"] == FEEDBACK
    assert assessments["research_infinigen"] == old_assessments["research_infinigen"]
    assert "InfiniGen" in chunk_lookup


def test_research_feedback_reruns_both_technologies_when_previous_assessments_are_ok(tmp_path):
    raw_chunks = [paper_chunk(technology) for technology in TECHS]
    previous = research_result(request_id="research-1")
    responses = {}
    for technology in TECHS:
        key = technology.lower()
        responses[f"rewrite_{key}_feedback"] = feedback_queries(technology)
        responses[f"retrieval_review_{key}_0"] = retrieval_review(technology)
        responses[f"research_{key}_0"] = tech_assessment(technology)
    pipeline = FakePipeline(tmp_path, FakeCorpus(raw_chunks), responses)

    assessments, _ = run_maturity(pipeline, _request("research", previous_result=previous))

    assert {call["purpose"] for call in pipeline.calls if call["purpose"].endswith("_feedback")} == {
        "rewrite_kivi_feedback", "rewrite_infinigen_feedback"
    }
    assert {technology for _, technology, _ in pipeline.corpus.search_calls} == set(TECHS)
    assert set(assessments) == {"research_kivi", "research_infinigen"}


def test_failed_previous_research_uses_initial_plan_with_feedback(tmp_path):
    failed_previous = research_result(
        status="failed", assessments={}, chunks=[], request_id="research-1"
    )
    failed_previous.update(
        sources={},
        error={"code": "api_error", "message": "research: APIError", "retryable": False},
    )
    responses = {"research_queries": core_plan(" feedback")}
    for technology in TECHS:
        key = technology.lower()
        responses[f"retrieval_review_{key}_0"] = retrieval_review(technology)
        responses[f"research_{key}_0"] = tech_assessment(technology)
    pipeline = FakePipeline(tmp_path, FakeCorpus([paper_chunk(t) for t in TECHS]), responses)

    run_maturity(pipeline, _request("research", previous_result=failed_previous))

    assert pipeline.calls[0]["purpose"] == "research_queries"
    assert pipeline.calls[0]["payload"]["feedback"] == FEEDBACK
    assert not any(call["purpose"].endswith("_feedback") for call in pipeline.calls)
    assert {technology for _, technology, _ in pipeline.corpus.search_calls} == set(TECHS)


def test_followup_feedback_researches_rewritten_facets_and_preserves_other_facets(tmp_path):
    papers = [paper_chunk(technology) for technology in TECHS]
    inherited_result = research_result()
    previous_market = perspective_assessment(
        "market", unknown_facets={"adoption"}, web_id="previous-web"
    )
    previous_market_result = {
        "contract_version": "agent-contract-v1",
        "run_id": "run-1",
        "request_id": "market-1",
        "attempt": 1,
        "context": deepcopy(CONTEXT),
        "view": "market",
        "status": "insufficient",
        "assessments": {"market": deepcopy(previous_market)},
        "chunks": [_contract_paper(chunk) for chunk in papers] + [web_chunk("previous-web")],
        "sources": {},
        "error": None,
    }
    new_web = web_chunk("new-web")
    responses = {
        "market_rewrite": {
            "queries": [{"facet": "adoption", "query": "adoption feedback query"}]
        },
        "market_reassessment": _assessment_for_rewrite(previous_market),
    }
    corpus = FakeCorpus(papers, web_responses={
        "adoption feedback query": [web_chunk("previous-web"), new_web],
    })
    pipeline = FakePipeline(tmp_path, corpus, responses)

    assessments, chunk_lookup = run_followup(
        pipeline,
        _request("market", previous_result=previous_market_result, research_context=inherited_result),
    )

    assert [call["purpose"] for call in pipeline.calls] == ["market_rewrite", "market_reassessment"]
    rewrite_payload = pipeline.calls[0]["payload"]
    assert rewrite_payload["feedback"] == FEEDBACK
    assert rewrite_payload["questions"] == QUESTIONS
    assert rewrite_payload["previous_assessment"] == previous_market
    unknown_facets = {claim["facet"] for claim in previous_market["claims"] if claim["kind"] == "unknown"}
    rewritten_facets = {item["facet"] for item in responses["market_rewrite"]["queries"]}
    assert unknown_facets <= rewritten_facets
    assert corpus.web_calls == [("adoption feedback query", 6, False)]
    reassessment = pipeline.calls[1]["payload"]
    assert reassessment["feedback"] == FEEDBACK
    assert reassessment["questions"] == QUESTIONS
    assert reassessment["missing_facets"] == ["adoption"]
    assert reassessment["previous_assessment"] == previous_market
    assert {"KIVI", "InfiniGen", "previous-web", "new-web"} <= set(chunk_lookup)
    assert sum(chunk["id"] == "previous-web" for chunk in chunk_lookup.values()) == 1
    assert assessments["market"]["claims"] == _assessment_for_rewrite(previous_market)["claims"]
    old_other_facets = [c for c in previous_market["claims"] if c["facet"] != "adoption"]
    new_other_facets = [c for c in assessments["market"]["claims"] if c["facet"] != "adoption"]
    assert new_other_facets == old_other_facets


def test_changed_paper_citations_fall_back_to_full_followup_with_feedback(tmp_path):
    papers = [paper_chunk(technology) for technology in TECHS]
    old_paper = {
        **paper_chunk("KIVI"),
        "id": "old-paper",
        "source_id": "old-paper-source",
        "page": 8,
    }
    previous_market = perspective_assessment(
        "market", paper_ids={"KIVI": "old-paper", "InfiniGen": "InfiniGen"}
    )
    previous_market_result = {
        "contract_version": "agent-contract-v1",
        "run_id": "run-1",
        "request_id": "market-1",
        "attempt": 1,
        "context": deepcopy(CONTEXT),
        "view": "market",
        "status": "ok",
        "assessments": {"market": previous_market},
        "chunks": [_contract_paper(chunk) for chunk in papers] + [_contract_paper(old_paper)],
        "sources": {},
        "error": None,
    }
    responses = {"market": perspective_assessment("market")}
    corpus = FakeCorpus(papers)
    pipeline = FakePipeline(tmp_path, corpus, responses)

    assessments, chunk_lookup = run_followup(
        pipeline,
        _request("market", previous_result=previous_market_result, research_context=research_result()),
    )

    assert [call["purpose"] for call in pipeline.calls] == ["market"]
    assert pipeline.calls[0]["payload"]["feedback"] == FEEDBACK
    assert [call[0] for call in corpus.web_calls] == [
        DEFAULT_WEB_QUERIES["market"],
        " ".join(QUESTIONS),
    ]
    assert "old-paper" not in chunk_lookup
    assert set(assessments) == {"market"}
