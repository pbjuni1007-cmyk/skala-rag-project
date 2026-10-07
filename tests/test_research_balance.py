"""Deterministic bias-balance findings and prompt policy checks."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from agents.researchers import prompts
from agents.researchers.result import balance_findings


EXAMPLES = json.loads(Path("docs/agent-contract-examples.json").read_text(encoding="utf-8"))["examples"]
QUOTE = "A complete source quote supporting a narrow claim."


def payload(name):
    return deepcopy(EXAMPLES[name]["payload"])


def claim(technology, facet, chunk_ids, *, kind="source_fact", caveats="실험 범위와 미확인 조건을 적음"):
    return {
        "technology": technology,
        "facet": facet,
        "text": "제공된 근거 범위의 판단",
        "kind": kind,
        "references": [{"chunk_id": chunk_id, "quote": QUOTE} for chunk_id in chunk_ids],
        "conditions": "확인한 자료 범위",
        "caveats": caveats,
    }


def market_balance_input():
    assessment = {
        "status": "ok",
        "claims": [
            claim("KIVI", "adoption", ["kivi-a"]),
            claim("KIVI", "costs", ["kivi-a", "kivi-b"]),
            claim("InfiniGen", "adoption", ["infini-a"]),
            claim("InfiniGen", "costs", ["infini-a"]),
        ],
        "conflicts": [],
        "gaps": [],
    }
    chunks = [
        {"id": "kivi-a", "source_id": "kivi-paper-a"},
        {"id": "kivi-b", "source_id": "kivi-paper-b"},
        {"id": "infini-a", "source_id": "infinigen-paper-a"},
    ]
    return assessment, chunks


def assessment_for(view):
    if view == "research":
        result = payload("research_ok")["assessments"]
        claims = result["research_kivi"]["claims"] + result["research_infinigen"]["claims"]
        return {"status": "ok", "claims": claims, "conflicts": [], "gaps": []}
    return payload(f"{view}_ok")["assessments"][view]


@pytest.mark.parametrize(
    ("view", "technology", "facet"),
    [
        ("research", "KIVI", "limitation"),
        ("market", "KIVI", "costs"),
        ("domain", "InfiniGen", "risks"),
    ],
)
def test_unknown_counter_facet_adds_gap_and_marks_assessment_insufficient(view, technology, facet):
    assessment = assessment_for(view)
    counter = next(
        item for item in assessment["claims"]
        if item["technology"] == technology and item["facet"] == facet
    )
    counter["kind"] = "unknown"

    updated, diagnostics = balance_findings(view, assessment)

    expected = f"{technology} {facet}: 한계·위험 근거 미확인"
    assert updated["status"] == "insufficient"
    assert expected in updated["gaps"]
    assert diagnostics["view"] == view
    assert set(updated) == {"status", "claims", "conflicts", "gaps"}


def test_missing_caveats_adds_one_deduplicated_gap_per_technology():
    assessment, chunks = market_balance_input()
    for item in assessment["claims"]:
        if item["technology"] == "KIVI":
            item["caveats"] = "  "
    assessment["gaps"] = ["기존 공백"]
    assessment["searched_chunks"] = chunks

    updated, _ = balance_findings("market", assessment)

    assert updated["status"] == "insufficient"
    assert updated["gaps"] == ["기존 공백", "KIVI: 한계·반대 근거를 기재한 주장이 없음"]


def test_empty_research_assessment_uses_technology_hint_for_balance_gaps():
    assessment = {
        "technology": "InfiniGen",
        "status": "insufficient",
        "claims": [],
        "conflicts": [],
        "gaps": ["검색 결과 없음: limitation 질의 결과 없음"],
    }

    updated, diagnostics = balance_findings("research", assessment)

    assert updated["status"] == "insufficient"
    assert updated["gaps"] == [
        "검색 결과 없음: limitation 질의 결과 없음",
        "InfiniGen: 한계·반대 근거를 기재한 주장이 없음",
        "InfiniGen limitation: 한계·위험 근거 미확인",
    ]
    assert diagnostics["technologies"]["InfiniGen"] == {
        "distinct_sources": 0,
        "max_source_share": 0.0,
    }


def test_stakeholder_has_no_counter_facet_rule_when_caveats_are_present():
    assessment = assessment_for("stakeholder")

    updated, diagnostics = balance_findings("stakeholder", assessment)

    assert updated == assessment
    assert diagnostics["technologies"] == {
        "KIVI": {"distinct_sources": 0, "max_source_share": 0.0},
        "InfiniGen": {"distinct_sources": 0, "max_source_share": 0.0},
    }


def test_no_findings_preserves_ok_status_and_computes_source_concentration():
    assessment, chunks = market_balance_input()
    original = deepcopy(assessment)

    updated, diagnostics = balance_findings("market", {**assessment, "chunks": chunks})

    assert updated == original
    assert updated["status"] == "ok"
    assert diagnostics["technologies"]["KIVI"]["distinct_sources"] == 2
    assert diagnostics["technologies"]["KIVI"]["max_source_share"] == pytest.approx(2 / 3)
    assert diagnostics["technologies"]["InfiniGen"] == {
        "distinct_sources": 1,
        "max_source_share": 1.0,
    }


def test_balance_policy_rules_are_present_in_research_prompts():
    prompt_text = "\n".join(
        [
            prompts.RESEARCH_PLAN_PROMPT,
            prompts.RETRIEVAL_REVIEW_PROMPT,
            prompts.RESEARCH_ASSESSMENT_PROMPT,
            prompts.RETRIEVAL_REWRITE_PROMPT,
            *prompts.PERSPECTIVE_PROMPTS.values(),
            prompts.PERSPECTIVE_REWRITE_PROMPT,
            prompts.PERSPECTIVE_REASSESSMENT_PROMPT,
        ]
    )

    assert "검색 범위에서 반대 근거 미확인" in prompt_text
    assert "`source_fact`와 `author_reported_result`" in prompt_text
    assert "`team_inference` 또는 `scenario`" in prompt_text
    assert "검색해 제공되지 않은 내용은 `unknown`" in prompt_text
    assert "feedback이 있으면 각 항목을 검색과 분석에 빠짐없이 반영하라" in prompt_text
    assert "해결되지 않은 항목은 구체적인 이유와 함께 gaps에 남겨라" in prompt_text
    assert "영어 ASCII 문장으로 바꾸라" in prompts.RESEARCH_PLAN_PROMPT
