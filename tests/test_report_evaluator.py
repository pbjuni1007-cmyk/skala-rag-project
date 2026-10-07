from copy import deepcopy
import json

import pytest

from rag.evaluator import CRITERIA, evaluate_report
from rag.evidence import collect_evidence
from rag.writer import write_report
from test_pipeline_improvements import report_fixture
from test_pipeline_improvements import chunks as report_chunks


def judge_result(overrides=None):
    overrides = overrides or {}
    return {"criteria": [
        {"criterion": criterion, "status": "pass", "rationale": "제공된 자료 범위에서 기준을 충족합니다.", "findings": []}
        | overrides.get(criterion, {})
        for criterion in CRITERIA
    ]}


def structured_with(value, seen=None):
    def structured(purpose, schema, instructions, content, check):
        if seen is not None:
            seen.append((purpose, instructions, content))
        assert purpose == "report_quality_judge"
        parsed = schema.model_validate(value).model_dump()
        errors = check(parsed)
        assert not errors, errors
        return parsed
    return structured


def evaluation_fixture(tmp_path):
    _, joined, claims, report = report_fixture(tmp_path)
    synthesis, evidence = collect_evidence({"synthesis": {"claims": report["synthesis_claims"]}}, report_chunks())
    complete_joined = {**joined, "claims": {**joined["claims"], **synthesis},
                       "evidence": {**joined["evidence"], **evidence}}
    return complete_joined, report


def test_hybrid_evaluation_passes_only_after_four_judge_criteria_and_code_checks(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    calls = []
    result = evaluate_report(structured_with(judge_result(), calls), report, joined,
                             report_chunks(), {})

    assert result["status"] == "pass"
    assert result["method"] == "hybrid"
    assert [item["criterion"] for item in result["criteria"]] == list(CRITERIA)
    assert result["code_checks"]["groundedness"]["passed"] is True
    assert result["code_checks"]["perspective_coverage"]["passed"] is True
    assert result["metrics"]["unique_source_count"] == 2
    assert result["limitations"]
    assert calls[0][0] == "report_quality_judge"
    assert calls[0][2]["claims"]
    assert calls[0][2]["source_contexts"]


def test_semantically_unsupported_claim_returns_groundedness_revision_request(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    claim = joined["claims"]["market-1"]
    finding = {"criterion": "groundedness", "target": "market-1", "claim_ids": ["market-1"],
               "evidence_ids": claim["evidence_ids"], "issue": "인용은 있지만 시장 채택 주장을 뒷받침하지 않습니다.",
               "revision_request": "market-1 문장을 인용이 실제로 말하는 범위로 줄이고 채택 여부는 미확인으로 표시하세요."}
    result = evaluate_report(structured_with(judge_result({"groundedness": {
        "status": "revise", "rationale": "인용과 주장의 의미가 일치하지 않습니다.", "findings": [finding]}})),
        report, joined, report_chunks(), {})

    assert result["code_checks"]["groundedness"]["passed"] is True
    assert result["status"] == "revise"
    assert result["problems"][0]["claim_ids"] == ["market-1"]
    assert result["revision_requests"] == [finding["revision_request"]]


def test_unconditional_winner_language_fails_neutrality(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    claim = joined["claims"]["synthesis-1"]
    finding = {"criterion": "neutrality", "target": "synthesis-1", "claim_ids": ["synthesis-1"],
               "evidence_ids": claim["evidence_ids"], "issue": "조건을 밝히지 않고 한 기술을 최선이라고 단정합니다.",
               "revision_request": "synthesis-1에서 우열 단정을 빼고 판단 기준과 적용 조건을 명시하세요."}
    result = evaluate_report(structured_with(judge_result({"neutrality": {
        "status": "revise", "rationale": "조건에 따른 비교가 필요합니다.", "findings": [finding]}})),
        report, joined, report_chunks(), {})

    assert result["criteria"][1]["status"] == "revise"
    assert result["problems"][0]["criterion"] == "neutrality"
    assert "synthesis-1" in result["problems"][0]["target"]


def test_source_concentration_is_reported_and_judge_controls_bias_verdict(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    used = joined["claims"]["market-1"]["evidence_ids"]
    finding = {"criterion": "bias_control", "target": "source distribution", "claim_ids": ["market-1"],
               "evidence_ids": used, "issue": "시장성 주장이 한 출처에 집중되어 독립 근거가 부족합니다.",
               "revision_request": "시장성 문장의 결론을 해당 출처의 범위로 제한하고 독립 근거가 없는 점을 밝히세요."}
    result = evaluate_report(structured_with(judge_result({"bias_control": {
        "status": "revise", "rationale": "한 출처의 반복 인용을 검토했습니다.", "findings": [finding]}})),
        report, joined, report_chunks(), {})

    assert result["metrics"]["source_reference_counts"]
    assert result["criteria"][2]["status"] == "revise"
    assert result["problems"][0]["evidence_ids"] == used


def test_missing_domain_perspective_fails_coverage_even_if_judge_misses_it(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    incomplete = deepcopy(report)
    incomplete["sections"][3]["claim_ids"] = []
    result = evaluate_report(structured_with(judge_result()), incomplete, joined,
                             report_chunks(), {})

    assert result["status"] == "revise"
    assert result["missing_perspectives"] == ["도메인 적용"]
    coverage = next(item for item in result["criteria"] if item["criterion"] == "perspective_coverage")
    assert coverage["status"] == "revise"
    assert any("도메인 적용" in request for request in result["revision_requests"])


def test_revised_report_can_be_evaluated_again_after_supervisor_edits_it(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    first = evaluate_report(structured_with(judge_result({"neutrality": {
        "status": "revise", "rationale": "조건이 부족합니다.", "findings": [{
            "criterion": "neutrality", "target": "synthesis-1", "claim_ids": ["synthesis-1"],
            "evidence_ids": joined["claims"]["synthesis-1"]["evidence_ids"],
            "issue": "우열 문장이 조건을 밝히지 않습니다.",
            "revision_request": "조건을 명시하세요."}]}})), report, joined, report_chunks(), {})
    revised = deepcopy(report)
    revised["synthesis_claims"][0]["text"] = "적용 조건을 충족하면 두 기술의 선택지를 각각 검토할 수 있습니다."
    second = evaluate_report(structured_with(judge_result()), revised, joined, report_chunks(), {})

    assert first["status"] == "revise"
    assert second["status"] == "pass"


def test_writer_receives_supervisor_feedback_without_owning_the_retry_route(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    seen = []
    draft = {key: value for key, value in report.items() if key != "gap_decisions"}

    def structured(purpose, schema, instructions, content, check):
        seen.append((purpose, instructions, content))
        parsed = schema.model_validate(draft).model_dump()
        errors = check(parsed)
        assert not errors, errors
        return parsed

    revision_requests = ["도메인 적용의 한계 조건을 명시하세요."]
    result = write_report(structured, joined, report_chunks(), revision_requests, report)

    assert result["sections"] == report["sections"]
    assert seen[0][0] == "synthesis_report"
    assert seen[0][2]["revision_requests"] == revision_requests
    assert seen[0][2]["previous_report"] == report
    assert "Judge 평가" in seen[0][1]


def test_judge_cannot_attach_findings_to_unknown_claim_or_evidence(tmp_path):
    _, joined, _, report = report_fixture(tmp_path)
    invalid = judge_result({"groundedness": {
        "status": "revise", "rationale": "근거가 부족합니다.", "findings": [{
            "criterion": "groundedness", "target": "invented", "claim_ids": ["invented"],
            "evidence_ids": [], "issue": "대상 claim이 입력에 없습니다.", "revision_request": "입력에 있는 주장만 평가하세요."}]}})

    def structured(purpose, schema, instructions, content, check):
        parsed = schema.model_validate(invalid).model_dump()
        errors = check(parsed)
        if errors:
            raise ValueError("; ".join(errors))
        return parsed

    with pytest.raises(ValueError, match="claim that is not in the report"):
        evaluate_report(structured, report, joined, report_chunks(), {})


def test_evaluator_judge_payload_does_not_include_environment_or_full_source_files(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    calls = []
    evaluate_report(structured_with(judge_result(), calls), report, joined, report_chunks(), {})
    serialized = json.dumps(calls[0][2], ensure_ascii=False)
    assert "OPENAI_API_KEY" not in serialized
    assert "source_metadata" not in serialized
