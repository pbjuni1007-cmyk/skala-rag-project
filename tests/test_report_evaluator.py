from copy import deepcopy
import json

from rag.evaluator import CRITERIA, evaluate_report
from rag.evidence import collect_evidence
from rag.render import build_report_markdown
from rag.schemas import EvaluationRequest, ReportResult
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


def refreshed_joined(joined, report):
    source_claims = {key: value for key, value in joined["claims"].items()
                     if value.get("perspective") != "synthesis"}
    source_evidence_ids = {evidence_id for claim in source_claims.values()
                           for evidence_id in claim.get("evidence_ids", [])}
    synthesis, evidence = collect_evidence({"synthesis": {"claims": report["synthesis_claims"]}}, report_chunks())
    return {**joined, "claims": {**source_claims, **synthesis},
            "evidence": {key: value for key, value in joined["evidence"].items()
                         if key in source_evidence_ids} | evidence}


def evaluation_request(joined, report, attempt=1, markdown=None):
    joined = refreshed_joined(joined, report)
    context = {"technologies": ["KIVI", "InfiniGen"], "domain": "문서 검토", "scenario": "업무 가정"}
    sources = {}
    for chunk in report_chunks():
        sources[chunk["source_id"]] = {
            "type": "paper_pool", "authors": "가상 테스트 작성자", "title": f"{chunk['source_id']} 테스트 자료",
            "url": f"https://example.invalid/{chunk['source_id']}", "version": "mock-v1", "date": "2026",
            "accessed_at": "2026-10-07T00:00:00Z",
        }
    generated, details = build_report_markdown(report, joined, sources, context)
    used_chunk_ids = {joined["evidence"][evidence_id]["chunk_id"]
                      for claim_id in details["used_claims"]
                      for evidence_id in joined["claims"][claim_id]["evidence_ids"]}
    report_result = ReportResult.model_validate({
        "contract_version": "agent-contract-v1", "run_id": "test-run", "request_id": "test-run:writer:1",
        "attempt": 1, "context": context, "status": "ok", "markdown": generated if markdown is None else markdown,
        "report": report, "joined": joined,
        "chunks": [chunk for chunk in report_chunks() if chunk["id"] in used_chunk_ids],
        "sources": sources, "error": None,
    })
    return EvaluationRequest.model_validate({
        "contract_version": "agent-contract-v1", "run_id": "test-run",
        "request_id": f"test-run:evaluator:{attempt}", "attempt": attempt, "context": context,
        "report_result": report_result,
    })


def evaluation_fixture(tmp_path):
    _, joined, _, report = report_fixture(tmp_path)
    return refreshed_joined(joined, report), report


def test_hybrid_evaluation_passes_four_criteria_and_keeps_check_scope_explicit(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    calls = []
    request = evaluation_request(joined, report)
    result = evaluate_report(structured_with(judge_result(), calls), request)

    assert result["status"] == "ok" and result["passed"] is True
    assert result["method"] == "hybrid"
    assert set(result["checks"]) == set(CRITERIA)
    assert all(check["passed"] for check in result["checks"].values())
    assert result["report_request_id"] == request.report_result.request_id
    assert calls[0][0] == "report_quality_judge"
    assert calls[0][2]["claims"]
    assert calls[0][2]["source_contexts"]
    assert calls[0][2]["mechanical_checks"]["neutrality"]["passed"] is None
    assert calls[0][2]["source_distribution"]["unique_source_count"] == 2


def test_semantically_unsupported_claim_returns_perspective_repair_request(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    claim = joined["claims"]["market-1"]
    finding = {"criterion": "groundedness", "target": "market-1", "claim_ids": ["market-1"],
               "evidence_ids": claim["evidence_ids"], "issue": "인용은 있지만 시장 채택 주장을 뒷받침하지 않습니다.",
               "revision_request": "market-1 문장을 인용이 실제로 말하는 범위로 줄이고 채택 여부는 미확인으로 표시하세요."}
    result = evaluate_report(structured_with(judge_result({"groundedness": {
        "status": "revise", "rationale": "인용과 주장의 의미가 일치하지 않습니다.", "findings": [finding]}})),
        evaluation_request(joined, report))

    assert result["checks"]["groundedness"]["passed"] is False
    assert result["checks"]["groundedness"]["claim_ids"] == ["market-1"]
    assert result["passed"] is False
    assert result["repair_requests"][0]["target"] == "market"
    assert result["repair_requests"][0]["claim_ids"] == ["market-1"]
    assert finding["revision_request"] in result["repair_requests"][0]["reason"]


def test_unconditional_winner_language_routes_to_writer(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    claim = joined["claims"]["synthesis-1"]
    finding = {"criterion": "neutrality", "target": "synthesis-1", "claim_ids": ["synthesis-1"],
               "evidence_ids": claim["evidence_ids"], "issue": "조건을 밝히지 않고 한 기술을 최선이라고 단정합니다.",
               "revision_request": "synthesis-1에서 우열 단정을 빼고 판단 기준과 적용 조건을 명시하세요."}
    result = evaluate_report(structured_with(judge_result({"neutrality": {
        "status": "revise", "rationale": "조건에 따른 비교가 필요합니다.", "findings": [finding]}})),
        evaluation_request(joined, report))

    assert result["checks"]["neutrality"]["passed"] is False
    assert result["repair_requests"][0]["target"] == "writer"
    assert "synthesis-1" in result["repair_requests"][0]["claim_ids"]


def test_source_concentration_is_context_for_judge_not_an_automatic_bias_failure(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    used = joined["claims"]["market-1"]["evidence_ids"]
    finding = {"criterion": "bias_control", "target": "source distribution", "claim_ids": ["market-1"],
               "evidence_ids": used, "issue": "시장성 주장이 한 출처에 집중되어 독립 근거가 부족합니다.",
               "revision_request": "시장성 문장의 결론을 해당 출처의 범위로 제한하고 독립 근거가 없는 점을 밝히세요."}
    calls = []
    result = evaluate_report(structured_with(judge_result({"bias_control": {
        "status": "revise", "rationale": "한 출처의 반복 인용을 검토했습니다.", "findings": [finding]}}), calls),
        evaluation_request(joined, report))

    assert calls[0][2]["source_distribution"]["source_reference_counts"]
    assert result["checks"]["bias_control"]["passed"] is False
    assert result["repair_requests"][0]["target"] == "market"


def test_missing_domain_perspective_creates_concrete_coverage_repair(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    incomplete = deepcopy(report)
    incomplete["sections"][3]["claim_ids"] = []
    result = evaluate_report(structured_with(judge_result()), evaluation_request(joined, incomplete))

    assert result["status"] == "ok" and result["passed"] is False
    assert result["checks"]["coverage"]["passed"] is False
    domain_request = next(item for item in result["repair_requests"] if item["target"] == "domain")
    assert domain_request["gap_ids"]


def test_revised_report_can_be_evaluated_again_with_a_new_request_id(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    first = evaluate_report(structured_with(judge_result({"neutrality": {
        "status": "revise", "rationale": "조건이 부족합니다.", "findings": [{
            "criterion": "neutrality", "target": "synthesis-1", "claim_ids": ["synthesis-1"],
            "evidence_ids": joined["claims"]["synthesis-1"]["evidence_ids"],
            "issue": "우열 문장이 조건을 밝히지 않습니다.", "revision_request": "조건을 명시하세요."}]}})),
        evaluation_request(joined, report))
    revised = deepcopy(report)
    revised["synthesis_claims"][0]["text"] = "적용 조건을 충족하면 두 기술의 선택지를 각각 검토할 수 있습니다."
    second = evaluate_report(structured_with(judge_result()), evaluation_request(joined, revised, attempt=2))

    assert first["passed"] is False
    assert second["status"] == "ok" and second["passed"] is True
    assert second["attempt"] == 2
    assert second["report_request_id"] == "test-run:writer:1"


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
    joined, report = evaluation_fixture(tmp_path)
    invalid = judge_result({"groundedness": {
        "status": "revise", "rationale": "근거가 부족합니다.", "findings": [{
            "criterion": "groundedness", "target": "invented", "claim_ids": ["invented"],
            "evidence_ids": [], "issue": "대상 claim이 입력에 없습니다.", "revision_request": "입력에 있는 주장만 평가하세요."}]}})
    def invalid_judge(purpose, schema, instructions, content, check):
        parsed = schema.model_validate(invalid).model_dump()
        errors = check(parsed)
        if errors:
            raise ValueError("; ".join(errors))
        return parsed

    result = evaluate_report(invalid_judge, evaluation_request(joined, report))

    assert result["status"] == "failed"
    assert result["passed"] is False and result["checks"] == {}
    assert result["repair_requests"] == []
    assert result["error"]["code"] == "invalid_response"


def test_markdown_mismatch_is_an_evaluation_execution_failure(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    request = evaluation_request(joined, report, markdown="# SUMMARY\nmodified")
    result = evaluate_report(structured_with(judge_result()), request)

    assert result["status"] == "failed"
    assert result["passed"] is False and result["checks"] == {}
    assert result["error"]["code"] == "artifact_mismatch"


def test_evaluator_payload_excludes_environment_and_full_source_files(tmp_path):
    joined, report = evaluation_fixture(tmp_path)
    calls = []
    evaluate_report(structured_with(judge_result(), calls), evaluation_request(joined, report))
    serialized = json.dumps(calls[0][2], ensure_ascii=False)
    assert "OPENAI_API_KEY" not in serialized
    assert "source_metadata" not in serialized


def test_graph_writer_and_evaluator_emit_the_shared_contract(tmp_path):
    pipeline, joined, _, report = report_fixture(tmp_path)

    def writer_structured(purpose, schema, instructions, content, check):
        if purpose == "synthesis_report":
            value = {key: item for key, item in report.items() if key != "gap_decisions"}
        else:
            value = {"gap_decisions": [
                {"gap_id": gap["id"], "status": "unresolved", "resolution": "추가 근거가 필요합니다.", "claim_ids": []}
                for gap in content["gap_records"]
            ]}
        parsed = schema.model_validate(value).model_dump()
        errors = check(parsed)
        assert not errors, errors
        return parsed

    pipeline.structured = writer_structured
    writer_state = pipeline.synthesize({"joined": joined, "run_config": {"run_id": "graph-test"}})

    assert writer_state["run_status"] == "report_drafted"
    assert writer_state["report_result"]["status"] == "ok"
    assert writer_state["report_result"]["request_id"] == "graph-test:writer:1"
    assert writer_state["markdown"].startswith("# SUMMARY")
    assert "# REFERENCE" in writer_state["markdown"]
    saved_request = json.loads((tmp_path / "nodes/report_writer_request.json").read_text())
    assert set(saved_request["results"]) == {"research", "market", "stakeholder", "domain"}
    assert set(saved_request["results"]["research"]["assessments"]) == {"research_kivi", "research_infinigen"}

    pipeline.structured = structured_with(judge_result())
    evaluation_state = pipeline.evaluate_report({**writer_state, "run_config": {"run_id": "graph-test"}})
    result = evaluation_state["evaluation_result"]
    assert result["status"] == "ok" and result["passed"] is True
    assert result["report_request_id"] == writer_state["report_result"]["request_id"]
    assert evaluation_state["run_status"] == "validated"
    saved_result = json.loads((tmp_path / "quality_evaluation.json").read_text())
    assert set(saved_result["checks"]) == set(CRITERIA)
