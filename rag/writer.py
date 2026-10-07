"""Report drafting from validated perspective claims and their source evidence."""

from copy import deepcopy

from agents.contracts import NodeError, ReportResult, WriteRequest, evidence_registry
from rag.budget import BudgetExceeded
from rag.corpus import normalized
from rag.conflicts import build_conflict_records, conflict_record_errors
from rag.evidence import (collect_evidence, gap_decision_errors, report_errors,
                          validate_assessment)
from rag.llm import APIError
from rag.payloads import synthesis_payload
from rag.request_budget import InputBudgetExceeded
from rag.schemas import GapDecisions, Report, ReportDraft


WRITER_INSTRUCTIONS = (
    "기존 claim을 배치하고 상충을 종합하라. 새 사실, 출처, 웹 검색을 추가하지 마라. "
    "각 claim의 reference_ids는 reference_table의 전체 원문 인용을 가리킨다. 종합 주장에는 해당 표의 chunk_id와 quote를 그대로 복사하라. "
    "summary_claim_ids는 중복 없이 2~3개를 고른다. 종합 claim을 최소 하나, market/stakeholder/domain 선행 claim을 최소 하나 포함하라. "
    "요약은 TRL 판단만 반복하지 말고 양 기술의 시장성, 역할별 효익과 부담, 업무 적용 조건을 함께 보여 줘야 한다. "
    "선택한 주장 본문과 인용, 기술명 표기를 합쳐 1200자 안에 담기도록 간결한 주장을 고른다. "
    "sections는 기술 성숙도, 시장성, 이해관계자, 도메인 적용, 관점 간 상충과 한계 순서로 작성하라. "
    "각 관점에는 양 기술의 claim을 배치하라. 기술 성숙도 장에는 research_로 시작하는 모든 claim을 포함해 원리, 한계, 실험조건, TRL을 보존하라. "
    "synthesis_claims는 관점 간 상충을 설명하는 2개 team_inference로, 기존 quote만 재사용한다. "
    "첫 종합은 stakeholder 평가와 conflicts에 나타난 실제 역할 간 효익과 부담의 충돌을 기술명과 함께 설명하라. "
    "둘째 종합은 그 충돌을 market의 도입 판단과 domain의 적용 조건, 검증 조건에 연결하라. "
    "어떤 조건에서 각 관점의 판단이 달라지는지를 제시하고 특정 기술을 무조건 우승자로 추천하지 마라. "
    "실험 수치끼리 직접 비교하기 어렵다는 주의만 두 종합에 반복하지 마라. 비교 한계는 해당 판단의 caveats에 남긴다. "
    "선행 근거에서 역할 충돌이 확인되지 않으면 그 범위를 명시하고 조건부 팀 해석으로 작성하라. "
    "새 종합 claim ID는 synthesis-1, synthesis-2로 sections와 summary에서 참조할 수 있다. 선행 평가의 한계와 조건을 지우지 마라. "
    "마지막 관점 간 상충과 한계 장에는 synthesis claim만 배치하라. 같은 claim을 여러 본문 장에 반복하지 마라. summary 재사용은 허용한다. "
    "기존 모든 claim을 자기 관점 본문에 한 번씩 포함하라. 공백 판정은 별도 호출이 맡으므로 수행하지 마라."
)


def source_claims(joined):
    """Drop a prior synthesis so a Supervisor can safely request a revised draft."""
    claims = {key: value for key, value in joined["claims"].items()
              if value.get("perspective") != "synthesis"}
    evidence_ids = {evidence_id for claim in claims.values() for evidence_id in claim.get("evidence_ids", [])}
    return {**joined, "claims": claims,
            "evidence": {key: value for key, value in joined["evidence"].items() if key in evidence_ids}}


def draft_report(structured, joined, chunks, revision_requests=None, previous_report=None):
    """Create the report draft; evaluation and retry routing belong to separate nodes."""
    joined = source_claims(joined)
    compact = [{key: claim[key] for key in (
        "id", "text", "technology", "kind", "facet", "perspective", "references", "caveats", "conditions")}
        for claim in joined["claims"].values()]

    def check_report(report):
        errors = validate_assessment({"claims": report["synthesis_claims"]}, chunks)
        previous_refs = {(ref["chunk_id"], normalized(ref["quote"]))
                         for claim in compact for ref in claim["references"]}
        for claim in report["synthesis_claims"]:
            if claim["kind"] != "team_inference":
                errors.append("Synthesis may only add explicitly labeled team inferences")
            if any((ref["chunk_id"], normalized(ref["quote"])) not in previous_refs
                   for ref in claim["references"]):
                errors.append("Synthesis must reuse existing evidence verbatim")
        if errors:
            return errors
        extra, _ = collect_evidence({"synthesis": {"claims": report["synthesis_claims"]}}, chunks)
        return report_errors(report, {**joined["claims"], **extra}, chunks)

    payload = synthesis_payload(compact, joined.get("conflicts", []))
    instructions = WRITER_INSTRUCTIONS
    if revision_requests:
        payload["revision_requests"] = list(revision_requests)
        payload["previous_report"] = deepcopy(previous_report or {})
        instructions += (
            " 이전 Judge 평가의 revision_requests를 모두 반영하라. 각 요청이 가리키는 claim이나 관점만 필요한 만큼 수정하고, "
            "근거 인용과 조건을 보존하라. 반영할 근거가 없으면 새 사실을 만들지 말고 해당 claim을 제한하거나 미확인으로 표현하라."
        )
    return structured("synthesis_report", ReportDraft, instructions, payload, check_report)


def review_gap_batch(structured, joined, source_metadata, index, batch):
    compact = [{key: claim[key] for key in ("id", "kind", "text", "conditions", "caveats")}
               for claim in joined["claims"].values()]
    return structured(f"synthesis_gaps_{index}", GapDecisions,
        "종합 역할의 근거 공백 재판정만 수행하라. 제공된 gap_records 각 ID를 정확히 한 번 판정하라. "
        "resolved는 공백 전체가 기존 주장으로 해소됐을 때만 사용하고 근거 claim_ids를 연결하라. "
        "unknown 주장만으로 해소하지 마라. 부분 해소, 채택, 비용 등 미확인 정보는 unresolved로 유지하라. "
        "각 resolution은 확인 범위와 남은 한계를 간결히 적고 새 사실이나 출처를 만들지 마라.",
        {"gap_records": batch, "source_metadata": source_metadata, "claims": compact},
        lambda value: gap_decision_errors(value["gap_decisions"], batch, joined["claims"]))["gap_decisions"]


class WriterArtifactMismatch(ValueError):
    """Input registries contain conflicting identifiers or report evidence."""


def _merge_results(request):
    assessments, chunk_map, source_objects = {}, {}, {}
    for result in request.results.values():
        for name, assessment in result.assessments.items():
            assessments[name] = assessment.model_dump(mode="json")
        for chunk in result.chunks:
            value = chunk.model_dump(mode="json")
            previous = chunk_map.get(chunk.id)
            previous_value = previous.model_dump(mode="json") if previous else None
            if previous_value is not None and previous_value != value:
                fields = sorted(key for key in set(value) | set(previous_value)
                                if value.get(key) != previous_value.get(key))
                raise WriterArtifactMismatch("Research views returned conflicting source chunk fields: "
                                             + ", ".join(fields))
            chunk_map[chunk.id] = chunk
        for source_id, source in result.sources.items():
            value = source.model_dump(mode="json")
            if source_id in source_objects and source_objects[source_id].model_dump(mode="json") != value:
                raise WriterArtifactMismatch("Research views returned conflicting source metadata")
            source_objects[source_id] = source
    chunks = evidence_registry(list(chunk_map.values()), source_objects)
    sources = {source_id: source.model_dump(mode="json") for source_id, source in source_objects.items()}
    claims, evidence = collect_evidence(assessments, chunks)
    joined = {
        "claims": claims, "evidence": evidence, "assessments": assessments,
        "conflicts": [item for assessment in assessments.values() for item in assessment["conflicts"]],
        "gaps": [item for assessment in assessments.values() for item in assessment["gaps"]],
        "gap_records": [{"id": f"gap-{name}-{index}", "perspective": name, "text": gap}
                        for name, assessment in assessments.items()
                        for index, gap in enumerate(assessment["gaps"], 1)],
    }
    joined["conflict_records"] = build_conflict_records(assessments, claims)
    errors = conflict_record_errors(joined["conflict_records"], claims, assessments)
    if errors:
        raise WriterArtifactMismatch("Conflict provenance validation failed")
    return joined, chunks, sources


def _node_error(exc):
    if isinstance(exc, APIError):
        code = exc.code
    elif isinstance(exc, BudgetExceeded):
        code = "budget_exceeded"
    elif isinstance(exc, InputBudgetExceeded):
        code = "input_budget_exceeded"
    elif isinstance(exc, WriterArtifactMismatch):
        code = "artifact_mismatch"
    else:
        code = "invalid_response"
    return NodeError(code=code, message=f"Writer failed: {type(exc).__name__}", retryable=False)


def write_report(request, structured):
    """Generator node contract: WriteRequest to a source-linked ReportResult."""
    request = WriteRequest.model_validate(request)
    try:
        joined, chunks, sources = _merge_results(request)
        previous = request.previous_report.report.model_dump(mode="json") if request.previous_report else None
        draft = draft_report(structured, joined, chunks, request.feedback, previous)
        gap_records = joined["gap_records"]
        if request.previous_report:
            decisions = [decision.model_dump(mode="json")
                         for decision in request.previous_report.report.gap_decisions]
        else:
            decisions = []
            for index in range(0, len(gap_records), 5):
                decisions.extend(review_gap_batch(structured, joined, sources, index // 5,
                                                  gap_records[index:index + 5]))
        report = Report.model_validate({**draft, "gap_decisions": decisions}).model_dump(mode="json")
        synthesis, synthesis_evidence = collect_evidence(
            {"synthesis": {"claims": report["synthesis_claims"]}}, chunks)
        joined = {**joined, "claims": {**joined["claims"], **synthesis},
                  "evidence": {**joined["evidence"], **synthesis_evidence}}
        errors = report_errors(report, joined["claims"], chunks, joined["gap_records"])
        if errors:
            raise ValueError("Report contract validation failed: " + "; ".join(errors))

        from rag.render import build_report_markdown
        markdown, context = build_report_markdown(report, joined, sources, request.context.model_dump(mode="json"))
        used_chunk_ids = {joined["evidence"][evidence_id]["chunk_id"]
                          for claim_id in context["used_claims"]
                          for evidence_id in joined["claims"][claim_id]["evidence_ids"]}
        chunk_by_id = {chunk["id"]: chunk for chunk in chunks}
        if used_chunk_ids - set(chunk_by_id):
            raise WriterArtifactMismatch("Report references a chunk outside its research results")
        used_sources = {source_id: sources[source_id] for source_id in context["used_sources"]}
        result = ReportResult.model_validate({
            **{key: getattr(request, key) for key in ("contract_version", "run_id", "request_id", "attempt", "context")},
            "status": "ok", "markdown": markdown, "report": report, "joined": joined,
            "chunks": [chunk_by_id[chunk_id] for chunk_id in sorted(used_chunk_ids)],
            "sources": used_sources, "error": None,
        })
        return result.model_dump(mode="json")
    except (APIError, BudgetExceeded, InputBudgetExceeded, ValueError, KeyError) as exc:
        failure = ReportResult.model_validate({
            **{key: getattr(request, key) for key in ("contract_version", "run_id", "request_id", "attempt", "context")},
            "status": "failed", "markdown": None, "report": None, "joined": None,
            "chunks": [], "sources": {}, "error": _node_error(exc),
        })
        return failure.model_dump(mode="json")
