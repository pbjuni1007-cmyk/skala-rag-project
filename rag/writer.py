"""Report drafting from validated perspective claims and their source evidence."""

from copy import deepcopy

from rag.corpus import normalized
from rag.evidence import collect_evidence, report_errors, validate_assessment
from rag.payloads import synthesis_payload
from rag.schemas import ReportDraft


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


def write_report(structured, joined, chunks, revision_requests=None, previous_report=None):
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
