from typing import Literal, TypedDict
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Query(Strict):
    technology: Literal["KIVI", "InfiniGen"]
    facet: Literal["mechanism", "limitation", "conditions", "maturity"]
    query: str = Field(min_length=1, max_length=400, pattern=r"^[\x20-\x7E]+$")


class Queries(Strict):
    queries: list[Query]


class RetrievalReviewItem(Strict):
    facet: Literal["mechanism", "limitation", "conditions", "maturity"]
    sufficient: bool
    chunk_ids: list[str]
    reason: str
    missing: list[str]


class RetrievalReview(Strict):
    items: list[RetrievalReviewItem]


class Reference(Strict):
    chunk_id: str
    quote: str


class Claim(Strict):
    technology: Literal["KIVI", "InfiniGen", "both"]
    facet: str
    text: str
    kind: Literal["source_fact", "author_reported_result", "team_inference", "scenario", "unknown"]
    references: list[Reference]
    caveats: str
    conditions: str


class Assessment(Strict):
    status: Literal["ok", "insufficient"]
    claims: list[Claim]
    conflicts: list[str]
    gaps: list[str]


class ConflictRecord(Strict):
    """Local review artifact; never added to an LLM response schema."""
    id: str
    perspective: str
    text: str
    candidate_claim_ids: list[str]
    explicit_claim_ids: list[str]
    verified_claim_ids: list[str]
    conditions: dict[str, str]
    evidence_ids: dict[str, list[str]]
    type: Literal["unclassified", "condition_difference", "contradiction", "tradeoff"]
    status: Literal["unresolved", "resolved"]
    rationale: str
    reviewed_by: str


class PerspectiveQuery(Strict):
    facet: str
    query: str = Field(min_length=1, max_length=400)


class PerspectiveQueries(Strict):
    queries: list[PerspectiveQuery]


class Section(Strict):
    title: str
    claim_ids: list[str]


class GapDecision(Strict):
    gap_id: str
    status: Literal["resolved", "unresolved"]
    resolution: str
    claim_ids: list[str]


class SynthesisClaim(Claim):
    kind: Literal["team_inference"]


class ReportDraft(Strict):
    summary_claim_ids: list[str]
    sections: list[Section]
    synthesis_claims: list[SynthesisClaim] = Field(min_length=2, max_length=2)


class GapDecisions(Strict):
    gap_decisions: list[GapDecision]


class Report(ReportDraft):
    gap_decisions: list[GapDecision]


class RunContext(Strict):
    technologies: list[Literal["KIVI", "InfiniGen"]] = Field(min_length=2, max_length=2)
    domain: str = Field(min_length=1)
    scenario: str = Field(min_length=1)

    @model_validator(mode="after")
    def has_both_technologies(self):
        if set(self.technologies) != {"KIVI", "InfiniGen"}:
            raise ValueError("RunContext must contain KIVI and InfiniGen exactly once")
        return self


class ContractEnvelope(Strict):
    contract_version: Literal["agent-contract-v1"]
    run_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    context: RunContext


NodeErrorCode = Literal["retrieval_error", "invalid_response", "api_error", "budget_exceeded",
                        "input_budget_exceeded", "uncertain_request", "artifact_mismatch", "render_error"]


class NodeError(Strict):
    code: NodeErrorCode
    message: str = Field(min_length=1)
    retryable: bool = False


class ResearchResult(ContractEnvelope):
    view: Literal["research", "market", "stakeholder", "domain"]
    status: Literal["ok", "insufficient", "failed"]
    assessments: dict[str, dict]
    chunks: list[dict]
    sources: dict[str, dict]
    error: NodeError | None

    @model_validator(mode="after")
    def result_matches_view_and_status(self):
        expected = {"research_kivi", "research_infinigen"} if self.view == "research" else {self.view}
        if self.status == "failed":
            if self.assessments or self.chunks or self.sources or self.error is None:
                raise ValueError("Failed ResearchResult requires empty payloads and a NodeError")
            return self
        if set(self.assessments) != expected or self.error is not None:
            raise ValueError("Successful ResearchResult requires the expected assessments and no error")
        statuses = [assessment.get("status") for assessment in self.assessments.values()]
        if any(status not in {"ok", "insufficient"} for status in statuses):
            raise ValueError("Successful ResearchResult assessments must be ok or insufficient")
        if self.status == "ok" and any(status != "ok" for status in statuses):
            raise ValueError("ResearchResult status ok requires all assessments to be ok")
        if self.status == "insufficient" and not any(status == "insufficient" for status in statuses):
            raise ValueError("ResearchResult status insufficient requires an insufficient assessment")
        if self.status == "insufficient" and any(status not in {"ok", "insufficient"} for status in statuses):
            raise ValueError("Insufficient ResearchResult cannot include failed assessments")
        if any(chunk.get("source_id") not in self.sources for chunk in self.chunks):
            raise ValueError("ResearchResult sources must include each chunk source")
        return self


class ReportResult(ContractEnvelope):
    status: Literal["ok", "failed"]
    markdown: str | None
    report: Report | None
    joined: dict | None
    chunks: list[dict]
    sources: dict[str, dict]
    error: NodeError | None

    @model_validator(mode="after")
    def result_fields_match_status(self):
        if self.status == "ok":
            if (any(value is None for value in (self.markdown, self.report, self.joined))
                    or not self.markdown.strip() or self.error is not None):
                raise ValueError("Successful ReportResult requires Markdown, report, joined data, and no error")
        elif (any(value is not None for value in (self.markdown, self.report, self.joined))
              or self.chunks or self.sources or self.error is None):
            raise ValueError("Failed ReportResult requires null content and a NodeError")
        return self


class WriteRequest(ContractEnvelope):
    results: dict[str, ResearchResult]
    feedback: list[str]
    previous_report: ReportResult | None

    @model_validator(mode="after")
    def has_all_perspectives(self):
        expected = {"research", "market", "stakeholder", "domain"}
        if set(self.results) != expected:
            raise ValueError("WriteRequest requires all four perspective results")
        if any(result.status != "ok" for result in self.results.values()):
            raise ValueError("WriteRequest perspective results must all be ok")
        if any(result.view != view for view, result in self.results.items()):
            raise ValueError("Each ResearchResult view must match its WriteRequest key")
        if any(result.run_id != self.run_id or result.context != self.context
               for result in self.results.values()):
            raise ValueError("WriteRequest perspective results must share its run and context")
        if self.previous_report and (self.previous_report.run_id != self.run_id
                                     or self.previous_report.context != self.context):
            raise ValueError("Previous report must share the WriteRequest run and context")
        return self


class EvaluationRequest(ContractEnvelope):
    report_result: ReportResult

    @model_validator(mode="after")
    def report_matches_run_context(self):
        if self.report_result.status != "ok":
            raise ValueError("EvaluationRequest requires a successful ReportResult")
        if self.report_result.run_id != self.run_id or self.report_result.context != self.context:
            raise ValueError("EvaluationRequest and ReportResult must share run and context")
        return self


QualityCriterion = Literal["groundedness", "neutrality", "bias_control", "coverage"]
PerspectiveLabel = Literal["기술 성숙도", "시장성", "이해관계자", "도메인 적용"]


class QualityFinding(Strict):
    criterion: QualityCriterion
    target: str = Field(min_length=1)
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    issue: str = Field(min_length=1)
    revision_request: str = Field(min_length=1)
    missing_perspectives: list[PerspectiveLabel] = Field(default_factory=list)


class QualityCriterionResult(Strict):
    criterion: QualityCriterion
    status: Literal["pass", "revise"]
    rationale: str = Field(min_length=1)
    findings: list[QualityFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def revision_has_findings(self):
        if self.status == "revise" and not self.findings:
            raise ValueError("A revise verdict requires at least one finding")
        return self


class QualityJudgeDraft(Strict):
    criteria: list[QualityCriterionResult]

    @model_validator(mode="after")
    def has_each_quality_criterion_once(self):
        expected = {"groundedness", "neutrality", "bias_control", "coverage"}
        actual = [item.criterion for item in self.criteria]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Judge must return each of the four quality criteria exactly once")
        return self


class QualityCheck(Strict):
    passed: bool
    reason: str = Field(min_length=1)
    claim_ids: list[str]
    gap_ids: list[str]


class RepairRequest(Strict):
    target: Literal["research", "market", "stakeholder", "domain", "writer"]
    reason: str = Field(min_length=1)
    claim_ids: list[str]
    gap_ids: list[str]


class EvaluationResult(ContractEnvelope):
    report_request_id: str = Field(min_length=1)
    status: Literal["ok", "failed"]
    method: Literal["structural", "llm", "hybrid"]
    passed: bool
    checks: dict[QualityCriterion, QualityCheck]
    repair_requests: list[RepairRequest]
    error: NodeError | None

    @model_validator(mode="after")
    def evaluation_fields_match_status(self):
        expected = {"groundedness", "neutrality", "bias_control", "coverage"}
        if self.status == "failed":
            if self.passed or self.checks or self.repair_requests or self.error is None:
                raise ValueError("Failed evaluation must not contain checks or repair requests")
            return self
        if self.error is not None or set(self.checks) != expected:
            raise ValueError("Successful evaluation requires all four checks and no error")
        all_passed = all(check.passed for check in self.checks.values())
        if self.passed != all_passed:
            raise ValueError("Evaluation passed must match all criterion checks")
        if self.passed == bool(self.repair_requests):
            raise ValueError("Passing evaluation has no repair requests; a failing evaluation has at least one")
        return self


class State(TypedDict, total=False):
    run_config: dict
    source_registry: dict
    research_queries: list
    research_hits: dict
    research_attempts: int
    tech_assessment: dict
    market_retrieval: dict
    stakeholder_retrieval: dict
    domain_retrieval: dict
    research_retrieval: dict
    market_result: dict
    stakeholder_result: dict
    domain_result: dict
    joined: dict
    report: dict
    report_result: dict
    markdown: str
    evaluation_result: dict
    report_revision_requests: list
    validation_result: dict
    report_attempts: int
    evaluation_attempts: int
    output_paths: dict
    run_status: str
