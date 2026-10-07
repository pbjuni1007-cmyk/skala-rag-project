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


class TechnologyQueries(Strict):
    queries: list[Query] = Field(min_length=4, max_length=4)


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
