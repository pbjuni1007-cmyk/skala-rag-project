"""JSON boundaries published in docs/agent-contract.md."""
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from rag.schemas import Assessment, ConflictRecord, Report, Strict

Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
View = Literal["research", "market", "stakeholder", "domain"]
Action = Literal["research", "market", "stakeholder", "domain", "writer", "evaluator", "publish", "stop"]
Criterion = Literal["groundedness", "neutrality", "bias_control", "coverage"]
ReasonCode = Literal[
    "initial_research", "missing_view", "evidence_gap", "evidence_ready", "report_ready",
    "quality_rework", "quality_passed", "limit_exceeded", "fatal_error",
]
VIEWS = ("research", "market", "stakeholder", "domain")
CRITERIA = ("groundedness", "neutrality", "bias_control", "coverage")
COMMON_FIELDS = ("contract_version", "run_id", "request_id", "attempt", "context")


class RunContext(Strict):
    technologies: list[Literal["KIVI", "InfiniGen"]]
    domain: Text
    scenario: Text

    @model_validator(mode="after")
    def technology_pair(self):
        if self.technologies != ["KIVI", "InfiniGen"]:
            raise ValueError("Use the agreed KIVI/InfiniGen technology pair")
        return self


class Message(Strict):
    contract_version: Literal["agent-contract-v1"]
    run_id: Text
    request_id: Text
    attempt: Annotated[int, Field(strict=True, ge=1)]
    context: RunContext


class NodeError(Strict):
    code: Literal[
        "retrieval_error", "invalid_response", "api_error", "budget_exceeded",
        "input_budget_exceeded", "uncertain_request", "artifact_mismatch", "render_error",
    ]
    message: Text
    retryable: bool


class ArtifactRef(Strict):
    path: Text
    sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class Chunk(Strict):
    model_config = ConfigDict(extra="allow")
    id: Text
    source_id: Text
    technology: Literal["KIVI", "InfiniGen", "both"]
    text: Text
    page: Annotated[int, Field(strict=True, ge=1)] | None
    section: Text | None

    @model_validator(mode="after")
    def location(self):
        if (self.page is None) == (self.section is None):
            raise ValueError("Provide a PDF page or a web section, exclusively")
        return self


class Source(Strict):
    model_config = ConfigDict(extra="allow")
    type: Literal["paper_pool", "external_web"]
    authors: Text
    title: Text
    url: Text
    version: Text
    date: str | None
    accessed_at: Text


def evidence_registry(chunks, sources):
    """Reject conflicting IDs before dictionaries can silently overwrite them."""
    known = {}
    for chunk in chunks:
        value = chunk.model_dump(mode="json")
        if chunk.id in known and known[chunk.id] != value:
            raise ValueError("Conflicting chunk ID")
        known[chunk.id] = value
        if chunk.source_id not in sources:
            raise ValueError("Unknown source ID")
        source = sources[chunk.source_id]
        if (source.type == "paper_pool") != (chunk.page is not None):
            raise ValueError("Source type and chunk location disagree")
    return list(known.values())


class ResearchResult(Message):
    view: View
    status: Literal["ok", "insufficient", "failed"]
    assessments: dict[str, Assessment]
    chunks: list[Chunk]
    sources: dict[str, Source]
    error: NodeError | None

    @model_validator(mode="after")
    def consistent_result(self):
        if self.status == "failed":
            if self.error is None or self.assessments or self.chunks or self.sources:
                raise ValueError("Failed research requires an error and empty payload")
            return self
        if self.error is not None:
            raise ValueError("Successful research cannot carry an execution error")
        keys = {"research_kivi", "research_infinigen"} if self.view == "research" else {self.view}
        if set(self.assessments) != keys:
            raise ValueError("Wrong assessment keys for this perspective")
        insufficient = [a for a in self.assessments.values() if a.status == "insufficient"]
        if (self.status == "insufficient") != bool(insufficient):
            raise ValueError("Research and assessment statuses disagree")
        if any(not any(g.strip() for g in a.gaps) for a in insufficient):
            raise ValueError("Insufficient assessments must explain their gaps")
        from rag.evidence import validate_assessment, validate_perspective

        chunks = evidence_registry(self.chunks, self.sources)
        errors = []
        for name, assessment in self.assessments.items():
            value = assessment.model_dump(mode="json")
            if self.view == "research":
                technology = "KIVI" if name == "research_kivi" else "InfiniGen"
                errors += validate_assessment(value, chunks, technology, core=self.status == "ok")
            elif self.status == "ok":
                errors += validate_perspective(value, chunks, self.view)
            else:
                errors += validate_assessment(value, chunks)
        if errors:
            raise ValueError("Invalid research evidence: " + "; ".join(errors))
        return self


def same_run(message, nested):
    if (message.run_id, message.context) != (nested.run_id, nested.context):
        raise ValueError("Run/context mismatch")


class ResearchRequest(Message):
    view: View
    questions: Annotated[list[Text], Field(min_length=1)]
    feedback: list[Text]
    previous_result: ResearchResult | None
    research_context: ResearchResult | None

    @model_validator(mode="after")
    def prerequisites(self):
        if self.previous_result is not None:
            same_run(self, self.previous_result)
            if self.previous_result.view != self.view or self.previous_result.attempt >= self.attempt:
                raise ValueError("Previous research must be an earlier result of this view")
        if self.view == "research":
            if self.research_context is not None:
                raise ValueError("Research has no upstream research context")
        elif self.research_context is None:
            raise ValueError("Perspective research needs technical evidence")
        else:
            same_run(self, self.research_context)
            if self.research_context.view != "research" or self.research_context.status != "ok":
                raise ValueError("Perspective research needs successful technical evidence")
        return self


class Joined(Strict):
    claims: dict[str, dict]
    evidence: dict[str, dict]
    assessments: dict[str, Assessment]
    conflicts: list[str]
    gaps: list[str]
    gap_records: list[dict]
    conflict_records: list[ConflictRecord]


class ReportResult(Message):
    status: Literal["ok", "failed"]
    markdown: Text | None
    report: Report | None
    joined: Joined | None
    chunks: list[Chunk]
    sources: dict[str, Source]
    error: NodeError | None

    @model_validator(mode="after")
    def consistent_result(self):
        if self.status == "failed":
            if self.error is None or any(x is not None for x in (self.markdown, self.report, self.joined)):
                raise ValueError("Failed report requires an error and null document")
            if self.chunks or self.sources:
                raise ValueError("Failed report cannot publish partial sources")
            return self
        if self.error is not None or any(x is None for x in (self.markdown, self.report, self.joined)):
            raise ValueError("Successful report requires all document forms")
        from rag.evidence import validate_assessment
        from rag.schemas import Claim

        chunks = evidence_registry(self.chunks, self.sources)
        for claim_id, raw in self.joined.claims.items():
            if raw.get("id") != claim_id:
                raise ValueError("Claim ID mismatch")
            claim = Claim.model_validate({k: raw[k] for k in Claim.model_fields})
            if validate_assessment({"claims": [claim.model_dump()]}, chunks):
                raise ValueError("Report claim has invalid evidence")
            for evidence_id in raw.get("evidence_ids", []):
                evidence = self.joined.evidence.get(evidence_id)
                if evidence is None or evidence.get("id") != evidence_id:
                    raise ValueError("Unknown report evidence ID")
        known = {c["id"]: c for c in chunks}
        for evidence in self.joined.evidence.values():
            chunk_id = evidence.get("chunk_id")
            chunk = known.get(chunk_id) if isinstance(chunk_id, str) else None
            if chunk is None or chunk["source_id"] != evidence.get("source_id"):
                raise ValueError("Report evidence cannot resolve its source")
        return self


class WriteRequest(Message):
    results: dict[View, ResearchResult]
    feedback: list[Text]
    previous_report: ReportResult | None

    @model_validator(mode="after")
    def prerequisites(self):
        if set(self.results) != set(VIEWS):
            raise ValueError("Writing needs all four views")
        for view, result in self.results.items():
            same_run(self, result)
            if result.view != view or result.status != "ok":
                raise ValueError("Writing needs successful current research")
        if self.previous_report is not None:
            same_run(self, self.previous_report)
            if self.previous_report.attempt >= self.attempt:
                raise ValueError("Previous report must have an earlier writer attempt")
        return self


class Check(Strict):
    passed: bool
    reason: Text
    claim_ids: list[Text]
    gap_ids: list[Text]


class RepairRequest(Strict):
    target: Literal["research", "market", "stakeholder", "domain", "writer"]
    reason: Text
    claim_ids: list[Text]
    gap_ids: list[Text]


class EvaluationRequest(Message):
    report_result: ReportResult

    @model_validator(mode="after")
    def prerequisites(self):
        same_run(self, self.report_result)
        if self.report_result.status != "ok":
            raise ValueError("Evaluation requires a generated report")
        return self


class EvaluationResult(Message):
    report_request_id: Text
    status: Literal["ok", "failed"]
    method: Literal["structural", "llm", "hybrid"]
    passed: bool
    checks: dict[Criterion, Check]
    repair_requests: list[RepairRequest]
    error: NodeError | None

    @model_validator(mode="after")
    def consistent_result(self):
        if self.status == "failed":
            if self.passed or self.checks or self.repair_requests or self.error is None:
                raise ValueError("Evaluation error is never a quality verdict")
        else:
            if self.error is not None or set(self.checks) != set(CRITERIA):
                raise ValueError("Evaluation needs exactly four criteria")
            if self.passed != all(c.passed for c in self.checks.values()):
                raise ValueError("Overall verdict disagrees with its criteria")
            if bool(self.repair_requests) == self.passed:
                raise ValueError("Failed quality needs repairs; passed quality has none")
        return self


class SupervisorDecision(Message):
    next_action: Action
    evidence_sufficient: bool
    reason_code: ReasonCode
    reason: Text
    feedback: list[Text]


class PublishRequest(Message):
    report_result: ReportResult
    evaluation_result: EvaluationResult

    @model_validator(mode="after")
    def prerequisites(self):
        for result in (self.report_result, self.evaluation_result):
            same_run(self, result)
            if result.status != "ok":
                raise ValueError("Publication requires successful report and evaluation")
        if (not self.evaluation_result.passed or
                self.evaluation_result.report_request_id != self.report_result.request_id):
            raise ValueError("Publication requires this report's passing verdict")
        return self


class PublishResult(Message):
    status: Literal["ok", "failed"]
    markdown_ref: ArtifactRef | None
    pdf_ref: ArtifactRef | None
    pdf_pages: Annotated[int, Field(strict=True, ge=1, le=10)] | None
    human_review_pending: bool
    error: NodeError | None

    @model_validator(mode="after")
    def consistent_result(self):
        outputs = (self.markdown_ref, self.pdf_ref, self.pdf_pages)
        if self.status == "failed":
            if any(x is not None for x in outputs) or self.error is None:
                raise ValueError("Failed publication cannot report completed files")
        elif any(x is None for x in outputs) or not self.human_review_pending or self.error is not None:
            raise ValueError("Publication needs checked files and pending human review")
        return self
