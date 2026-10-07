"""Models for the agent-contract-v1 research boundary."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator

from rag.schemas import Assessment

CONTRACT_VERSION = "agent-contract-v1"
VIEWS = ("research", "market", "stakeholder", "domain")
TECHNOLOGIES = ("KIVI", "InfiniGen")
NODE_ERROR_CODES = (
    "retrieval_error",
    "invalid_response",
    "api_error",
    "budget_exceeded",
    "input_budget_exceeded",
    "uncertain_request",
    "artifact_mismatch",
    "render_error",
)

View = Literal["research", "market", "stakeholder", "domain"]
Technology = Literal["KIVI", "InfiniGen", "both"]
NodeErrorCode = Literal[
    "retrieval_error",
    "invalid_response",
    "api_error",
    "budget_exceeded",
    "input_budget_exceeded",
    "uncertain_request",
    "artifact_mismatch",
    "render_error",
]


def required_assessment_keys(view: str) -> set[str]:
    if view == "research":
        return {"research_kivi", "research_infinigen"}
    if view in {"market", "stakeholder", "domain"}:
        return {view}
    raise ValueError(f"Unknown research view: {view}")


class RunContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technologies: list[Literal["KIVI", "InfiniGen"]]
    domain: StrictStr
    scenario: StrictStr

    @field_validator("technologies")
    @classmethod
    def exact_technologies(cls, value):
        if value != ["KIVI", "InfiniGen"]:
            raise ValueError('technologies must be exactly ["KIVI", "InfiniGen"]')
        return value

    @field_validator("domain", "scenario")
    @classmethod
    def non_blank_context(cls, value):
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["agent-contract-v1"]
    run_id: StrictStr
    request_id: StrictStr
    attempt: StrictInt = Field(ge=1)
    context: RunContext

    @field_validator("run_id", "request_id")
    @classmethod
    def non_blank_ids(cls, value):
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class Chunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: StrictStr
    source_id: StrictStr
    technology: Technology
    text: StrictStr
    page: StrictInt | None
    section: StrictStr | None


class Source(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["paper_pool", "external_web"]
    authors: StrictStr
    title: StrictStr
    url: StrictStr
    version: StrictStr
    date: StrictStr | None
    accessed_at: StrictStr


class NodeError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: NodeErrorCode
    message: StrictStr
    retryable: bool = False


class ResearchResult(Envelope):
    view: View
    status: Literal["ok", "insufficient", "failed"]
    assessments: dict[str, Assessment]
    chunks: list[Chunk]
    sources: dict[str, Source]
    error: NodeError | None

    @model_validator(mode="after")
    def validate_status_shape(self):
        if self.status == "failed":
            if self.assessments or self.chunks or self.sources or self.error is None:
                raise ValueError("failed results require empty assessments, chunks and sources plus an error")
            return self

        if set(self.assessments) != required_assessment_keys(self.view):
            raise ValueError("assessment keys do not match the result view")
        if self.error is not None:
            raise ValueError(f"{self.status} results must not contain an error")

        statuses = [assessment.status for assessment in self.assessments.values()]
        if self.status == "ok" and any(status != "ok" for status in statuses):
            raise ValueError("ok results require every assessment to be ok")
        if self.status == "insufficient":
            insufficient = [a for a in self.assessments.values() if a.status == "insufficient"]
            if not insufficient:
                raise ValueError("insufficient results require an insufficient assessment")
            if any(not assessment.gaps or not any(gap.strip() for gap in assessment.gaps) for assessment in insufficient):
                raise ValueError("insufficient assessments require gap reasons")
        return self


class ResearchRequest(Envelope):
    view: View
    questions: list[StrictStr] = Field(min_length=1)
    feedback: list[StrictStr]
    previous_result: ResearchResult | None
    research_context: ResearchResult | None

    @field_validator("questions")
    @classmethod
    def non_blank_questions(cls, value):
        if any(not question.strip() for question in value):
            raise ValueError("questions must contain non-blank strings")
        return value

    @model_validator(mode="after")
    def validate_nested_results(self):
        previous = self.previous_result
        research = self.research_context

        if self.view == "research":
            if research is not None:
                raise ValueError("research requests require research_context=null")
        else:
            if research is None or research.view != "research" or research.status != "ok":
                raise ValueError("non-research requests require an ok research result")

        nested = [result for result in (previous, research) if result is not None]
        for result in nested:
            if result.run_id != self.run_id:
                raise ValueError("nested results must share run_id")
            if result.context != self.context:
                raise ValueError("nested results must share context")
            if result.contract_version != self.contract_version:
                raise ValueError("nested results must share contract_version")

        if previous is not None:
            if previous.view != self.view:
                raise ValueError("previous_result must have the same view")
            if previous.request_id == self.request_id:
                raise ValueError("previous_result must have a different request_id")
            if previous.attempt >= self.attempt:
                raise ValueError("previous_result must have a lower attempt")

        if research is not None and research.request_id == self.request_id:
            raise ValueError("research_context must have a different request_id")
        return self


def parse_request(value: dict | BaseModel) -> ResearchRequest:
    """Validate a request payload and normalize Pydantic errors to ValueError."""
    payload = value.model_dump(mode="python") if isinstance(value, BaseModel) else value
    try:
        return ResearchRequest.model_validate(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
