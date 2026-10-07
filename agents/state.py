"""Control state is bounded; source text, complete results and logs stay in RunStore."""
from typing import TypedDict

from pydantic import Field
from rag.schemas import Strict
from agents.contracts import VIEWS


class Limits(Strict):
    worker_attempts: int = Field(default=2, ge=1, le=5)
    writer_attempts: int = Field(default=3, ge=1, le=5)
    supervisor_steps: int = Field(default=24, ge=1, le=100)


class AgentState(TypedDict):
    # Routing and recovery only. Sequential nodes replace their own latest entries;
    # they never write concurrently, so no reducer is needed.
    contract_version: str
    run_id: str
    context: dict
    identity: str
    status: str
    next_action: str
    step_count: int
    attempts: dict[str, int]
    pending: dict | None
    last_error: dict | None
    feedback: dict[str, list[str]]
    evidence_sufficient: bool
    # Latest bounded summaries and immutable file references, never raw documents
    # or accumulated trace messages. run_id links these to the external events.
    summaries: dict[str, dict]
    results: dict[str, dict]
    report: dict | None
    evaluation: dict | None
    publication: dict | None
    repair_source: dict | None
