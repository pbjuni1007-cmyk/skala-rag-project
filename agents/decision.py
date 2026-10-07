"""Supervisor model adapter using the existing guarded RAG Gateway."""
import json
from pathlib import Path

from agents.contracts import COMMON_FIELDS, SupervisorDecision

PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "supervisor.txt"


class GatewayDecider:
    def __init__(self, gateway):
        self.gateway = gateway

    def __call__(self, request):
        # The local Gateway retains input limits, cost reservation and call receipts.
        # request_id changes on every decision, including after a resumed run.
        response = self.gateway.generate(
            f"supervisor_{request['request_id']}",
            PROMPT.read_text(),
            json.dumps(request, ensure_ascii=False, allow_nan=False),
            SupervisorDecision.model_json_schema(),
        )
        decision = SupervisorDecision.model_validate_json(response)
        value = decision.model_dump(mode="json")
        for field in COMMON_FIELDS:
            if value[field] != request[field]:
                raise ValueError("Supervisor response identity changed")
        return decision
