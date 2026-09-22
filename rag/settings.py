from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import math
import os
import re
import shlex


def read_env(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.fullmatch(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*", line)
            if not match:
                raise ValueError(f"Invalid env syntax on line {number}; values hidden")
            parts = shlex.split(match[2], comments=True)
            if len(parts) > 1:
                raise ValueError(f"Quote env values with spaces on line {number}")
            values[match[1]] = parts[0] if parts else ""
    return values


@dataclass
class Settings:
    values: dict[str, str] = field(repr=False)

    @classmethod
    def load(cls, path: str = ".env.local"):
        return cls({**read_env(Path(path)), **os.environ})

    def get(self, key, default=""):
        return self.values.get(key) or default

    def integer(self, key, default):
        value = int(self.get(key, str(default)))
        if value < 1:
            raise ValueError(f"{key} must be positive")
        return value

    def number(self, key, default=None):
        value = float(self.get(key, str(default) if default is not None else ""))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number")
        return value

    def validate_paid(self):
        if not self.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing")
        if self.get("LLM_REASONING_PROFILE", "balanced") not in {"balanced", "fixed"}:
            raise ValueError("Unsupported reasoning profile")
        self.integer("RAG_MAX_CONCURRENCY", 3)
        # Pricing is verified for this exact model/tier only. No silent fallback.
        if self.get("OPENAI_MODEL") != "gpt-5.6-luna":
            raise ValueError("Model change requires an updated, reviewed pricing contract")
        if self.get("OPENAI_REASONING_EFFORT", "max") not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported reasoning effort")
        if self.number("OPENAI_INPUT_USD_PER_MILLION_TOKENS") != 0.20 or self.number("OPENAI_OUTPUT_USD_PER_MILLION_TOKENS") != 1.20:
            raise ValueError("Pricing differs from the reviewed Standard price contract")
        age = (date.today() - date.fromisoformat(self.get("PRICING_CHECKED_AT"))).days
        if not 0 <= age <= 30:
            raise ValueError("Verify official pricing again (date missing, future or older than 30 days)")
        if not self.number("PROJECT_SPEND_LIMIT_KRW", 45000) <= self.number("PROJECT_BUDGET_KRW", 50000) <= 50000:
            raise ValueError("Project spending must stay within the approved 50,000 KRW")
        self.number("USD_TO_KRW")
        if self.number("BUDGET_COST_MULTIPLIER", 1.10) < 1.10:
            raise ValueError("Keep the approved 10% exchange/fee buffer")
        if self.integer("LLM_MAX_INPUT_TOKENS", 24000) > 24000:
            raise ValueError("This pricing contract limits input to 24,000 tokens")
        if self.integer("LLM_MAX_OUTPUT_TOKENS", 8000) > 128000:
            raise ValueError("Model output limit exceeded")
        if not 0 <= int(self.get("OPENAI_MAX_RETRIES", "2")) <= 2:
            raise ValueError("At most two API retries are permitted")

    def reasoning_effort(self, purpose):
        """Route known roles; unknown purposes retain the configured effort."""
        fallback = self.get("OPENAI_REASONING_EFFORT", "max")
        if self.get("LLM_REASONING_PROFILE", "balanced") == "fixed":
            return fallback
        parts = purpose.lower().split("_")
        if any(part in {"repair", "query", "queries", "rewrite"} for part in parts) or purpose.startswith("retrieval_review"):
            return "low"
        if purpose == "synthesis_report" or purpose.startswith("synthesis_gaps_"):
            return "max"
        if parts[0] in {"research", "market", "stakeholder", "domain", "perspective", "facet"}:
            return "medium"
        return fallback

    def public(self):
        keys = ("OPENAI_MODEL", "OPENAI_REASONING_EFFORT", "PRICING_CHECKED_AT", "USD_TO_KRW",
                "OPENAI_INPUT_USD_PER_MILLION_TOKENS", "OPENAI_OUTPUT_USD_PER_MILLION_TOKENS",
                "PROJECT_BUDGET_KRW", "PROJECT_SPEND_LIMIT_KRW", "BUDGET_COST_MULTIPLIER",
                "LLM_MAX_INPUT_TOKENS", "LLM_MAX_OUTPUT_TOKENS", "EMBEDDING_MODEL", "EMBEDDING_REVISION")
        return {**{key: self.get(key) for key in keys},
                "LLM_REASONING_PROFILE": self.get("LLM_REASONING_PROFILE", "balanced"),
                "RAG_MAX_CONCURRENCY": str(self.integer("RAG_MAX_CONCURRENCY", 3)),
                "effective_reasoning_efforts": {purpose: self.reasoning_effort(purpose) for purpose in
                    ("research_queries", "rewrite", "retrieval_review", "research", "market", "stakeholder",
                     "domain", "market_reassessment_facet_costs", "synthesis_report_repair",
                     "synthesis_report", "synthesis_gaps_0", "unknown")}}
