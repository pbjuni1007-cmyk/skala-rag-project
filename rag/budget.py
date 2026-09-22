"""Cross-process reservations; uncertain requests retain their full reservation."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import math
import os
import uuid


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, settings):
        self.settings = settings
        self.path = Path(settings.get("BUDGET_LEDGER_PATH", ".local/usage.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {"schema": 1, "entries": []}
            if data.get("schema") != 1 or not isinstance(data.get("entries"), list):
                raise ValueError("Invalid budget ledger; never reset it automatically")
            for entry in data["entries"]:
                for key in ("reserved_krw", "charged_estimate_krw"):
                    if key in entry and (not isinstance(entry[key], (int, float)) or not math.isfinite(entry[key]) or entry[key] < 0):
                        raise ValueError("Invalid ledger amount; inspection required")
            try:
                yield data
                write_json(self.path, data)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def cost(self, input_tokens, output_tokens):
        # Charge every input token at the higher cache-write price ($0.25).
        # This ignores read discounts and conservatively covers cache creation.
        usd = (input_tokens * 0.25 + output_tokens * 1.20) / 1_000_000
        return math.ceil(usd * self.settings.number("USD_TO_KRW") * self.settings.number("BUDGET_COST_MULTIPLIER", 1.10) * 1e6) / 1e6

    @staticmethod
    def total(data):
        return sum(entry.get("charged_estimate_krw", entry["reserved_krw"]) for entry in data["entries"])

    def reserve(self, run_id, purpose, input_limit, output_limit):
        amount = self.cost(input_limit, output_limit)
        with self.locked() as data:
            if any(e["state"] == "limit_violation" for e in data["entries"]):
                raise BudgetExceeded("A provider usage violation requires ledger review before more calls")
            if sum(e["run_id"] == run_id for e in data["entries"]) >= self.settings.integer("RAG_MAX_API_CALLS_PER_RUN", 40):
                raise BudgetExceeded("API call limit reached, including failed attempts")
            if self.total(data) + amount > self.settings.number("PROJECT_SPEND_LIMIT_KRW", 45000):
                raise BudgetExceeded("Project budget cannot cover this request's maximum cost")
            entry = {"id": uuid.uuid4().hex, "run_id": run_id, "purpose": purpose,
                     "created_at": datetime.now(timezone.utc).isoformat(), "state": "reserved",
                     "reserved_krw": amount, "input_limit": input_limit, "output_limit": output_limit,
                     "pricing": self.settings.public()}
            data["entries"].append(entry)
        return entry["id"]

    def settle(self, reservation, response):
        usage = response.get("usage")
        with self.locked() as data:
            entry = next(e for e in data["entries"] if e["id"] == reservation)
            if response.get("service_tier", "default") != "default" or not usage:
                entry["state"] = "uncertain"
                raise RuntimeError("Missing usage or unexpected processing tier; reservation retained")
            count_in, count_out = usage["input_tokens"], usage["output_tokens"]
            if not all(isinstance(n, int) and n >= 0 for n in (count_in, count_out)):
                raise RuntimeError("Invalid usage; reservation retained")
            entry.update(state="settled", usage=usage, response_id=response.get("id"),
                         charged_estimate_krw=self.cost(count_in, count_out))
            if count_in > entry["input_limit"] or count_out > entry["output_limit"]:
                # Persist actual cost, then stop future work for inspection.
                entry["state"] = "limit_violation"
        if entry["state"] == "limit_violation":
            raise RuntimeError("Provider usage exceeded reserved token limits")

    def summary(self):
        with self.locked() as data:
            return {"calls": len(data["entries"]), "conservative_total_krw": round(self.total(data), 4),
                    "unsettled_calls": sum(e["state"] != "settled" for e in data["entries"])}
