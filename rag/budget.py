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
            if getattr(self, "expected_ledger_id", None) and data.get("ledger_id") != self.expected_ledger_id:
                raise ValueError("Budget ledger was removed or replaced; inspection required")
            if data.get("schema") != 1 or not isinstance(data.get("entries"), list):
                raise ValueError("Invalid budget ledger; never reset it automatically")
            for entry in data["entries"]:
                for key in ("reserved_krw", "charged_estimate_krw"):
                    if key in entry and (not isinstance(entry[key], (int, float)) or not math.isfinite(entry[key]) or entry[key] < 0):
                        raise ValueError("Invalid ledger amount; inspection required")
            history_path = getattr(self, "history_path", None)
            if history_path is not None and history_path.exists():
                self._validate_history(data, json.loads(history_path.read_text()))
            try:
                yield data
                write_json(self.path, data)
                # Persist after the ledger, but before a reserved call can be sent.
                # A crash between writes can leave a shorter history, never an
                # anchor for a request that was not reserved in the ledger.
                if history_path is not None:
                    write_json(history_path, data)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def bind_identity(self, expected=None):
        """Give an existing cumulative ledger a stable identity without resetting it."""
        path = str(self.path.resolve())
        if expected is not None:
            if expected.get("path") != path or not expected.get("ledger_id"):
                raise ValueError("Resume rejected: budget ledger identity changed")
            self.expected_ledger_id = expected["ledger_id"]
        with self.locked() as data:
            ledger_id = data.setdefault("ledger_id", uuid.uuid4().hex)
            if not isinstance(ledger_id, str) or not ledger_id:
                raise ValueError("Invalid budget ledger identity")
        self.expected_ledger_id = ledger_id
        return {"path": path, "ledger_id": ledger_id}

    @staticmethod
    def _validate_history(current, previous):
        by_id = {entry["id"]: entry for entry in current["entries"]}
        if (current.get("ledger_id") != previous.get("ledger_id") or
                len(by_id) != len(current["entries"])):
            raise ValueError("Budget ledger history identity changed")
        mutable = {"state", "usage", "response_id", "charged_estimate_krw"}
        for entry in previous["entries"]:
            value = by_id.get(entry["id"])
            if value is None or any(value.get(key) != item for key, item in entry.items() if key not in mutable):
                raise ValueError("Budget ledger history was removed or changed")
            if entry["state"] != "reserved" and value != entry:
                raise ValueError("Budget ledger settled history changed")

    def bind_history(self, path, *, resume=False):
        self.history_path = Path(path)
        if resume and not self.history_path.is_file():
            raise ValueError("Resume rejected: budget ledger history is missing")
        with self.locked():
            pass

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
