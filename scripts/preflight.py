"""A real, billed connection smoke test. Never prints credentials."""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.settings import Settings
from rag.llm import Gateway
from rag.budget import write_json

if __name__ == "__main__":
    run = "preflight-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    gateway = Gateway(Settings.load(), run, Path("outputs") / run)
    result = gateway.lookup()
    result["response"] = gateway.generate("connection", "Reply with exactly OK.", "Connection test.", max_output=1024)
    result["reasoning_effort"] = gateway.settings.get("OPENAI_REASONING_EFFORT", "max")
    result["budget"] = gateway.budget.summary()
    write_json(Path("outputs") / run / "connection.json", result)
    print(json.dumps(result, ensure_ascii=False))
