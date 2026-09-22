"""Exercise paid-call boundaries without credentials or network access."""
from datetime import date
from io import BytesIO
import json
from pathlib import Path
import urllib.error

import pytest

from rag.llm import APIError, Gateway
from rag.settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings({
        "OPENAI_API_KEY": "fake-test-key",
        "OPENAI_MODEL": "gpt-5.6-luna",
        "OPENAI_REASONING_EFFORT": "max",
        "PRICING_CHECKED_AT": date.today().isoformat(),
        "USD_TO_KRW": "1500",
        "OPENAI_INPUT_USD_PER_MILLION_TOKENS": "0.20",
        "OPENAI_OUTPUT_USD_PER_MILLION_TOKENS": "1.20",
        "LLM_MAX_INPUT_TOKENS": "24000",
        "LLM_MAX_OUTPUT_TOKENS": "4096",
        "OPENAI_MAX_RETRIES": "2",
        "OPENAI_TIMEOUT_SECONDS": "240",
        "BUDGET_LEDGER_PATH": str(tmp_path / "ledger.json"),
    })


@pytest.fixture(autouse=True)
def no_network_or_retry_wait(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unit tests must not contact the API")

    monkeypatch.setattr("rag.llm.urllib.request.urlopen", forbidden)
    monkeypatch.setattr("rag.llm.time.sleep", lambda _: None)


def completed(text="검증된 응답"):
    return {
        "id": "fake-response",
        "model": "gpt-5.6-luna",
        "status": "completed",
        "service_tier": "default",
        "usage": {"input_tokens": 100, "output_tokens": 200},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
    }


def entries(settings):
    path = Path(settings.get("BUDGET_LEDGER_PATH"))
    return json.loads(path.read_text())["entries"] if path.exists() else []


@pytest.mark.parametrize("status", [429, 500, None])
def test_retryable_failure_stops_after_two_retries_and_keeps_each_reservation(settings, tmp_path, monkeypatch, status):
    gateway = Gateway(settings, "retry-run", tmp_path / "out")
    calls, delays = [], []

    def fail(path, payload):
        if path == "responses/input_tokens":
            return {"input_tokens": 100}
        calls.append(path)
        if status is None:
            raise TimeoutError("sensitive transport detail")
        raise urllib.error.HTTPError("https://example.invalid", status, "sensitive body", {}, None)

    monkeypatch.setattr(gateway, "request", fail)
    monkeypatch.setattr("rag.llm.time.sleep", delays.append)
    with pytest.raises(APIError, match="reservation retained") as failure:
        gateway.generate("research", "Use supplied evidence", "short input")

    attempts = 3 if status == 429 else 1
    assert calls == ["responses"] * attempts
    assert len(delays) == attempts - 1
    assert len(entries(settings)) == attempts
    assert gateway.budget.summary()["unsettled_calls"] == attempts
    receipts = [json.loads(p.read_text()) for p in (tmp_path / 'out/calls').glob('*.json')]
    assert len(receipts) == attempts
    assert all(r['status'] == 'failed' and r['retryable'] == (status == 429) for r in receipts)
    assert all("charged_estimate_krw" not in entry for entry in entries(settings))
    assert "sensitive" not in str(failure.value)


@pytest.mark.parametrize("status", [400, 401])
def test_authentication_and_request_errors_are_not_retried(settings, tmp_path, monkeypatch, status):
    gateway = Gateway(settings, "auth-run", tmp_path / "out")
    calls = []

    def fail(path, payload):
        if path == "responses/input_tokens":
            return {"input_tokens": 100}
        calls.append(path)
        raise urllib.error.HTTPError("https://example.invalid", status, "hidden", {}, None)

    monkeypatch.setattr(gateway, "request", fail)
    with pytest.raises(APIError, match=f"HTTP {status}"):
        gateway.generate("research", "instructions", "input")
    assert calls == ["responses"]
    assert gateway.budget.summary()["unsettled_calls"] == 1


@pytest.mark.parametrize("retry_limit", ["-1", "3"])
def test_invalid_retry_limit_is_rejected_before_any_request(settings, tmp_path, retry_limit):
    settings.values["OPENAI_MAX_RETRIES"] = retry_limit
    with pytest.raises(ValueError, match="two API retries"):
        Gateway(settings, "invalid", tmp_path / "out")
    assert entries(settings) == []


def test_success_after_429_settles_only_the_successful_attempt(settings, tmp_path, monkeypatch):
    gateway = Gateway(settings, "recovered", tmp_path / "out")
    calls = []

    def request(path, payload):
        if path == "responses/input_tokens":
            return {"input_tokens": 100}
        calls.append(payload)
        if len(calls) == 1:
            raise urllib.error.HTTPError("https://example.invalid", 429, "hidden", {}, None)
        return completed()

    monkeypatch.setattr(gateway, "request", request)
    assert gateway.generate("research", "instructions", "input") == "검증된 응답"
    ledger = entries(settings)
    assert len(ledger) == 2
    assert "charged_estimate_krw" not in ledger[0]
    assert ledger[1]["state"] == "settled"
    expected = ledger[0]["reserved_krw"] + gateway.budget.cost(100, 200)
    assert gateway.budget.summary()["conservative_total_krw"] == round(expected, 4)
    assert calls[0] == calls[1]
    assert calls[0]["reasoning"]["effort"] == "medium"
    assert calls[0]["max_output_tokens"] == settings.integer("LLM_MAX_OUTPUT_TOKENS", 1)


def test_reasoning_only_incomplete_response_is_billed_and_cannot_be_reused(settings, tmp_path, monkeypatch):
    out = tmp_path / "incomplete"
    gateway = Gateway(settings, "incomplete-run", out)
    response = completed()
    response.update(status="incomplete", output=[{"type": "reasoning", "summary": []}],
                    incomplete_details={"reason": "max_output_tokens"})
    response["usage"]["output_tokens_details"] = {"reasoning_tokens": 200}
    monkeypatch.setattr(gateway, "request", lambda *args: {"input_tokens": 100} if args[0] == "responses/input_tokens" else response)
    with pytest.raises(APIError, match="generation incomplete"):
        gateway.generate("research", "instructions", "input")

    receipt = json.loads(next((out / "calls").glob("*.json")).read_text())
    assert receipt["status"] == "incomplete" and receipt["text"] == ""
    assert receipt["usage"]["output_tokens_details"]["reasoning_tokens"] == 200
    assert entries(settings)[0]["charged_estimate_krw"] == gateway.budget.cost(100, 200)
    resumed = Gateway(settings, "retry-incomplete", tmp_path / "resumed", cache_dir=out)
    assert receipt["request_hash"] not in resumed.cached


@pytest.mark.parametrize("failure", [TimeoutError(), urllib.error.URLError("hidden")])
def test_failed_exact_token_count_never_sends_generation_or_reserves(settings, tmp_path, monkeypatch, failure):
    settings.values["LLM_MAX_INPUT_TOKENS"] = "1024"
    gateway = Gateway(settings, "count-fail", tmp_path / "out")
    calls = []

    def request(path, payload):
        calls.append(path)
        raise failure

    monkeypatch.setattr(gateway, "request", request)
    with pytest.raises(APIError, match="no generation request sent"):
        gateway.generate("research", "instructions", "한글 자료 " * 300)
    assert calls == ["responses/input_tokens"]
    assert entries(settings) == []


@pytest.mark.parametrize("count", [None, "100", True, -1, 513])
def test_invalid_or_over_limit_token_count_never_sends_generation(settings, tmp_path, monkeypatch, count):
    settings.values["LLM_MAX_INPUT_TOKENS"] = "1024"
    gateway = Gateway(settings, "count-invalid", tmp_path / "out")
    calls = []

    def request(path, payload):
        calls.append(path)
        return {"input_tokens": count} if path == "responses/input_tokens" else completed()

    monkeypatch.setattr(gateway, "request", request)
    with pytest.raises((APIError, ValueError)):
        gateway.generate("research", "instructions", "한글 자료 " * 300)
    assert calls == ["responses/input_tokens"]
    assert entries(settings) == []


def test_exact_count_can_admit_korean_text_below_token_cap(settings, tmp_path, monkeypatch):
    settings.values["LLM_MAX_INPUT_TOKENS"] = "1024"
    gateway = Gateway(settings, "count-ok", tmp_path / "out")
    calls = []
    content = "한글 자료 " * 300

    def request(path, payload):
        calls.append((path, payload))
        return {"input_tokens": 512} if path == "responses/input_tokens" else completed()

    monkeypatch.setattr(gateway, "request", request)
    assert gateway.generate("research", "instructions", content) == "검증된 응답"
    assert [path for path, _ in calls] == ["responses/input_tokens", "responses"]
    for key in ("model", "instructions", "input", "reasoning"):
        assert calls[0][1][key] == calls[1][1][key]
    assert calls[1][1]["input"] == content


@pytest.mark.parametrize("change", [None, "content", "instructions", "schema", "reasoning", "output_cap"])
def test_resume_reuses_only_an_exact_request(settings, tmp_path, monkeypatch, change):
    original = tmp_path / "original"
    first = Gateway(settings, "original-run", original)
    monkeypatch.setattr(first, "request", lambda *args: {"input_tokens": 100} if args[0] == "responses/input_tokens" else completed("original answer"))
    first.generate("research", "instructions", "input")
    if change == "reasoning":
        settings.values["LLM_REASONING_PROFILE"] = "fixed"
        settings.values["OPENAI_REASONING_EFFORT"] = "high"
    elif change == "output_cap":
        settings.values["LLM_MAX_OUTPUT_TOKENS"] = "2048"
    resumed = Gateway(settings, "resumed-run", tmp_path / "resumed", cache_dir=original)
    calls = []

    def request(path, payload):
        if path == "responses/input_tokens":
            return {"input_tokens": 100}
        calls.append(path)
        return completed("new answer")

    monkeypatch.setattr(resumed, "request", request)
    answer = resumed.generate(
        "research", "changed instructions" if change == "instructions" else "instructions",
        "changed input" if change == "content" else "input",
        schema={"type": "object", "properties": {}} if change == "schema" else None,
    )
    if change is None:
        assert answer == "original answer" and calls == []
        assert len(entries(settings)) == 1
        receipt = json.loads((tmp_path / "resumed/calls/research-reused.json").read_text())
        assert receipt["reused_in_run"] == "resumed-run"
    else:
        assert answer == "new answer" and calls == ["responses"]
        assert len(entries(settings)) == 2


def test_tampered_receipt_is_not_reused(settings, tmp_path, monkeypatch):
    original = tmp_path / "original"
    first = Gateway(settings, "original-run", original)
    monkeypatch.setattr(first, "request", lambda *args: {"input_tokens": 100} if args[0] == "responses/input_tokens" else completed())
    first.generate("research", "instructions", "input")
    path = next((original / "calls").glob("*.json"))
    receipt = json.loads(path.read_text())
    receipt["text"] = "unverified replacement"
    path.write_text(json.dumps(receipt))
    resumed = Gateway(settings, "resumed-run", tmp_path / "resumed", cache_dir=original)
    calls = []

    def request(path, payload):
        if path == "responses/input_tokens":
            return {"input_tokens": 100}
        calls.append(path)
        return completed("fresh answer")

    monkeypatch.setattr(resumed, "request", request)
    assert resumed.generate("research", "instructions", "input") == "fresh answer"
    assert calls == ["responses"]


def test_http_transport_uses_the_configured_timeout(settings, tmp_path, monkeypatch):
    gateway = Gateway(settings, "transport", tmp_path / "out")
    observed = []

    def urlopen(request, timeout):
        observed.append(timeout)
        return BytesIO(b'{"id": "fake-model"}')

    monkeypatch.setattr("rag.llm.urllib.request.urlopen", urlopen)
    assert gateway.lookup()["model_id"] == "fake-model"
    assert observed == [settings.integer("OPENAI_TIMEOUT_SECONDS", 1)]
