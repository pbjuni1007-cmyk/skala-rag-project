from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
import json
import pytest
from rag.budget import Budget, BudgetExceeded
from rag.settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings({"OPENAI_API_KEY": "test-only", "OPENAI_MODEL": "gpt-5.6-luna", "OPENAI_REASONING_EFFORT": "max",
                     "PRICING_CHECKED_AT": date.today().isoformat(), "USD_TO_KRW": "1500",
                     "OPENAI_INPUT_USD_PER_MILLION_TOKENS": "0.20", "OPENAI_OUTPUT_USD_PER_MILLION_TOKENS": "1.20",
                     "BUDGET_LEDGER_PATH": str(tmp_path / "ledger.json")})


def test_concurrent_reservations_never_exceed_limit(settings):
    settings.values["PROJECT_SPEND_LIMIT_KRW"] = "30"
    def reserve(_):
        try:
            Budget(settings).reserve("run", "concurrent", 24000, 8000)
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve, range(10))) == 1
    assert Budget(settings).summary()["conservative_total_krw"] <= 30


def test_uncertain_calls_and_new_runs_share_limit(settings):
    settings.values["PROJECT_SPEND_LIMIT_KRW"] = "30"
    first = Budget(settings)
    first.reserve("first-run", "timeout", 24000, 8000)
    with pytest.raises(BudgetExceeded):
        Budget(settings).reserve("second-run", "retry", 24000, 8000)
    assert first.summary()["unsettled_calls"] == 1


def test_usage_includes_reasoning_and_ignores_cache_discount(settings):
    budget = Budget(settings)
    reservation = budget.reserve("run", "paid", 24000, 8000)
    budget.settle(reservation, {"service_tier": "default", "id": "test", "usage": {
        "input_tokens": 100, "output_tokens": 1000, "input_tokens_details": {"cached_tokens": 100},
        "output_tokens_details": {"reasoning_tokens": 999}}})
    assert budget.summary()["conservative_total_krw"] == round(budget.cost(100, 1000), 4)
    assert budget.summary()["unsettled_calls"] == 0


@pytest.mark.parametrize("key,value", [("PROJECT_BUDGET_KRW", "50001"), ("PROJECT_SPEND_LIMIT_KRW", "51000"),
    ("OPENAI_INPUT_USD_PER_MILLION_TOKENS", "0"), ("USD_TO_KRW", "nan"), ("OPENAI_MODEL", "other"),
    ("OPENAI_REASONING_EFFORT", "invalid"), ("LLM_MAX_INPUT_TOKENS", "300000"), ("PRICING_CHECKED_AT", "2000-01-01")])
def test_invalid_paid_settings_fail_closed(settings, key, value):
    settings.values[key] = value
    with pytest.raises(ValueError):
        settings.validate_paid()


def test_wrong_tier_never_releases_reservation(settings):
    b = Budget(settings)
    r = b.reserve("run", "test", 24000, 8000)
    with pytest.raises(RuntimeError):
        b.settle(r, {"service_tier": "fast", "usage": {"input_tokens": 10, "output_tokens": 10}})
    assert b.summary()["unsettled_calls"] == 1
