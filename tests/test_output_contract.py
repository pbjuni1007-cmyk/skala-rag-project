"""The generated report must follow the approved Markdown reading order."""
from copy import deepcopy

import pytest

from rag.evidence import report_errors
from test_pipeline_improvements import chunks, report_fixture


@pytest.mark.parametrize("summary_count", [2, 3])
def test_agreed_order_and_summary_size_pass(tmp_path, summary_count):
    _, joined, claims, report = report_fixture(tmp_path)
    report["summary_claim_ids"] = ["synthesis-1", "market-1", "domain-1", "stakeholder-1"][:summary_count]

    assert report_errors(report, claims, chunks(), joined["gap_records"]) == []


@pytest.mark.parametrize("summary_count", [0, 1, 4])
def test_outside_summary_size_is_rejected(tmp_path, summary_count):
    _, joined, claims, report = report_fixture(tmp_path)
    report["summary_claim_ids"] = ["synthesis-1", "market-1", "domain-1", "stakeholder-1"][:summary_count]

    errors = report_errors(report, claims, chunks(), joined["gap_records"])

    assert "SUMMARY requires 2..3 concise claims" in errors


@pytest.mark.parametrize("mutation", ["reverse", "swap", "missing", "duplicate"])
def test_section_order_missing_and_duplicates_are_rejected(tmp_path, mutation):
    _, joined, claims, report = report_fixture(tmp_path)
    sections = report["sections"]
    if mutation == "reverse":
        sections.reverse()
    elif mutation == "swap":
        sections[1], sections[2] = sections[2], sections[1]
    elif mutation == "missing":
        sections.pop(2)
    else:
        sections.insert(2, deepcopy(sections[1]))

    errors = report_errors(report, claims, chunks(), joined["gap_records"])

    assert "Report must contain exactly the five agreed body sections in order" in errors


@pytest.mark.parametrize("ids", [
    ["research_kivi-4", "research_infinigen-4"],
    ["synthesis-1", "synthesis-2"],
    ["market-1", "domain-1"],
])
def test_summary_requires_synthesis_and_nonresearch_judgment(tmp_path, ids):
    _, joined, claims, report = report_fixture(tmp_path)
    report["summary_claim_ids"] = ids
    assert "SUMMARY must include synthesis and a market/stakeholder/domain assessment" in report_errors(
        report, claims, chunks(), joined["gap_records"])


@pytest.mark.parametrize("perspective", ["market", "stakeholder", "domain"])
def test_summary_accepts_each_followup_perspective(tmp_path, perspective):
    _, joined, claims, report = report_fixture(tmp_path)
    report["summary_claim_ids"] = ["synthesis-1", f"{perspective}-1"]
    assert report_errors(report, claims, chunks(), joined["gap_records"]) == []


def test_summary_duplicate_ids_do_not_fill_three_slots(tmp_path):
    _, joined, claims, report = report_fixture(tmp_path)
    report["summary_claim_ids"] = ["synthesis-1", "market-1", "market-1"]
    assert "SUMMARY must not repeat claim IDs" in report_errors(report, claims, chunks(), joined["gap_records"])
