from copy import deepcopy
import json
from pathlib import Path
import pytest
from rag.render import render_report
from test_render import report_fixture


def test_submission_writes_only_markdown_without_loading_fonts(tmp_path, report_fixture):
    report, joined, sources, settings, config = report_fixture
    settings.values["REPORT_FONT_PATH"] = "/nonexistent/no-font.ttf"
    paths = render_report(tmp_path, report, joined, sources, settings, config)
    text = Path(paths["markdown"]).read_text()
    assert not list(tmp_path.glob("*.pdf"))
    assert "pdf" not in paths
    assert text.startswith("# SUMMARY\n")
    assert text.rsplit("# ", 1)[1].startswith("REFERENCE\n")
    assert "Unused Source Title" not in text
    assert "KIVI Source Title" in text and "InfiniGen Source Title" in text
    checks = json.loads((tmp_path / "document_validation.json").read_text())
    assert checks["pdf_generated"] is False
    assert checks["semantic_review"] == "pending"
    assert "Quoted source evidence" in Path(paths["citation_review"]).read_text()


def test_submission_metadata_uses_md_extension(tmp_path, report_fixture):
    report, joined, sources, settings, config = report_fixture
    settings.values.update(REPORT_CAMPUS="판교", REPORT_CLASS="7반", REPORT_CONTRIBUTORS="가+나+다+라")
    paths = render_report(tmp_path, report, joined, sources, settings, config)
    assert Path(paths["markdown"]).name == "RAG-Output_판교_7반_가+나+다+라.md"


def test_overlong_summary_does_not_write_submission(tmp_path, report_fixture):
    report, joined, sources, settings, config = report_fixture
    joined["claims"]["claim-1"]["text"] = "긴 문장 " * 500
    with pytest.raises(ValueError, match="SUMMARY"):
        render_report(tmp_path, report, joined, sources, settings, config)
    assert not list(tmp_path.glob("*.md"))


def test_resolved_gap_is_archived_with_support_and_not_repeated_as_limit(tmp_path, report_fixture):
    report, joined, sources, settings, config = report_fixture
    joined["gap_records"] = [
        {"id": "gap-001", "perspective": "research", "text": "시나리오는 아직 작성되지 않았다."},
        {"id": "gap-002", "perspective": "domain", "text": "실제 채택은 여전히 미확인이다."},
    ]
    report["gap_decisions"] = [
        {"gap_id": "gap-001", "status": "resolved", "resolution": "후속 관점에서 적용 가정을 작성했다.", "claim_ids": ["claim-1"]},
        {"gap_id": "gap-002", "status": "unresolved", "resolution": "실제 채택 근거가 없다.", "claim_ids": []},
    ]
    paths = render_report(tmp_path, report, joined, sources, settings, config)
    body, gaps = Path(paths["markdown"]).read_text(), Path(paths["gap_review"]).read_text()
    assert "시나리오는 아직 작성되지 않았다." not in body
    assert "실제 채택은 여전히 미확인이다." in body
    assert "시나리오는 아직 작성되지 않았다." in gaps and "claim-1" in gaps
    assert "후속 관점에서 적용 가정을 작성했다." in gaps
    assert "**사람 검수:** 미검수" in gaps


def test_legacy_gaps_are_not_silently_considered_resolved(tmp_path, report_fixture):
    report, joined, sources, settings, config = report_fixture
    paths = render_report(tmp_path, report, joined, sources, settings, config)
    assert joined["gaps"][0] in Path(paths["markdown"]).read_text()
    assert "unreviewed" in Path(paths["gap_review"]).read_text()


def test_repeated_legacy_body_claims_appear_once_plus_summary(tmp_path, report_fixture):
    report, joined, sources, settings, config = report_fixture
    paths = render_report(tmp_path, report, joined, sources, settings, config)
    body = Path(paths["markdown"]).read_text()
    assert body.count(joined["claims"]["claim-1"]["text"]) == 2


def test_comparison_cells_link_full_conditions_and_escape_claim_pipes(tmp_path, report_fixture):
    report, joined, sources, settings, config = report_fixture
    report["sections"][0]["claim_ids"] = []
    joined["claims"]["claim-1"].update(facet="costs", text="비용 | 선택 조건", conditions="A100 | batch 8", caveats="다른 설정과 비교 금지")
    joined["claims"]["claim-2"]["facet"] = "costs"
    paths = render_report(tmp_path, report, joined, sources, settings, config)
    body = Path(paths["markdown"]).read_text()
    assert "| 비교 질문 | KIVI | InfiniGen |" in body
    assert "비용 &#124; 선택 조건" in body
    assert "A100" not in body
    assert "다른 설정과 비교 금지" not in body
    assert "[조건·한계·원문](citation_review.md#claim-1)" in body
    assert "**한계:** 다른 설정과 비교 금지" in Path(paths["citation_review"]).read_text()
    assert "**조건:** A100 | batch 8" in Path(paths["citation_review"]).read_text()


@pytest.mark.parametrize("technology, label", [("KIVI", "KIVI"), ("InfiniGen", "InfiniGen"), ("both", "KIVI · InfiniGen")])
def test_summary_names_technology_even_when_claim_uses_an_implicit_subject(tmp_path, report_fixture, technology, label):
    report, joined, sources, settings, config = report_fixture
    joined["claims"]["claim-1"].update(technology=technology, text="운영자의 관리 부담을 함께 고려한다.")
    paths = render_report(tmp_path, report, joined, sources, settings, config)
    summary = Path(paths["markdown"]).read_text().split("**대상 기술:**")[0]
    assert f"**{label}** · [출처 사실] 운영자의 관리 부담을 함께 고려한다." in summary


def test_table_moves_long_details_without_mutating_or_losing_originals(tmp_path, report_fixture):
    report, joined, sources, settings, config = report_fixture
    report["sections"][0]["claim_ids"] = []
    target = joined["claims"]["claim-1"]
    target.update(facet="costs", conditions="실험 설정과 원문 해시 " + "a" * 64 + " 조건 | 값\n" * 300,
                  caveats="남은 확인 항목 " + "다른 실험 조건\n" * 200)
    joined["claims"]["claim-2"]["facet"] = "costs"
    before = deepcopy(joined)
    paths = render_report(tmp_path, report, joined, sources, settings, config)
    body, annex = Path(paths["markdown"]).read_text(), Path(paths["citation_review"]).read_text()
    assert "a" * 64 not in body
    assert target["conditions"] in annex and target["caveats"] in annex
    assert target["text"] in body and target["text"] in annex
    assert target["references"][0]["quote"] in annex
    assert "[1, 물리 p.1]" in body
    assert "[조건·한계·원문](citation_review.md#claim-1)" in body
    assert joined == before


def test_complete_report_keeps_every_body_claim_once_and_every_reference(tmp_path):
    from rag.evidence import collect_evidence
    from rag.settings import Settings
    from test_pipeline_improvements import report_fixture as full_fixture, chunks

    _, joined, claims, report = full_fixture(tmp_path)
    _, extra = collect_evidence({"synthesis": {"claims": report["synthesis_claims"]}}, chunks())
    joined["claims"] = claims
    joined["evidence"].update(extra)
    for cid, claim in claims.items():
        claim["text"] = f"固有본문-{cid}-끝"
    sources = {tech: {"authors": "Test Author", "title": tech + " paper", "date": "2024",
                      "version": "v1", "type": "paper_pool", "accessed_at": "2026-09-22",
                      "url": "https://example.invalid/" + tech} for tech in ("KIVI", "InfiniGen")}
    paths = render_report(tmp_path, report, joined, sources, Settings({}),
                          {"domain": "문서 검토", "scenario": "테스트 가정"})
    text = Path(paths["markdown"]).read_text()
    body = text.split("# 기술 성숙도\n", 1)[1].split("# REFERENCE\n", 1)[0]
    annex = Path(paths["citation_review"]).read_text()
    assert len(claims) == 28
    for cid, claim in claims.items():
        assert body.count(claim["text"]) == 1
        assert f"## {cid}\n" in annex
        for eid in claim["evidence_ids"]:
            assert eid in annex
            assert joined["evidence"][eid]["quote"] in annex
    checks = json.loads((tmp_path / "document_validation.json").read_text())
    assert checks["body_unique_claims"] == 28
    assert set(checks["used_sources"]) == {"KIVI", "InfiniGen"}
