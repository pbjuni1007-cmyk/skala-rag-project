from copy import deepcopy
import json
from pathlib import Path

from pypdf import PdfReader
import pytest
from reportlab.lib.pagesizes import A4

from rag.evidence import validate_assessment
from rag.render import filename, render_pdf_report as render_report
from rag.schemas import Reference
from rag.settings import Settings


@pytest.fixture
def report_fixture():
    root = Path(__file__).resolve().parents[1]
    settings = Settings({"REPORT_FONT_PATH": str(root / "assets/fonts/NanumGothic-Regular.ttf"),
                         "REPORT_FONT_BOLD_PATH": str(root / "assets/fonts/NanumGothic-Bold.ttf")})
    claims, evidence, sources = {}, {}, {}
    for number, technology in enumerate(("KIVI", "InfiniGen", "Unused"), 1):
        cid, eid, sid = f"claim-{number}", f"evidence-{number}", f"source-{number}"
        claims[cid] = {"id": cid, "technology": technology, "kind": "source_fact", "text": f"{technology} 기술의 근거 있는 평가입니다.",
                       "facet": "mechanism", "conditions": "공개 실험 조건", "caveats": "실제 도입 여부는 미확인",
                       "evidence_ids": [eid], "references": [{"chunk_id": cid, "quote": f"Quoted source evidence for {technology}."}]}
        evidence[eid] = {"id": eid, "source_id": sid, "page": number, "section": f"Section {number}",
                         "quote": f"Quoted source evidence for {technology}."}
        sources[sid] = {"authors": f"Author {number}", "title": f"{technology} Source Title", "date": "2024",
                        "version": f"v{number}", "accessed_at": "2026-09-21T00:00:00+00:00", "url": f"https://example.invalid/{number}"}
    report = {"summary_claim_ids": ["claim-1", "claim-2"], "synthesis_claims": [],
              "sections": [{"title": title, "claim_ids": ["claim-1", "claim-2"]} for title in
                           ("기술 성숙도", "시장성", "이해관계자", "도메인 적용", "관점 간 상충과 한계")]}
    joined = {"claims": claims, "evidence": evidence, "gaps": ["기업의 실제 도입률은 공개 근거가 없습니다."]}
    return report, joined, sources, settings, {"domain": "문서 검토 지원", "scenario": "시험용 업무 가정"}


@pytest.mark.parametrize("kind,expected", [("Design", "RAG-Design_판교-1반_가+나+다+라.pdf"),
                                           ("Output", "RAG-Output_판교_1반_가+나+다+라.pdf")])
def test_submission_filename_preserves_required_delimiters(kind, expected):
    settings = Settings({"REPORT_CAMPUS": "판교", "REPORT_CLASS": "1반", "REPORT_CONTRIBUTORS": "가+나+다+라"})
    assert filename(settings, kind) == expected


@pytest.mark.parametrize("kind", ["Design", "Output"])
def test_missing_submission_metadata_uses_review_filename(kind):
    assert filename(Settings({}), kind) == f"RAG-{kind}_review.pdf"


@pytest.mark.parametrize("key,value", [("REPORT_CAMPUS", "서울"), ("REPORT_CONTRIBUTORS", "가+나+다"),
                                       ("REPORT_CLASS", "../../escape"), ("REPORT_CONTRIBUTORS", "가+나+다+라/")])
def test_invalid_submission_metadata_cannot_create_a_submission_filename(key, value):
    values = {"REPORT_CAMPUS": "판교", "REPORT_CLASS": "1반", "REPORT_CONTRIBUTORS": "가+나+다+라"}
    values[key] = value
    with pytest.raises(ValueError):
        filename(Settings(values), "Output")


def test_pdf_and_markdown_cite_only_used_sources_and_keep_review_quotes(tmp_path, report_fixture):
    report, joined, sources, settings, config = report_fixture
    paths = render_report(tmp_path, report, joined, sources, settings, config)
    reader = PdfReader(paths["pdf"])
    pdf_text = "\n".join(page.extract_text() for page in reader.pages)
    markdown = Path(paths["markdown"]).read_text()
    review = Path(paths["citation_review"]).read_text()
    checks = json.loads((tmp_path / "pdf_validation.json").read_text())
    first_page = reader.pages[0].extract_text()
    assert first_page.index("SUMMARY") < first_page.index("KIVI × InfiniGen")
    assert "REFERENCE" in reader.pages[-1].extract_text()
    assert checks["used_sources"] == ["source-1", "source-2"]
    assert checks["used_claims"] == ["claim-1", "claim-2"]
    assert checks["used_evidence"] == ["evidence-1", "evidence-2"]
    assert checks["summary_height_pt"] < A4[1] / 2
    for text in (pdf_text, markdown):
        assert "KIVI Source Title" in text and "InfiniGen Source Title" in text
        assert "Unused Source Title" not in text and "example.invalid/3" not in text
        assert "[1, p.1]" in text and "[2, p.2]" in text
        assert "2024" in text and "2026-09-21" in text and "v1" in text
    assert "Quoted source evidence for KIVI." in review
    assert "evidence-1 | source-1 | 물리 페이지 1" in review
    assert "Unused" not in review
    assert checks["visual_review"] == "pending" and checks["semantic_review"] == "pending"


def test_summary_over_half_a_page_is_rejected_before_pdf_creation(tmp_path, report_fixture):
    report, joined, sources, settings, config = report_fixture
    joined["claims"]["claim-1"]["text"] = "요약은 물리 페이지의 절반을 넘어서는 안 됩니다. " * 300
    with pytest.raises(ValueError, match="SUMMARY exceeds half"):
        render_report(tmp_path, report, joined, sources, settings, config)
    assert not list(tmp_path.glob("*.pdf"))
    assert not (tmp_path / "pdf_validation.json").exists()


@pytest.mark.parametrize("quote", ["", "too short", "This quotation is absent from the actual source."])
def test_missing_or_unverifiable_quote_is_rejected_before_report_selection(report_fixture, quote):
    _, joined, _, _, _ = report_fixture
    claim = deepcopy(joined["claims"]["claim-1"])
    claim["references"][0]["quote"] = quote
    errors = validate_assessment({"claims": [claim]}, [{"id": "claim-1", "text": "Quoted source evidence for KIVI."}])
    assert any("quotation does not exist" in error for error in errors)


def test_reference_schema_requires_a_quote():
    with pytest.raises(ValueError):
        Reference.model_validate({"chunk_id": "claim-1"})


@pytest.mark.parametrize("kind,conditions,references,accepted", [
    ("source_fact", "A100 GPU and the paper's stated model configuration", True, True),
    ("author_reported_result", "A100 GPU and the paper's stated model configuration", True, True),
    ("scenario", "An assumed enterprise workload", True, False),
    ("unknown", "The real deployment is unknown", True, False),
    ("team_inference", "An inferred deployment environment", True, False),
    ("source_fact", "", True, False),
    ("source_fact", "A100 GPU", False, False),
])
def test_core_experimental_conditions_require_sourced_facts(report_fixture, kind, conditions, references, accepted):
    _, joined, _, _, _ = report_fixture
    base = joined["claims"]["claim-1"]
    claims = []
    for facet in ("mechanism", "limitation", "maturity", "conditions"):
        claim = deepcopy(base)
        claim["facet"] = facet
        if facet == "maturity":
            claim.update(kind="team_inference", text="공개 근거로 추정한 TRL 4")
        elif facet == "conditions":
            claim.update(kind=kind, conditions=conditions)
            if not references:
                claim["references"] = []
        claims.append(claim)
    errors = validate_assessment({"claims": claims},
                                 [{"id": "claim-1", "text": "Quoted source evidence for KIVI."}],
                                 technology="KIVI", core=True)
    assert (errors == []) is accepted
