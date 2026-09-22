from pathlib import Path
from html import escape, unescape
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, Table, TableStyle
from pypdf import PdfReader

from rag.budget import write_json
from rag.conflicts import conflicts_for_joined, conflict_review_markdown

LABELS = {"source_fact": "출처 사실", "author_reported_result": "저자 보고 결과", "team_inference": "팀 추론",
          "scenario": "적용 가정", "unknown": "미확인"}


def fonts(settings):
    regular = Path(settings.get("REPORT_FONT_PATH", "assets/fonts/NanumGothic-Regular.ttf"))
    bold = Path(settings.get("REPORT_FONT_BOLD_PATH", "assets/fonts/NanumGothic-Bold.ttf"))
    if not regular.is_file() or not bold.is_file():
        raise ValueError("Korean TTF font missing; see README")
    pdfmetrics.registerFont(TTFont("Korean", str(regular)))
    pdfmetrics.registerFont(TTFont("KoreanBold", str(bold)))
    pdfmetrics.registerFontFamily("Korean", normal="Korean", bold="KoreanBold", italic="Korean", boldItalic="KoreanBold")


def styles():
    base = dict(fontName="Korean", wordWrap="CJK", textColor=colors.HexColor("#172B3A"))
    return {
        "title": ParagraphStyle("title", **base, fontSize=22, leading=30, spaceAfter=12),
        "h1": ParagraphStyle("h1", **{**base, "fontName": "KoreanBold"}, fontSize=15, leading=22, spaceBefore=14, spaceAfter=12, keepWithNext=True),
        "h2": ParagraphStyle("h2", **{**base, "fontName": "KoreanBold"}, fontSize=10.5, leading=16, spaceBefore=8, spaceAfter=5, keepWithNext=True),
        "body": ParagraphStyle("body", **base, fontSize=10, leading=16, spaceAfter=8),
        "small": ParagraphStyle("small", **{**base, "textColor": colors.HexColor("#526673")}, fontSize=8, leading=12, spaceAfter=6),
    }


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#C9D7DF"))
    canvas.line(48, 42, A4[0] - 48, 42)
    canvas.setFont("Korean", 8)
    canvas.setFillColor(colors.HexColor("#526673"))
    canvas.drawString(48, 28, "SKALA | KV Cache Evidence Review | 검토용")
    canvas.drawRightString(A4[0] - 48, 28, str(doc.page))
    canvas.restoreState()


def filename(settings, kind):
    campus, classroom, people = [settings.get(k) for k in ("REPORT_CAMPUS", "REPORT_CLASS", "REPORT_CONTRIBUTORS")]
    if not (campus and classroom and people):
        return "RAG-" + kind + "_review.pdf"
    if campus not in {"광주", "울산", "판교"} or len(people.split("+")) != 4:
        raise ValueError("Submission metadata requires valid campus and four contributors")
    if any(not re.fullmatch(r"[가-힣A-Za-z0-9+ _-]+", v) for v in (campus, classroom, people)):
        raise ValueError("Invalid submission filename metadata")
    campus_part = campus + "-" + classroom if kind == "Design" else campus + "_" + classroom
    return f"RAG-{kind}_{campus_part}_{people}.pdf"


def render_report(out, report, joined, sources, settings, config):
    """Complete only when both formats and their review sheets have been written."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pending = {"format": ["markdown", "pdf"], "pdf_generated": False, "render_status": "started"}
    for name in ("document_validation.json", "pdf_validation.json"):
        write_json(out / name, pending)
    try:
        return _render_report(out, report, joined, sources, settings, config)
    except Exception as exc:
        failure = {**pending, "render_status": "failed", "render_error": str(exc), "semantic_review": "pending"}
        for name in ("document_validation.json", "pdf_validation.json"):
            write_json(out / name, failure)
        raise


def _render_report(out, report, joined, sources, settings, config):
    """Write matching Markdown/PDF reports and complete Markdown review sheets."""
    import hashlib
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    claims, evidence = joined["claims"], joined["evidence"]
    conflict_records = conflicts_for_joined(joined)
    used_claims = list(dict.fromkeys(report["summary_claim_ids"] + [c for section in report["sections"] for c in section["claim_ids"]]))
    used_evidence = list(dict.fromkeys(e for c in used_claims for e in claims[c]["evidence_ids"]))
    used_sources = sorted({evidence[e]["source_id"] for e in used_evidence})
    numbering = {source: i + 1 for i, source in enumerate(used_sources)}
    gap_records = joined.get("gap_records", [{"id": f"legacy-{i}", "perspective": "legacy", "text": gap}
                                           for i, gap in enumerate(joined.get("gaps", []), 1)])
    decisions = {d["gap_id"]: d for d in report.get("gap_decisions", [])}

    def citations(claim):
        refs = []
        for eid in claim["evidence_ids"]:
            ev = evidence[eid]
            where = f"물리 p.{ev['page']}" if ev.get("page") else ev["section"]
            refs.append(f"[{numbering[ev['source_id']]}, {where}]")
        return " ".join(dict.fromkeys(refs))

    def cell(text):
        return text.replace("|", "&#124;").replace("\n", "<br>")

    def claim_text(cid, table=False):
        c = claims[cid]
        parts = [f"**{LABELS[c['kind']]} · {cid}**", c["text"] + " " + citations(c)]
        if table:
            parts.append(f"[조건·한계·원문](citation_review.md#{cid})")
        else:
            for label, key in (("조건", "conditions"), ("한계", "caveats")):
                if c[key].strip():
                    parts.append(f"**{label}:** {c[key]}")
        return "<br><br>".join(cell(x) for x in parts) if table else "\n\n".join(parts)

    def technology_label(claim):
        return "KIVI · InfiniGen" if claim["technology"] == "both" else claim["technology"]

    summary = [f"**{technology_label(claims[c])}** · [{LABELS[claims[c]['kind']]}] {claims[c]['text']} {citations(claims[c])}"
               for c in report["summary_claim_ids"]]
    if len("\n".join(summary)) > 1200:
        raise ValueError("SUMMARY draft is too long; shorten before user PDF conversion")
    md = ["# SUMMARY", "", *[part for text in summary for part in (text, "")],
          "**대상 기술:** KIVI / InfiniGen  ", "**단일 도메인:** " + config["domain"], "",
          "**적용 가정:** " + config["scenario"], ""]
    facets = {"adoption": "채택 동기·공개 신호", "alternatives": "대안·연동", "costs": "비용·유지 부담",
              "user": "문서 검토자", "operator": "AI·인프라 운영자", "governance": "구매·보안·관리 담당자",
              "fit": "적합 조건", "risks": "정확도·운영 위험", "evaluation": "확인할 실험"}
    body_seen = set()
    for section in report["sections"]:
        title = section["title"]
        md.extend(["# " + title, ""])
        # Validation rejects repeated body claims in new runs. Old saved runs can
        # still be exported without duplicating every paragraph in the final section.
        ids = list(dict.fromkeys(cid for cid in section["claim_ids"] if cid not in body_seen))
        body_seen.update(ids)
        if title in {"시장성", "이해관계자", "도메인 적용"} and ids:
            md.extend(["| 비교 질문 | KIVI | InfiniGen |", "| --- | --- | --- |"])
            order = list(dict.fromkeys(claims[cid]["facet"] for cid in ids if claims[cid]["technology"] != "both"))
            for facet in order:
                cells = []
                for tech in ("KIVI", "InfiniGen"):
                    matches = [cid for cid in ids if claims[cid]["technology"] == tech and claims[cid]["facet"] == facet]
                    cells.append("<br><br>".join(claim_text(cid, True) for cid in matches) or "이 질문의 별도 주장은 제시되지 않음")
                md.append(f"| {cell(facets.get(facet, facet))} | {cells[0]} | {cells[1]} |")
            md.append("")
            for cid in ids:
                if claims[cid]["technology"] == "both":
                    md.extend([claim_text(cid), ""])
        else:
            for cid in ids:
                md.extend([f"## {claims[cid]['technology']} · {cid}", "", claim_text(cid), ""])
        if title == "관점 간 상충과 한계":
            unresolved = [g for g in gap_records if decisions.get(g["id"], {}).get("status") != "resolved"]
            md.extend(["## 남은 근거 공백", ""])
            if unresolved:
                grouped = {}
                for gap in unresolved:
                    grouped.setdefault(gap["text"].strip(), []).append(gap["id"])
                for text, gids in grouped.items():
                    md.extend([f"- {text} ({', '.join(gids)})", ""])
            else:
                md.extend(["종합 단계의 해소 판단과 근거를 공백 검수표에 정리했습니다.", ""])
            md.extend(["공백별 판단 근거와 후속 확인 항목: [공백 검수표](gap_review.md).", ""])
            md.extend(["상충의 원문·관련 주장 후보·조건·해소 상태: [상충 검수표](conflict_review.md).", ""])
    md.extend(["# REFERENCE", ""])
    for sid in used_sources:
        source = sources[sid]
        date = source.get("date", "unknown")
        date = "발행일 미상" if date == "unknown" else date
        author = source["authors"].rstrip(".")
        version = source.get("version", "unknown")
        venue = f"arXiv, {version}" if source.get("type") == "paper_pool" else f"공식 웹 자료, 스냅샷 {version[:12]}"
        md.extend([f"[{numbering[sid]}] {author} ({date}). **{source['title']}**. {venue}. "
                   f"조회 {source['accessed_at'][:10]}.  ", source["url"], ""])
    document = "\n".join(md)
    path = out / filename(settings, "Output").replace(".pdf", ".md")
    path.write_text(document)
    (out / "report.md").write_text(document)

    review_claims = list(dict.fromkeys(used_claims + [cid for d in decisions.values() for cid in d["claim_ids"]]
                        + [cid for r in conflict_records for cid in r["candidate_claim_ids"] + r["verified_claim_ids"]]))
    review = ["# 인용 검수", "", "각 주장의 전체 본문·실험조건·한계를 원문과 대조하는 검수표입니다. 현재 의미 검수는 대기 중입니다.", ""]
    for cid in review_claims:
        c = claims[cid]
        review.extend(["## " + cid, "", f"**기술:** {technology_label(c)} · **주장 종류:** {LABELS[c['kind']]}", "",
                       c["text"], "", "**조건:** " + c["conditions"], "",
                       "**한계:** " + c["caveats"], "", "판정: 미검수", ""])
        for eid in c["evidence_ids"]:
            ev = evidence[eid]
            review.extend([f"- {eid} | {ev['source_id']} | 물리 페이지 {ev.get('page')} | {ev['section']}", "",
                           "> " + ev["quote"], ""])
    (out / "citation_review.md").write_text("\n".join(review))
    gap_review = ["# 근거 공백 검수", "", "원래 공백을 삭제하지 않고 종합 시점의 판단과 근거를 보존합니다. 해소 여부의 의미 검수는 사람에게 남깁니다.", ""]
    for gap in gap_records:
        decision = decisions.get(gap["id"], {"status": "unreviewed", "resolution": "이전 실행: 종합 공백 판정 없음", "claim_ids": []})
        gap_review.extend([f"## {gap['id']} · {gap['perspective']}", "", "**원래 공백:** " + gap["text"], "",
                           "**종합 판단:** " + decision["status"], "", "**판단 이유:** " + decision["resolution"], "",
                           "**근거 주장:** " + (", ".join(f"[{cid}](citation_review.md#{cid})" for cid in decision["claim_ids"]) or "없음"),
                           "", "**사람 검수:** 미검수", ""])
    (out / "gap_review.md").write_text("\n".join(gap_review))
    (out / "conflict_review.md").write_text(conflict_review_markdown(conflict_records, claims))
    write_json(out / "conflict_records.json", conflict_records)
    pdf_path, pdf_checks = _write_pdf(out, document, settings)
    pdf_checks.update(used_sources=used_sources, used_claims=used_claims, used_evidence=used_evidence)
    write_json(out / "pdf_validation.json", pdf_checks)
    write_json(out / "document_validation.json", {"format": ["markdown", "pdf"], "pdf_generated": True,
        "summary_characters": len("\n".join(summary)), "pdf_half_page": "passed", "pdf_pages": pdf_checks["pdf_pages"],
        "used_sources": used_sources, "used_claims": used_claims, "used_evidence": used_evidence,
        "body_unique_claims": len(body_seen), "gap_count": len(gap_records),
        "conflict_count": len(conflict_records),
        "resolved_conflicts": sum(r["status"] == "resolved" for r in conflict_records),
        "resolved_gaps": sum(decisions.get(g["id"], {}).get("status") == "resolved" for g in gap_records),
        "semantic_review": "pending", "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "format_decision": "Markdown and PDF generated from the same report; semantic review remains pending"})
    return {"markdown": str(path), "pdf": str(pdf_path), "citation_review": str(out / "citation_review.md"),
            "gap_review": str(out / "gap_review.md"), "conflict_review": str(out / "conflict_review.md")}


def _pdf_inline(text):
    """Translate the renderer's small Markdown subset without interpreting source HTML."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = unescape(text.replace("<br>", "\n"))
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.replace("\n", "<br/>")


def _write_pdf(out, document, settings):
    """Render the exact Markdown body; never rewrite reports or review annexes."""
    import hashlib
    from tempfile import NamedTemporaryFile

    fonts(settings)
    st = styles()
    st["label"] = ParagraphStyle("claim-label", parent=st["body"], keepWithNext=True)
    st["cell"] = ParagraphStyle("cell", parent=st["body"], fontSize=8.5, leading=13, spaceAfter=0)
    width = A4[0] - 108  # Frame padding inside the 48pt margins.
    lines = document.splitlines()
    story, summary_flowables = [], []
    in_summary, first_chapter = True, True
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("**대상 기술:**"):
            in_summary = False
        if line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].startswith("| "):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c) for c in cells):
                    rows.append([Paragraph(_pdf_inline(c), st["cell"]) for c in cells])
                i += 1
            n = len(rows[0])
            widths = [width * .16, width * .42, width * .42] if n == 3 else [width / n] * n
            table = Table(rows, colWidths=widths, repeatRows=1, splitByRow=1, splitInRow=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3F3")),
                ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#C9D7DF")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))
            story.extend([table, Spacer(1, 10)])
            continue
        flowable = None
        if line.startswith("# "):
            if not first_chapter:
                story.append(PageBreak())
            first_chapter = False
            flowable = Paragraph(_pdf_inline(line[2:]), st["h1"])
        elif line.startswith("## "):
            flowable = Paragraph(_pdf_inline(line[3:]), st["h2"])
        elif line.strip():
            flowable = Paragraph(_pdf_inline(line), st["label"] if re.fullmatch(r"\*\*[^*]+\*\*", line) else st["body"])
        if flowable:
            story.append(flowable)
            if in_summary:
                summary_flowables.append(flowable)
        i += 1
    summary_height = sum(f.wrap(width, A4[1])[1] + f.getSpaceBefore() + f.getSpaceAfter()
                         for f in summary_flowables)
    summary_bottom = 48 + summary_height
    if summary_bottom > A4[1] / 2:
        raise ValueError("SUMMARY exceeds half of the physical page; revise the summary selection")
    # The PDF explains where its linked Markdown annexes live without adding a new chapter.
    story.insert(len(summary_flowables), Paragraph(
        "상세 조건·원문: 같은 실행 폴더의 citation_review.md / 공백 판단: gap_review.md", st["small"]))
    path = out / filename(settings, "Output")
    with NamedTemporaryFile(dir=out, suffix=".pdf", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        SimpleDocTemplate(str(temporary), pagesize=A4, leftMargin=48, rightMargin=48,
                          topMargin=42, bottomMargin=55, title="KV Cache 다관점 평가", author="SKALA team").build(
                              story, onFirstPage=footer, onLaterPages=footer)
        reader = PdfReader(temporary)
        if "SUMMARY" not in reader.pages[0].extract_text() or not any("REFERENCE" in page.extract_text() for page in reader.pages) or not document.rsplit("\n# ", 1)[-1].startswith("REFERENCE\n"):
            raise ValueError("PDF chapter layout validation failed")
        checks = {"summary_height_pt": round(summary_height, 2), "summary_bottom_pt": round(summary_bottom, 2),
                  "half_page_limit_pt": A4[1] / 2, "pdf_pages": len(reader.pages), "pdf_generated": True,
                  "pdf_sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
                  "markdown_sha256": hashlib.sha256(document.encode()).hexdigest(),
                  "visual_review": "pending", "semantic_review": "pending"}
        temporary.replace(path)
        return path, checks
    finally:
        temporary.unlink(missing_ok=True)


def render_pdf_report(out, report, joined, sources, settings, config):
    """Compatibility entry point with the same dual-format contract."""
    return render_report(out, report, joined, sources, settings, config)
