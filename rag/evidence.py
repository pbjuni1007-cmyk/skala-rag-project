"""Mechanical citation validation. Semantic support still needs a human review."""
from collections import Counter
from rag.corpus import normalized


def validate_assessment(result, chunks, technology=None, core=False):
    errors = []
    known = {c["id"]: c for c in chunks}
    for i, claim in enumerate(result["claims"]):
        refs = claim["references"]
        if claim["kind"] not in {"scenario", "unknown"} and not refs:
            errors.append(f"claim {i}: missing source evidence")
        if technology and claim["technology"] != technology:
            errors.append(f"claim {i}: wrong technology")
        for ref in refs:
            chunk = known.get(ref["chunk_id"])
            quote = normalized(ref["quote"])
            if not chunk or len(quote) < 12 or quote not in normalized(chunk["text"]):
                errors.append(f"claim {i}: quotation does not exist in supplied chunk {ref['chunk_id']}")
            if technology and chunk and chunk.get("technology") not in {None, technology, "both"}:
                errors.append(f"claim {i}: evidence belongs to another technology")
        if claim["kind"] == "author_reported_result" and not claim["conditions"].strip():
            errors.append(f"claim {i}: numerical result requires experimental conditions")
        if claim["kind"] in {"team_inference", "scenario", "unknown"} and not claim["caveats"].strip():
            errors.append(f"claim {i}: inference/scenario/unknown must disclose limitations")
    if core:
        for facet in ("mechanism", "limitation", "conditions", "maturity"):
            if not any(c["facet"] == facet and c["references"] for c in result["claims"]):
                errors.append(f"missing core facet: {facet}")
        for claim in result["claims"]:
            if claim["facet"] == "conditions" and (claim["kind"] not in {"source_fact", "author_reported_result"} or not claim["conditions"].strip()):
                errors.append("core conditions must be sourced experimental facts with nonempty conditions")
        maturity = [c for c in result["claims"] if c["facet"] == "maturity"]
        if not any(c["kind"] == "team_inference" and "TRL" in c["text"] for c in maturity):
            errors.append("TRL must be labeled as a team estimate from public evidence")
    return errors



CORE_FACETS = {"mechanism", "limitation", "conditions", "maturity"}
PERSPECTIVE_FACETS = {
    "market": {"adoption", "alternatives", "costs"},
    "stakeholder": {"user", "operator", "governance"},
    "domain": {"fit", "risks", "evaluation"},
}


def validate_retrieval_review(review, chunks, technology):
    """Check identities and coverage, not the truth of the model's sufficiency judgment."""
    errors = []
    items = review["items"]
    if len(items) != 4 or {i["facet"] for i in items} != CORE_FACETS:
        errors.append("Retrieval review requires exactly four distinct core facets")
    known = {c["id"]: c for c in chunks}
    for item in items:
        facet, ids = item["facet"], item["chunk_ids"]
        if len(ids) != len(set(ids)):
            errors.append(f"{facet}: duplicate chunk IDs")
        for chunk_id in ids:
            chunk = known.get(chunk_id)
            if chunk is None:
                errors.append(f"{facet}: unknown chunk ID {chunk_id}")
            elif chunk.get("technology") != technology:
                errors.append(f"{facet}: evidence belongs to another technology")
        if not item["reason"].strip():
            errors.append(f"{facet}: relevance/sufficiency reason is required")
        if item["sufficient"] and (not ids or item["missing"]):
            errors.append(f"{facet}: sufficient review needs evidence and no missing requirements")
        if not item["sufficient"] and not any(m.strip() for m in item["missing"]):
            errors.append(f"{facet}: insufficient review must describe missing evidence")
    return errors


def validate_perspective(result, chunks, name):
    errors = validate_assessment(result, chunks)
    expected = {(t, f) for t in ("KIVI", "InfiniGen") for f in PERSPECTIVE_FACETS[name]}
    pairs = [(c["technology"], c["facet"]) for c in result["claims"]]
    if len(pairs) != 6 or set(pairs) != expected:
        errors.append(f"{name}: exactly three distinct required facets per technology are required")
    return errors

def collect_evidence(assessments, chunks):
    lookup = {c["id"]: c for c in chunks}
    claims, evidence = {}, {}
    for perspective, result in assessments.items():
        for i, claim in enumerate(result.get("claims", []), 1):
            claim_id = f"{perspective}-{i}"
            evidence_ids = []
            for j, ref in enumerate(claim["references"], 1):
                evidence_id = f"E-{claim_id}-{j}"
                chunk = lookup[ref["chunk_id"]]
                evidence[evidence_id] = {"id": evidence_id, **ref, "source_id": chunk["source_id"],
                    "page": chunk.get("page"), "section": chunk.get("section"),
                    "conditions": claim["conditions"], "limitations": claim["caveats"]}
                evidence_ids.append(evidence_id)
            claims[claim_id] = {"id": claim_id, "perspective": perspective, **claim, "evidence_ids": evidence_ids}
    return claims, evidence


def report_errors(report, claims, chunks, gap_records=None):
    errors = []
    if not 2 <= len(report["summary_claim_ids"]) <= 3:
        errors.append("SUMMARY requires 2..3 concise claims")
    refs = list(report["summary_claim_ids"])
    if len(refs) != len(set(refs)):
        errors.append("SUMMARY must not repeat claim IDs")
    summary_perspectives = {claims[c].get("perspective") for c in refs if c in claims}
    if "synthesis" not in summary_perspectives or not summary_perspectives.intersection({"market", "stakeholder", "domain"}):
        errors.append("SUMMARY must include synthesis and a market/stakeholder/domain assessment")
    required = ["기술 성숙도", "시장성", "이해관계자", "도메인 적용", "관점 간 상충과 한계"]
    if [s["title"] for s in report["sections"]] != required:
        errors.append("Report must contain exactly the five agreed body sections in order")
    body_ids = []
    for section in report["sections"]:
        body_ids.extend(section["claim_ids"])
        if section["title"] == "관점 간 상충과 한계" and any(
            c not in claims or claims[c].get("perspective") != "synthesis" for c in section["claim_ids"]
        ):
            errors.append("Final body section must contain synthesis claims only")
        refs.extend(section["claim_ids"])
        if not section["claim_ids"]:
            errors.append("Empty report section")
        if section["title"] != "관점 간 상충과 한계":
            technologies = {claims[c]["technology"] for c in section["claim_ids"] if c in claims}
            if not ({"KIVI", "InfiniGen"} <= technologies or "both" in technologies):
                errors.append(f"{section['title']}: both technologies must be represented")
        if section["title"] == "기술 성숙도":
            for tech in ("KIVI", "InfiniGen"):
                facets = {claims[c]["facet"] for c in section["claim_ids"] if c in claims and claims[c]["technology"] == tech}
                if not {"mechanism", "limitation", "conditions", "maturity"} <= facets:
                    errors.append(f"{tech}: retain all four research facets in the technology section")
    missing = set(claims) - set(body_ids)
    if missing:
        errors.append("Every claim must appear exactly once in body: missing " + ", ".join(sorted(missing)))
    expected_sections = {"market": "시장성", "stakeholder": "이해관계자", "domain": "도메인 적용", "synthesis": "관점 간 상충과 한계"}
    for section in report["sections"]:
        for claim_id in section["claim_ids"]:
            perspective = claims.get(claim_id, {}).get("perspective", "")
            expected = "기술 성숙도" if perspective.startswith("research_") else expected_sections.get(perspective)
            if expected and section["title"] != expected:
                errors.append(f"{claim_id}: wrong perspective section; expected {expected}")
    if any(count > 1 for count in Counter(body_ids).values()):
        errors.append("A claim must not repeat across or within body sections; summary reuse is allowed")
    if gap_records is not None:
        errors.extend(gap_decision_errors(report.get("gap_decisions", []), gap_records, claims))
    for claim_id in refs:
        if claim_id not in claims:
            errors.append(f"Unknown report claim {claim_id}")
    errors.extend(validate_assessment({"claims": report["synthesis_claims"]}, chunks))
    return errors


def gap_decision_errors(decisions, gap_records, claims):
    errors = []
    expected = {g["id"] for g in gap_records}
    ids = [d["gap_id"] for d in decisions]
    if len(ids) != len(expected) or set(ids) != expected:
        errors.append("Every input gap must have exactly one decision; unknown or duplicate gaps are forbidden")
    for decision in decisions:
        claim_ids = decision["claim_ids"]
        if decision["status"] not in {"resolved", "unresolved"}:
            errors.append(f"{decision['gap_id']}: invalid gap status")
        if not decision["resolution"].strip():
            errors.append(f"{decision['gap_id']}: gap decision requires an explanation")
        if any(c not in claims or claims[c].get("perspective") == "synthesis" for c in claim_ids):
            errors.append(f"{decision['gap_id']}: gap evidence must use existing assessment claim IDs")
        if decision["status"] == "resolved" and (not claim_ids or not any(
            c in claims and claims[c]["kind"] != "unknown" for c in claim_ids
        )):
            errors.append(f"{decision['gap_id']}: resolved gap needs evidence beyond unknown claims")
    return errors
