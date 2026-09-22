"""Deterministic pipeline contracts; no keys, network, embeddings or paid calls."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from rag.evidence import (CORE_FACETS, PERSPECTIVE_FACETS, collect_evidence,
                          report_errors, validate_retrieval_review)
from rag.graph import Pipeline, StructuredValidationError
from rag.schemas import Queries, Report

TECHS = ("KIVI", "InfiniGen")
FACETS = ("mechanism", "limitation", "conditions", "maturity")
QUOTE = "The experiment uses a documented laboratory GPU configuration."


def chunks():
    return [{"id": tech, "technology": tech, "source_id": tech, "page": 1,
             "text": QUOTE + " The complete source also describes separate batch and accuracy experiments."}
            for tech in TECHS]


def claim(tech="KIVI", facet="mechanism", **changes):
    value = {"technology": tech, "facet": facet, "text": "원문에 근거한 분석",
             "kind": "source_fact", "references": [{"chunk_id": tech, "quote": QUOTE}],
             "conditions": "공통 장비 A100; 별도 정확도 실험과 배치 조건은 병합하지 않음",
             "caveats": "기업 운용 검증은 확인되지 않음"}
    if facet == "maturity":
        value.update(kind="team_inference", text="공개 실험실 근거로 추정한 TRL 4")
    return {**value, **changes}


def assessment(tech="KIVI", facets=FACETS):
    return {"status": "ok", "claims": [claim(tech, f) for f in facets], "conflicts": [], "gaps": []}


def queries(suffix=""):
    return {"queries": [{"technology": "KIVI", "facet": f, "query": f + suffix} for f in FACETS]}


def review(deficient=False):
    return {"items": [{"facet": f, "sufficient": not (deficient and f == "conditions"),
                       "chunk_ids": ["KIVI"], "reason": "GPU configuration missing" if deficient and f == "conditions" else "Specific source support",
                       "missing": ["A100 experimental setup section"] if deficient and f == "conditions" else []}
                      for f in FACETS]}


class FakeGateway:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, purpose, instructions, content, schema):
        self.calls.append((purpose, instructions, json.loads(content), schema))
        response = next(self.responses)
        return response if isinstance(response, str) else json.dumps(response)


def pipeline(tmp_path, responses=()):
    corpus = SimpleNamespace(chunks=chunks(), sources={}, searches=[], web_queries=[])

    def search(query, technology, top_k):
        corpus.searches.append((query, technology, top_k))
        return [c for c in corpus.chunks if c["technology"] == technology]

    def web_search(query, top_k):
        corpus.web_queries.append(query)
        return []

    corpus.search, corpus.web_search = search, web_search
    return Pipeline(SimpleNamespace(integer=lambda name, default: default),
                    {"top_k": 3, "domain": "문서 검토", "scenario": "업무 가정", "technologies": list(TECHS)},
                    corpus, FakeGateway(responses), tmp_path)


def test_retrieval_review_precedes_normal_assessment_and_saves_diagnostics(tmp_path):
    p = pipeline(tmp_path, [review(), assessment()])
    _, result, qlog, hits = p.research_technology("KIVI", queries()["queries"])
    assert result["status"] == "ok" and len(qlog) == len(hits) == 4
    assert [c[0] for c in p.gateway.calls] == ["retrieval_review_kivi_0", "research_kivi_0"]
    assert result["retrieval_diagnostics"][0]["status"] == "accepted"
    stored = json.loads((tmp_path / "nodes/research_kivi.json").read_text())
    assert stored["retrieval_diagnostics"][0]["review"] == review()
    assert "4에서 5로 높이지" in p.gateway.calls[-1][1]


def test_deficiency_rewrites_with_actual_reason_previous_hits_and_queries(tmp_path):
    p = pipeline(tmp_path, [review(True), queries(" revised"), review(), assessment()])
    _, result, qlog, _ = p.research_technology("KIVI", queries()["queries"])
    assert result["status"] == "ok" and len(p.corpus.searches) == 8
    rewrite = p.gateway.calls[1][2]
    assert rewrite["missing_reasons"][0]["missing"] == ["A100 experimental setup section"]
    assert rewrite["missing_reasons"][0]["reason"] == "GPU configuration missing"
    assert rewrite["previous_hits"][0][0]["text"] == chunks()[0]["text"]
    assert len(rewrite["previous_queries"]) == 4
    assert {q["attempt"] for q in qlog} == {0, 1}
    assert len(result["retrieval_diagnostics"]) == 2


def test_two_insufficient_attempts_stop_without_assessment_or_more_rewrites(tmp_path):
    p = pipeline(tmp_path, [review(True), queries(" revised"), review(True)])
    _, result, _, _ = p.research_technology("KIVI", queries()["queries"])
    assert result["status"] == "failed" and len(p.corpus.searches) == 8
    assert len(p.gateway.calls) == 3
    assert all(d["status"] == "insufficient_or_invalid" for d in result["retrieval_diagnostics"])
    assert "A100 experimental setup section" in result["error"]


def test_invalid_assessment_after_single_repair_drives_retrieval_rewrite(tmp_path):
    bad = assessment()
    bad["claims"][0]["references"][0]["chunk_id"] = "fabricated"
    bad_patch = {"patches": [{"target_id": "claims:0:reference:0", "chunk_id": "fabricated", "quote": bad["claims"][0]["references"][0]["quote"]}]}
    p = pipeline(tmp_path, [review(), bad, bad_patch, queries(" revised"), review(), assessment()])
    _, result, _, _ = p.research_technology("KIVI", queries()["queries"])
    assert result["status"] == "ok"
    assert [c[0] for c in p.gateway.calls].count("research_kivi_0_repair_0") == 1
    assert any("supplied same-technology source chunk" in error for error in p.gateway.calls[3][2]["missing_reasons"])


@pytest.mark.parametrize("change", ["duplicate_facet", "unknown_id", "other_tech", "duplicate_id", "missing_reason"])
def test_retrieval_review_rejects_invalid_identifiers_or_coverage(change):
    value = review()
    if change == "duplicate_facet":
        value["items"][-1]["facet"] = "mechanism"
    elif change == "unknown_id":
        value["items"][0]["chunk_ids"] = ["missing"]
    elif change == "other_tech":
        value["items"][0]["chunk_ids"] = ["InfiniGen"]
    elif change == "duplicate_id":
        value["items"][0]["chunk_ids"] *= 2
    else:
        value["items"][0]["reason"] = " "
    assert validate_retrieval_review(value, chunks(), "KIVI")


def test_structured_failure_preserves_error_location_without_rejected_input(tmp_path):
    p = pipeline(tmp_path, [{"queries": "secret-input-marker"}] * 2)
    with pytest.raises(StructuredValidationError) as exc:
        p.structured("bad", Queries, "test", {})
    assert exc.value.errors[0]["loc"] == ["queries"]
    assert "secret-input-marker" not in str(exc.value)
    assert len(p.gateway.calls) == 2


def perspective_assessment(name):
    return {"status": "ok", "claims": [claim(t, f) for t in TECHS for f in sorted(PERSPECTIVE_FACETS[name])],
            "gaps": [f"{name} historical gap"], "conflicts": []}


def state_fixture():
    return {"tech_assessment": {t: assessment(t) for t in TECHS}, **{
        name + "_result": {**perspective_assessment(name), "searched_chunks": chunks()}
        for name in PERSPECTIVE_FACETS}}


def test_all_perspectives_receive_full_immutable_conditions_and_source_chunks(tmp_path):
    p = pipeline(tmp_path, [perspective_assessment(n) for n in PERSPECTIVE_FACETS])
    state = state_fixture()
    before = deepcopy(state)
    for name in PERSPECTIVE_FACETS:
        assert p.perspective(name, state)[name + "_result"]["status"] == "ok"
        payload = p.gateway.calls[-1][2]
        assert payload["tech_assessment"] == state["tech_assessment"]
        assert payload["chunks"] == chunks()
        assert len(payload["chunks"][0]["text"]) > len(QUOTE)
        assert payload["tech_assessment"]["KIVI"]["claims"][0]["conditions"] == before["tech_assessment"]["KIVI"]["claims"][0]["conditions"]
    assert state == before and len(set(p.corpus.web_queries)) == 3


def test_perspective_rejects_missing_required_facet_after_one_repair(tmp_path):
    bad = perspective_assessment("market")
    bad["claims"][0]["facet"] = "mechanism"
    p = pipeline(tmp_path, [bad, bad])
    assert p.perspective("market", state_fixture())["market_result"]["status"] == "failed"
    assert len(p.gateway.calls) == 2


def test_real_join_preserves_original_chunks_conditions_raw_gaps_and_state(tmp_path):
    p, state = pipeline(tmp_path), state_fixture()
    state["market_result"]["searched_chunks"][0]["text"] = QUOTE
    before = deepcopy(state)
    result = p.join(state)
    joined = result["joined"]
    assert result["run_status"] == "joined" and len(joined["claims"]) == 26
    assert state == before
    assert next(c for c in p.all_chunks if c["id"] == "KIVI")["text"] == chunks()[0]["text"]
    assert joined["claims"]["research_kivi-1"]["conditions"] == assessment()["claims"][0]["conditions"]
    assert joined["gaps"] == [f"{n} historical gap" for n in PERSPECTIVE_FACETS]
    assert [g["text"] for g in joined["gap_records"]] == joined["gaps"]
    assert len({g["id"] for g in joined["gap_records"]}) == 3


def report_fixture(tmp_path):
    p = pipeline(tmp_path)
    joined = p.join(state_fixture())["joined"]
    synth = [claim("KIVI", "tradeoff", kind="team_inference"), claim("InfiniGen", "tradeoff", kind="team_inference")]
    extra, _ = collect_evidence({"synthesis": {"claims": synth}}, chunks())
    claims = {**joined["claims"], **extra}
    report = {"summary_claim_ids": ["synthesis-1", "market-1"], "synthesis_claims": synth,
              "sections": [{"title": title, "claim_ids": [cid for cid, c in claims.items() if c["perspective"] in perspectives]}
                           for title, perspectives in [("기술 성숙도", {"research_kivi", "research_infinigen"}),
                                                       ("시장성", {"market"}), ("이해관계자", {"stakeholder"}),
                                                       ("도메인 적용", {"domain"}), ("관점 간 상충과 한계", {"synthesis"})]],
              "gap_decisions": [{"gap_id": g["id"], "status": "unresolved", "resolution": "추가 근거가 필요함", "claim_ids": []}
                                for g in joined["gap_records"]]}
    return p, joined, claims, report


def test_valid_report_allows_summary_reuse_and_passes_identified_gaps_to_synthesis(tmp_path):
    p, joined, claims, report = report_fixture(tmp_path)
    assert report_errors(report, claims, chunks(), joined["gap_records"]) == []
    p.gateway = FakeGateway([{k: v for k, v in report.items() if k != "gap_decisions"},
                             {"gap_decisions": report["gap_decisions"]}])
    result = p.synthesize({"joined": joined})
    assert result["run_status"] == "validated"
    assert p.gateway.calls[1][2]["gap_records"] == joined["gap_records"]
    assert result["joined"]["gaps"] == joined["gaps"]
    assert result["joined"]["gap_records"] == joined["gap_records"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown_gap", "wrong_claim", "empty_explanation", "unknown_only", "synthesis_claim", "empty_claims"])
def test_gap_decisions_reject_missing_duplicate_or_unsubstantiated_resolution(tmp_path, mutation):
    _, joined, claims, report = report_fixture(tmp_path)
    decision = report["gap_decisions"][0]
    decision.update(status="resolved", resolution="기존 평가로 확인됨", claim_ids=["market-1"])
    if mutation == "missing":
        report["gap_decisions"].pop()
    elif mutation == "duplicate":
        report["gap_decisions"].append(deepcopy(decision))
    elif mutation == "unknown_gap":
        decision["gap_id"] = "invented"
    elif mutation == "wrong_claim":
        decision["claim_ids"] = ["invented"]
    elif mutation == "empty_explanation":
        decision["resolution"] = " "
    elif mutation == "unknown_only":
        claims["market-1"]["kind"] = "unknown"
    elif mutation == "synthesis_claim":
        decision["claim_ids"] = ["synthesis-1"]
    else:
        decision["claim_ids"] = []
    assert report_errors(report, claims, chunks(), joined["gap_records"])


def test_resolved_gap_with_existing_nonunknown_claim_is_accepted(tmp_path):
    _, joined, claims, report = report_fixture(tmp_path)
    report["gap_decisions"][0].update(status="resolved", resolution="기존 조건 claim에서 확인", claim_ids=["research_kivi-3"])
    assert report_errors(report, claims, chunks(), joined["gap_records"]) == []


@pytest.mark.parametrize("mutation", ["duplicate_body", "nonsynthesis_final", "wrong_claim_id", "missing_claim", "wrong_section"])
def test_report_rejects_body_repetition_and_wrong_final_or_unknown_claim(tmp_path, mutation):
    _, joined, claims, report = report_fixture(tmp_path)
    if mutation == "missing_claim":
        report["sections"][1]["claim_ids"].remove("market-3")
    elif mutation == "wrong_section":
        report["sections"][1]["claim_ids"], report["sections"][3]["claim_ids"] = report["sections"][3]["claim_ids"], report["sections"][1]["claim_ids"]
    elif mutation == "duplicate_body":
        report["sections"][1]["claim_ids"].append("research_kivi-1")
    elif mutation == "nonsynthesis_final":
        report["sections"][-1]["claim_ids"] = ["market-1"]
    else:
        report["sections"][1]["claim_ids"].append("invented")
    assert report_errors(report, claims, chunks(), joined["gap_records"])


def test_runtime_schema_requires_gap_decisions():
    assert "gap_decisions" in Report.model_json_schema()["required"]


@pytest.mark.parametrize("invalid_query", ["한국어로 지나치게 넓어진 재질문", "x" * 401, ""])
def test_search_query_rejects_language_drift_and_unbounded_rewrite(invalid_query):
    from pydantic import ValidationError
    q = queries()
    q["queries"][0]["query"] = invalid_query
    with pytest.raises(ValidationError):
        Queries.model_validate(q)


def test_synthesis_batches_gaps_and_keeps_final_schema(tmp_path):
    p, joined, claims, report = report_fixture(tmp_path)
    joined["gap_records"] = [{"id": f"g{i}", "perspective": "market", "text": "Missing adoption"} for i in range(11)]
    seen = []
    def generate(purpose, instructions, content, schema):
        payload = json.loads(content)
        if purpose == "synthesis_report":
            assert "gap_records" not in payload
            return json.dumps({k: v for k, v in report.items() if k != "gap_decisions"})
        batch = payload["gap_records"]
        seen.append(len(batch))
        return json.dumps({"gap_decisions": [{"gap_id": g["id"], "status": "unresolved", "resolution": "Evidence absent", "claim_ids": []} for g in batch]})
    p.gateway.generate = generate
    result = p.synthesize({"joined": joined})
    assert result["run_status"] == "validated"
    assert seen == [5, 5, 1]
    assert len(Report.model_validate(result["report"]).gap_decisions) == 11


def test_invalid_gap_batch_blocks_final_report(tmp_path):
    p, joined, _, report = report_fixture(tmp_path)
    p.gateway = FakeGateway([{k: v for k, v in report.items() if k != "gap_decisions"}, {"gap_decisions": []}, {"gap_decisions": []}])
    result = p.synthesize({"joined": joined})
    assert result["run_status"] == "incomplete" and not result["validation_result"]["passed"]


@pytest.mark.parametrize("count", [0, 1, 3, 28])
def test_synthesis_schema_cannot_repeat_all_prior_claims(tmp_path, count):
    from pydantic import ValidationError
    from rag.schemas import ReportDraft
    _, _, _, report = report_fixture(tmp_path)
    draft = {k: v for k, v in report.items() if k != "gap_decisions"}
    draft["synthesis_claims"] = [report["synthesis_claims"][0]] * count
    with pytest.raises(ValidationError): ReportDraft.model_validate(draft)


def test_synthesis_schema_requires_inference_kind(tmp_path):
    from pydantic import ValidationError
    from rag.schemas import ReportDraft
    _, _, _, report = report_fixture(tmp_path)
    draft = {k: v for k, v in report.items() if k != "gap_decisions"}
    draft["synthesis_claims"][0]["kind"] = "source_fact"
    with pytest.raises(ValidationError): ReportDraft.model_validate(draft)


def test_perspective_retries_only_missing_facets_once_and_preserves_branch_trace(tmp_path):
    initial = perspective_assessment('market')
    initial['status'] = 'insufficient'
    initial['claims'][0].update(kind='unknown', references=[])
    missing = initial['claims'][0]['facet']
    p = pipeline(tmp_path, [initial, {'queries': [{'facet': missing, 'query': 'official adoption evidence'}]}, initial])
    seen = []
    p.corpus.web_search = lambda query, top_k, **kwargs: seen.append(kwargs) or []
    value = p.perspective('market', state_fixture())
    assert value['market_result']['status'] == 'insufficient'
    trace = value['market_retrieval']
    assert trace['attempt_count'] == 2 and trace['missing_facets'] == [missing]
    assert trace['remaining_missing_facets'] == [missing]
    assert len(p.gateway.calls) == 3 and seen[-1] == {'expanded_only': True}
    assert 'domain_retrieval' not in value and 'stakeholder_retrieval' not in value


def test_perspective_api_failure_is_failed_not_insufficient(tmp_path):
    from rag.llm import APIError
    p = pipeline(tmp_path)
    def fail(*args):
        raise APIError('mock provider failed')
    p.gateway.generate = fail
    value = p.perspective('domain', state_fixture())
    assert value['domain_result']['status'] == 'failed'
    assert value['domain_retrieval']['attempt_count'] == 1


def test_followup_web_evidence_is_token_bounded_and_inherited_claims_remain(tmp_path):
    initial = perspective_assessment('market')
    initial['status'] = 'insufficient'
    initial['claims'][0].update(kind='unknown', references=[])
    missing = initial['claims'][0]['facet']
    p = pipeline(tmp_path, [initial, {'queries': [{'facet': missing, 'query': 'adoption'}]}, initial])
    p.corpus.evidence_tokens = lambda text: len(text.split())
    p.config.update(perspective_web_token_budget=4, perspective_expanded_token_budget=4)
    p.corpus.web_search = lambda query, top_k, **kwargs: [dict(id=f"web-{bool(kwargs)}-{i}", text='one two three four', source_id='web') for i in range(6)]
    value = p.perspective('market', state_fixture())
    assert value['market_result']['status'] == 'insufficient'
    payload = p.gateway.calls[-1][2]
    assert {c['id'] for c in chunks()} <= {c['id'] for c in payload['chunks']}
    assert sum(len(c['text'].split()) for c in payload['chunks'] if c['source_id'] == 'web') <= 8
    assert len(value['market_result']['claims']) == 6


@pytest.mark.parametrize("name", ["market", "stakeholder", "domain"])
def test_reassessment_omits_only_top_level_search_diagnostics_and_preserves_evidence(tmp_path, name):
    initial = perspective_assessment(name)
    initial["status"] = "insufficient"
    initial["claims"][0].update(kind="unknown", references=[])
    missing = initial["claims"][0]["facet"]
    p = pipeline(tmp_path, [initial, {"queries": [{"facet": missing, "query": "official evidence"}]}, initial])
    p.corpus.web_search = lambda query, top_k, **kwargs: []
    state = state_fixture()
    for assessment in state["tech_assessment"].values():
        assessment["retrieval_diagnostics"] = [{"queries": ["duplicate trace"], "review": {"text": "x" * 30000}}]
        assessment["claims"][0]["conditions"] = {"accuracy": {"lengths": [128, 1920]}, "retrieval_diagnostics": "substantive nested condition must survive"}
        assessment["claims"][0]["caveats"] = {"limitations": ["no enterprise evidence"]}
        assessment["conflicts"] = ["different experiments"]
        assessment["gaps"] = ["unverified deployment"]
        assessment["future_substantive_field"] = {"preserve": [1, 2, 3]}
    before = deepcopy(state)
    value = p.perspective(name, state)
    first, rewritten, reassessed = p.gateway.calls
    assert [call[0] for call in p.gateway.calls] == [name, name + "_rewrite", name + "_reassessment"]
    assert first[2]["tech_assessment"] == before["tech_assessment"]
    expected = {tech: {key: item for key, item in assessment.items() if key != "retrieval_diagnostics"}
                for tech, assessment in before["tech_assessment"].items()}
    assert reassessed[2]["tech_assessment"] == expected
    assert reassessed[2]["previous_assessment"] == initial
    assert reassessed[2]["missing_facets"] == [missing]
    assert reassessed[2]["chunks"] == first[2]["chunks"] == chunks()
    assert reassessed[2]["source_metadata"] == first[2]["source_metadata"]
    assert value[name + "_result"]["claims"] == initial["claims"]
    assert value[name + "_retrieval"]["attempt_count"] == 2
    assert value[name + "_retrieval"]["remaining_missing_facets"] == [missing]
    assert state == before
    old_payload = {**reassessed[2], "tech_assessment": before["tech_assessment"]}
    assert len(json.dumps(reassessed[2]).encode()) < len(json.dumps(old_payload).encode())


def test_reassessment_restores_protected_quote_without_regenerating_it(tmp_path):
    from rag.evidence import validate_perspective
    from rag.schemas import Assessment
    valid = perspective_assessment("stakeholder")
    bad = deepcopy(valid)
    bad["claims"][0]["references"][0]["quote"] = "This altered quotation is absent from the source."
    content = {"previous_assessment": deepcopy(valid), "missing_facets": ["user"],
               "chunks": chunks(), "tech_assessment": state_fixture()["tech_assessment"],
               "source_metadata": {"paper": {"version": "fixed"}}}
    before = deepcopy(content)
    p = pipeline(tmp_path, [bad])
    seen = []
    def check(value):
        seen.append(deepcopy(content["previous_assessment"]))
        return validate_perspective(value, content["chunks"], "stakeholder")
    assert p.structured("stakeholder_reassessment", Assessment, "repair", content, check) == valid
    assert len(p.gateway.calls) == 1
    assert p.gateway.calls[0][2] == before
    assert seen == [before["previous_assessment"]]
    assert content == before


@pytest.mark.parametrize("case", ["mixed", "preserve", "schema", "other_purpose", "no_previous"])
def test_non_quote_only_repair_keeps_original_task(tmp_path, case):
    from rag.schemas import Assessment
    valid = perspective_assessment("stakeholder")
    first = deepcopy(valid)
    content = {"previous_assessment": deepcopy(valid), "chunks": chunks(),
               "tech_assessment": state_fixture()["tech_assessment"], "missing_facets": ["user"]}
    if case == "no_previous":
        content.pop("previous_assessment")
    if case == "schema":
        first.pop("status")
    before = deepcopy(content)
    p = pipeline(tmp_path, [first, valid])
    calls = []
    quote_error = "claim 0: quotation does not exist in supplied chunk KIVI"
    def check(value):
        calls.append(value)
        if len(calls) > 1 or case == "schema":
            return []
        if case == "mixed":
            return [quote_error, "Preserve sufficient facets verbatim; reassess missing facets only"]
        if case == "preserve":
            return ["Preserve sufficient facets verbatim; reassess missing facets only"]
        return [quote_error]
    purpose = "stakeholder" if case == "other_purpose" else "stakeholder_reassessment"
    assert p.structured(purpose, Assessment, "repair", content, check) == valid
    assert len(p.gateway.calls) == 2
    assert p.gateway.calls[0][2] == before
    expected_task = deepcopy(before)
    assert p.gateway.calls[1][2]["task"] == expected_task
    assert content == before


def test_mixed_quote_and_preservation_repair_keeps_original_sufficient_claims(tmp_path):
    from rag.evidence import validate_perspective
    from rag.schemas import Assessment
    previous = perspective_assessment("market")
    previous["status"] = "insufficient"
    for c in previous["claims"]:
        if c["facet"] == "adoption":
            c.update(kind="unknown", references=[])
    preserved = [c for c in previous["claims"] if c["facet"] != "adoption"]
    bad = deepcopy(previous)
    next(c for c in bad["claims"] if c["facet"] == "alternatives")["text"] = "Must restore original sufficient claim"
    next(c for c in bad["claims"] if c["facet"] == "adoption")["references"] = [{"chunk_id": "KIVI", "quote": "This quotation is absent from the supplied source."}]
    content = {"previous_assessment": previous, "missing_facets": ["adoption"], "chunks": chunks(),
               "tech_assessment": state_fixture()["tech_assessment"], "source_metadata": {"version": "fixed"}}
    before = deepcopy(content)
    p = pipeline(tmp_path, [bad, {"patches": [{"target_id": "claims:0:reference:0", "quote": QUOTE}]}])
    def check(value):
        errors = validate_perspective(value, content["chunks"], "market")
        if [c for c in value["claims"] if c["facet"] != "adoption"] != preserved:
            errors.append("Preserve sufficient facets verbatim; reassess missing facets only")
        return errors
    repaired = p.structured("market_reassessment", Assessment, "repair", content, check)
    assert [c for c in repaired["claims"] if c["facet"] != "adoption"] == preserved
    assert repaired["claims"][0]["references"] == [{"chunk_id": "KIVI", "quote": QUOTE}]
    assert len(p.gateway.calls) == 2 and p.gateway.calls[0][2] == before
    repair = p.gateway.calls[1][2]
    assert repair["target_id"] == "claims:0:reference:0"
    assert repair["chunks"] == [chunks()[0]]
    assert "previous_answer" not in repair and "previous_assessment" not in repair
    assert len(repaired["claims"]) == 6
    assert content == before


def test_research_completion_order_preserves_values_and_serialized_technology_order(tmp_path, monkeypatch):
    import rag.graph as graph
    plan = {"queries": [{"technology": t, "facet": f, "query": f} for t in TECHS for f in FACETS]}
    outputs = []
    for reverse in [False, True]:
        p = pipeline(tmp_path / str(reverse))
        p.structured = lambda *args, **kwargs: deepcopy(plan)
        p.research_technology = lambda tech, queries: (tech, assessment(tech), [{"attempt": 0}], {})
        monkeypatch.setattr(graph, "as_completed", lambda futures, reverse=reverse: iter(reversed(futures)) if reverse else iter(futures))
        result = p.research({})
        assert list(result["tech_assessment"]) == list(TECHS)
        assert result["tech_assessment"] == {t: assessment(t) for t in TECHS}
        assert list(result["research_retrieval"]) == list(TECHS)
        outputs.append(json.dumps(result["tech_assessment"], ensure_ascii=False))
    assert outputs[0] == outputs[1]
