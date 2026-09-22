from copy import deepcopy
import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from rag.request_budget import InputBudgetExceeded, enforce_input_budget, deduplicate_chunks
from rag.repair import plan_repairs, apply_patch, restore_sufficient
from rag.schemas import Assessment, Queries, ReportDraft
from rag.evidence import validate_perspective
from rag.graph import StructuredValidationError
from rag.llm import Gateway
from test_pipeline_improvements import pipeline, perspective_assessment, chunks, QUOTE
from test_gateway import settings, completed, entries, no_network_or_retry_wait


def test_budget_margin_and_headroom_are_derived_without_guessing():
    assert enforce_input_budget('patch', 23488, 24000) == 23488
    assert enforce_input_budget('initial', 22976, 24000, 512) == 22976
    with pytest.raises(InputBudgetExceeded) as exc:
        enforce_input_budget('old-market-repair', 23757, 24000)
    assert exc.value.diagnostic['available_input_tokens'] == 23488
    assert exc.value.diagnostic['generation_sent'] is False
    with pytest.raises(InputBudgetExceeded):
        enforce_input_budget('initial', 22977, 24000, 512)
    with pytest.raises(ValueError):
        enforce_input_budget('invalid', True, 24000)


def test_chunk_dedup_is_lossless_and_rejects_conflicting_ids():
    first = chunks()[0]
    content = {'chunks': [first, deepcopy(first), {**first, 'id': 'different-reference-id'}], 'conditions': {'keep': True}}
    before = deepcopy(content)
    result = deduplicate_chunks(content)
    assert len(result['chunks']) == 2 and result['chunks'][0] == first
    assert result['conditions'] == before['conditions'] and content == before
    result['chunks'][0]['text'] = 'local mutation'
    assert content == before
    with pytest.raises(ValueError, match='Conflicting'):
        deduplicate_chunks({'chunks': [first, {**first, 'text': 'not the original full text'}]})


def make_bad():
    value = perspective_assessment('market')
    value['claims'][0]['references'][0]['quote'] = 'This nonexistent quotation must not pass.'
    return value


@pytest.mark.parametrize('patch', [
    {'patches': []},
    {'patches': [{'target_id': 'invented', 'quote': QUOTE}]},
    {'patches': [{'target_id': 'claims:0:reference:0', 'quote': QUOTE}] * 2},
    {'patches': [{'target_id': 'claims:0:reference:0', 'quote': QUOTE, 'chunk_id': 'other'}]},
])
def test_invalid_missing_duplicate_or_extra_patch_fields_rejected(patch):
    bad = make_bad()
    units = plan_repairs(bad, validate_perspective(bad, chunks(), 'market'), {'chunks': chunks()}, chunks())
    with pytest.raises((ValueError, ValidationError)):
        apply_patch(bad, units[0], patch)


def test_claim_patch_rejects_unrequested_changes_and_keeps_nested_evidence():
    bad = perspective_assessment('market')
    bad['claims'][0].update(kind='author_reported_result', conditions='')
    unit = plan_repairs(bad, validate_perspective(bad, chunks(), 'market'), {'chunks': chunks()}, chunks())[0]
    fixed = deepcopy(bad['claims'][0]); fixed['conditions'] = 'the original experiment conditions'
    malicious = {**fixed, 'technology': 'InfiniGen'}
    with pytest.raises(ValueError, match='immutable'):
        apply_patch(bad, unit, {'patches': [{'target_id': unit.target_id, 'claim': malicious}]})
    output = apply_patch(bad, unit, {'patches': [{'target_id': unit.target_id, 'claim': fixed}]})
    assert output['claims'][0]['references'] == bad['claims'][0]['references']
    assert output['claims'][1:] == bad['claims'][1:]
    assert bad['claims'][0]['conditions'] == ''


def test_valid_patch_schema_still_runs_original_full_validator(tmp_path):
    bad = make_bad()
    p = pipeline(tmp_path, [bad, {'patches': [{'target_id': 'claims:0:reference:0', 'quote': 'Still absent from the source chunk.'}]}])
    with pytest.raises(StructuredValidationError, match='quotation'):
        p.structured('market', Assessment, 'task', {'chunks': chunks()}, lambda v: validate_perspective(v, chunks(), 'market'))
    assert len(p.gateway.calls) == 2


def test_two_citations_split_into_single_attempt_units_and_merge_all(tmp_path):
    bad = make_bad()
    bad['claims'][1]['references'][0]['quote'] = 'Another invalid source quotation.'
    patches = [{'patches': [{'target_id': f'claims:{i}:reference:0', 'quote': QUOTE}]} for i in (0, 1)]
    p = pipeline(tmp_path, [bad, *patches])
    preflights = []
    p.gateway.preflight = lambda purpose, *args, **kwargs: preflights.append(purpose)
    output = p.structured('market', Assessment, 'task', {'chunks': chunks()}, lambda v: validate_perspective(v, chunks(), 'market'))
    assert output == perspective_assessment('market')
    assert preflights == ['market', 'market_repair_0', 'market_repair_1']
    assert len(p.gateway.calls) == 3
    assert all(len(c[2]['chunks']) == 1 for c in p.gateway.calls[1:])
    assert all(c[2]['chunks'][0]['text'] == chunks()[0]['text'] for c in p.gateway.calls[1:])


def test_all_units_checked_before_generation_and_indivisible_oversize_stops(tmp_path):
    bad = make_bad(); bad['claims'][1]['references'][0]['quote'] = 'Second invalid quotation.'
    p = pipeline(tmp_path, [bad])
    def preflight(purpose, *args, **kwargs):
        if purpose.endswith('_repair_1'):
            raise InputBudgetExceeded(purpose, 30000, 24000)
    p.gateway.preflight = preflight
    with pytest.raises(InputBudgetExceeded):
        p.structured('market', Assessment, 'task', {'chunks': chunks()}, lambda v: validate_perspective(v, chunks(), 'market'))
    assert len(p.gateway.calls) == 1


def test_unsupported_schema_oversize_is_not_generically_split(tmp_path):
    p = pipeline(tmp_path, [{'queries': 'wrong schema'}])
    def preflight(purpose, *args, **kwargs):
        if purpose.endswith('_repair'):
            raise InputBudgetExceeded(purpose, 30000, 24000)
    p.gateway.preflight = preflight
    with pytest.raises(InputBudgetExceeded):
        p.structured('queries', Queries, 'task', {'original': 'kept'})
    assert len(p.gateway.calls) == 1


def test_repair_part_limit_stops_without_extra_generation(tmp_path):
    bad = make_bad()
    bad['claims'][0]['references'] = [{'chunk_id': 'KIVI', 'quote': f'Invalid quotation number {i}.'} for i in range(9)]
    p = pipeline(tmp_path, [bad])
    with pytest.raises(InputBudgetExceeded, match='repair_parts_limit_8'):
        p.structured('market', Assessment, 'task', {'chunks': chunks()}, lambda v: validate_perspective(v, chunks(), 'market'))
    assert len(p.gateway.calls) == 1


def test_layout_patch_preserves_synthesis_claims_and_avoids_full_answer_resend(tmp_path):
    from test_pipeline_improvements import report_fixture
    p, joined, claims, report = report_fixture(tmp_path)
    valid = {k:v for k,v in report.items() if k != 'gap_decisions'}
    bad = deepcopy(valid);bad['sections'].reverse()
    from test_pipeline_improvements import FakeGateway
    p.gateway = FakeGateway([bad, {k:valid[k] for k in ('summary_claim_ids', 'sections')}])
    def check(value):
        return [] if value['sections'] == valid['sections'] else ['Report sections must follow the agreed order']
    output = p.structured('synthesis_report', ReportDraft, 'layout', {'claims': list(claims.values()), 'conflicts': []}, check)
    assert output == valid
    payload = p.gateway.calls[1][2]
    assert 'previous_answer' not in payload and 'claim_metadata' in payload
    assert all('text' not in c and 'conditions' not in c for c in payload['claim_metadata'])


def test_every_new_generation_exact_counted_and_planning_count_reused(settings, tmp_path, monkeypatch):
    gateway = Gateway(settings, 'guard', tmp_path/'out')
    calls = []
    def request(path, payload):
        calls.append(path)
        return {'input_tokens': 100} if path == 'responses/input_tokens' else completed()
    monkeypatch.setattr(gateway, 'request', request)
    gateway.preflight('x', 'instructions', 'short', reserve_input=512)
    gateway.generate('x', 'instructions', 'short')
    assert calls == ['responses/input_tokens', 'responses']
    record = json.loads(next((tmp_path/'out/input_budget').glob('*.json')).read_text())
    assert record['status'] == 'passed' and record['repair_reserve'] == 512
    assert record['available_input_tokens'] == 22976


def test_new_request_oversize_count_has_diagnostic_and_no_paid_reservation(settings, tmp_path, monkeypatch):
    gateway = Gateway(settings, 'guard', tmp_path/'out')
    calls = []
    def request(path, payload):
        calls.append(path);return {'input_tokens': 23757}
    monkeypatch.setattr(gateway, 'request', request)
    with pytest.raises(InputBudgetExceeded):
        gateway.generate('market_repair', 'instructions', 'input')
    assert calls == ['responses/input_tokens'] and entries(settings) == []
    record = json.loads(next((tmp_path/'out/input_budget').glob('*.json')).read_text())
    assert record['input_tokens'] == 23757 and record['available_input_tokens'] == 23488


def test_exact_completed_cache_bypasses_even_large_headroom(settings, tmp_path, monkeypatch):
    first = Gateway(settings, 'first', tmp_path/'first')
    monkeypatch.setattr(first, 'request', lambda path,payload: {'input_tokens': 100} if path == 'responses/input_tokens' else completed())
    first.generate('x','instructions','input')
    resumed = Gateway(settings,'resume',tmp_path/'resume',cache_dir=tmp_path/'first')
    monkeypatch.setattr(resumed, 'request', lambda *args: pytest.fail('cache reuse must not call provider'))
    assert resumed.preflight('x','instructions','input',reserve_input=100000)['cached']
    assert resumed.generate('x','instructions','input') == '검증된 응답'


def test_actual_market_saved_failure_replayed_as_one_literal_quote_patch(tmp_path):
    fixture = json.loads((Path(__file__).parent/'fixtures/market_guardrail_replay.json').read_text())
    content = {k:deepcopy(fixture[k]) for k in ('previous_assessment','missing_facets','chunks','tech_assessment','source_metadata')}
    before = deepcopy(content)
    preserved = [c for c in content['previous_assessment']['claims'] if c['facet'] not in content['missing_facets']]
    def check(value):
        errors = validate_perspective(value, content['chunks'], 'market')
        if [c for c in value['claims'] if c['facet'] not in content['missing_facets']] != preserved:
            errors.append('Preserve sufficient facets verbatim; reassess missing facets only')
        return errors
    assert check(fixture['candidate']) == fixture['original_errors']
    assert fixture['legacy_exact_input_count'] > 24000 - 512
    p = pipeline(tmp_path, [fixture['candidate'], {'patches': [{'target_id': 'claims:3:reference:2', 'quote': fixture['literal_quote']}]}])
    output = p.structured('market_reassessment', Assessment, 'original task', content, check)
    assert check(output) == [] and len(output['claims']) == 6
    assert [c for c in output['claims'] if c['facet'] not in content['missing_facets']] == preserved
    assert content == before and len(p.gateway.calls) == 2
    payload = p.gateway.calls[1][2]
    assert payload['chunks'][0] == next(c for c in content['chunks'] if c['id']=='infinigen_repo:s8:c0')
    assert len(json.dumps(payload,ensure_ascii=False).encode()) < 16000
    # This is size evidence only, never a claimed provider token count.
