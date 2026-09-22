"""Conflict provenance is deterministic; no model or network calls."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from rag.conflicts import build_conflict_records, conflict_record_errors, conflicts_for_joined
from rag.render import render_report
from test_render import report_fixture
from test_pipeline_improvements import pipeline, state_fixture


def inputs():
    claims = {
        'market-1': {'perspective': 'market', 'text': '사용자 비용 절감', 'kind': 'team_inference',
                     'conditions': '장문 조건', 'caveats': '업무 미검증', 'evidence_ids': ['E-1']},
        'stakeholder-1': {'perspective': 'stakeholder', 'text': '운영자 부담', 'kind': 'source_fact',
                          'conditions': 'CPU 전송', 'caveats': '', 'evidence_ids': ['E-2']},
    }
    assessments = {'market': {'conflicts': ['market-1 이익과 stakeholder-1 부담이 공존한다.']},
                   'stakeholder': {'conflicts': ['원문 그대로 남긴 상충']}}
    return assessments, claims


def test_candidates_are_not_semantic_verification_and_source_is_preserved():
    assessments, claims = inputs()
    before = deepcopy((assessments, claims))
    records = build_conflict_records(assessments, claims)
    assert (assessments, claims) == before
    assert records[0]['explicit_claim_ids'] == ['market-1', 'stakeholder-1']
    assert records[0]['candidate_claim_ids'] == ['market-1', 'stakeholder-1']
    assert records[1]['candidate_claim_ids'] == ['stakeholder-1']
    assert all(r['verified_claim_ids'] == [] and r['status'] == 'unresolved'
               and r['type'] == 'unclassified' for r in records)
    assert conflict_record_errors(records, claims, assessments) == []


@pytest.mark.parametrize('mutation', ['duplicate_record', 'missing_record', 'unknown_id', 'changed_text',
                                    'bad_link', 'duplicate_link', 'altered_conditions', 'altered_evidence',
                                    'resolved_without_review', 'verified_without_review'])
def test_invalid_records_are_rejected(mutation):
    assessments, claims = inputs()
    records = build_conflict_records(assessments, claims)
    r = records[0]
    if mutation == 'duplicate_record': records.append(deepcopy(r))
    elif mutation == 'missing_record': records.pop()
    elif mutation == 'unknown_id': r['id'] = 'conflict-madeup-1'
    elif mutation == 'changed_text': r['text'] = '새 원문'
    elif mutation == 'bad_link': r['verified_claim_ids'] = ['no-such-claim']
    elif mutation == 'duplicate_link': r['candidate_claim_ids'].append('market-1')
    elif mutation == 'altered_conditions': r['conditions']['market-1'] = '새 조건'
    elif mutation == 'altered_evidence': r['evidence_ids']['market-1'] = ['E-fabricated']
    elif mutation == 'resolved_without_review': r['status'] = 'resolved'
    elif mutation == 'verified_without_review': r['verified_claim_ids'] = ['market-1']
    assert conflict_record_errors(records, claims, assessments)


def test_resolved_requires_reviewed_sourced_claims_beyond_unknown():
    assessments, claims = inputs()
    records = build_conflict_records(assessments, claims)
    records[0].update(status='resolved', type='condition_difference',
                      verified_claim_ids=['market-1', 'stakeholder-1'],
                      reviewed_by='검수자', rationale='두 조건과 인용을 대조하여 적용 범위를 구분함')
    assert conflict_record_errors(records, claims, assessments) == []
    claims['stakeholder-1']['kind'] = 'unknown'
    assert conflict_record_errors(records, claims, assessments)


def test_human_can_link_other_perspective_only_with_preserved_context():
    assessments, claims = inputs()
    records = build_conflict_records(assessments, claims)
    record = records[1]
    record.update(verified_claim_ids=['market-1', 'stakeholder-1'],
                  reviewed_by='검수자', rationale='두 관점의 원문과 조건을 확인함')
    assert conflict_record_errors(records, claims, assessments)
    record['conditions']['market-1'] = claims['market-1']['conditions']
    record['evidence_ids']['market-1'] = claims['market-1']['evidence_ids'][:]
    assert conflict_record_errors(records, claims, assessments) == []


def test_equal_text_in_two_perspectives_keeps_both_provenances():
    assessments, claims = inputs()
    assessments['stakeholder']['conflicts'] = assessments['market']['conflicts'][:]
    records = build_conflict_records(assessments, claims)
    assert records[0]['text'] == records[1]['text']
    assert records[0]['id'] != records[1]['id']
    assert conflict_record_errors(records, claims, assessments) == []


def test_join_keeps_original_claims_conflicts_and_adds_records_without_calls(tmp_path):
    p, state = pipeline(tmp_path), state_fixture()
    state['market_result']['conflicts'] = ['원래 시장성 상충']
    before = deepcopy(state)
    joined = p.join(state)['joined']
    assert state == before and p.gateway.calls == []
    assert joined['conflicts'] == ['원래 시장성 상충']
    assert joined['conflict_records'][0]['perspective'] == 'market'
    assert len(joined['conflict_records'][0]['candidate_claim_ids']) == 6


def test_render_legacy_conflicts_and_new_records_are_linked(tmp_path, report_fixture):
    report, joined, sources, settings, config = report_fixture
    joined['conflicts'] = ['기존 실행의 상충 원문']
    joined['assessments'] = {}
    before = deepcopy(joined)
    paths = render_report(tmp_path, report, joined, sources, settings, config)
    assert joined == before
    review = Path(paths['conflict_review']).read_text()
    assert '기존 실행의 상충 원문' in review and '**해소 상태:** unresolved' in review
    assert 'conflict_review.md' in Path(paths['markdown']).read_text()
    records = json.loads((tmp_path / 'conflict_records.json').read_text())
    assert records[0]['candidate_claim_ids'] == []
    joined['conflict_records'] = records
    assert conflicts_for_joined(joined) == records
    joined['conflict_records'][0]['status'] = 'resolved'
    with pytest.raises(ValueError, match='Conflict review validation failed'):
        render_report(tmp_path, report, joined, sources, settings, config)
