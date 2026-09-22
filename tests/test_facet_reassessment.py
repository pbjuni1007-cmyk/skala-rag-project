from copy import deepcopy
import json

import pytest

from rag.evidence import validate_perspective
from rag.graph import BASE
from rag.reassessment import reassess_facets
from rag.request_budget import InputBudgetExceeded
from test_pipeline_improvements import pipeline, perspective_assessment, state_fixture, chunks


def setup(tmp_path, missing=('costs', 'adoption')):
    p = pipeline(tmp_path)
    original = perspective_assessment('market')
    content = {'previous_assessment': original, 'missing_facets': list(missing), 'chunks': chunks(),
               'tech_assessment': state_fixture()['tech_assessment'], 'source_metadata': {'frozen': True}}
    units = [{**deepcopy(original), 'claims': [c for c in deepcopy(original['claims']) if c['facet'] == f],
              'conflicts': ['new conflict ' + f], 'gaps': ['new gap ' + f]} for f in missing]
    p.gateway.responses = iter(units)
    generated = p.gateway.generate
    def generate(purpose, instructions, body, schema):
        data = json.loads(body)
        if '_facet_' in purpose:
            p.gateway.calls.append((purpose, instructions, data, schema))
            facet = data.get('task', data)['missing_facets'][0]
            return json.dumps(units[list(missing).index(facet)])
        return generated(purpose, instructions, body, schema)
    p.gateway.generate = generate
    seen = []
    def preflight(purpose, instructions, body, schema, reserve_input=0):
        seen.append((purpose, json.loads(body), len(p.gateway.calls)))
        if purpose == 'market_reassessment':
            raise InputBudgetExceeded(purpose, 24673, 24000, reserve_input)
    p.gateway.preflight = preflight
    def validate(value):
        errors = validate_perspective(value, content['chunks'], 'market')
        if [c for c in value['claims'] if c['facet'] not in missing] != [c for c in original['claims'] if c['facet'] not in missing]:
            errors.append('protected claims changed')
        return errors
    return p, content, units, seen, validate


def test_split_preflights_all_facets_preserves_sources_and_merges_six_claims(tmp_path):
    p, content, _, seen, check = setup(tmp_path)
    before = deepcopy(content)
    result = reassess_facets(p, 'market_reassessment', 'full assessment', content, check, BASE)
    assert len(result['claims']) == 6 and not check(result)
    assert content == before
    assert result['gaps'] == content['previous_assessment']['gaps'] + ['new gap costs', 'new gap adoption']
    assert result['conflicts'] == content['previous_assessment']['conflicts'] + ['new conflict costs', 'new conflict adoption']
    assert [row[2] for row in seen[:3]] == [0, 0, 0]
    for purpose, _, payload, _ in p.gateway.calls:
        assert payload['chunks'] == content['chunks']
        assert payload['tech_assessment'] == content['tech_assessment']
        assert payload['source_metadata'] == content['source_metadata']
        assert len(payload['previous_assessment']['claims']) == 2
        assert len(payload['missing_facets']) == 1
    saved = json.loads((tmp_path/'reassessments/market_reassessment-split.json').read_text())
    assert saved['status'] == 'validated' and len(saved['completed_facets']) == 2


def test_last_facet_overflow_blocks_every_generation(tmp_path):
    p, content, _, _, check = setup(tmp_path)
    def preflight(purpose, *args, **kwargs):
        if purpose in {'market_reassessment', 'market_reassessment_facet_adoption'}:
            raise InputBudgetExceeded(purpose, 25000, 24000)
    p.gateway.preflight = preflight
    with pytest.raises(InputBudgetExceeded):
        reassess_facets(p, 'market_reassessment', '', content, check, BASE)
    assert p.gateway.calls == []
    assert json.loads((tmp_path/'reassessments/market_reassessment-split.json').read_text())['status'] == 'failed'


def test_unit_failure_saves_previous_success_and_original(tmp_path):
    p, content, units, _, check = setup(tmp_path)
    generate = p.gateway.generate
    def fail(purpose, *args):
        if purpose.endswith('adoption'):
            raise StopIteration()
        return generate(purpose, *args)
    p.gateway.generate = fail
    with pytest.raises(StopIteration):
        reassess_facets(p, 'market_reassessment', '', content, check, BASE)
    saved = json.loads((tmp_path/'reassessments/market_reassessment-split.json').read_text())
    assert saved['status'] == 'failed'
    assert list(saved['completed_facets']) == ['costs']
    assert saved['original_assessment'] == content['previous_assessment']


def test_wrong_facet_identity_is_rejected_even_with_valid_quotes(tmp_path):
    p, content, units, _, check = setup(tmp_path, ('costs',))
    bad = deepcopy(units[0]); bad['claims'][1]['technology'] = 'KIVI'
    units[0].update(bad)
    with pytest.raises(ValueError):
        reassess_facets(p, 'market_reassessment', '', content, check, BASE)
    assert len(p.gateway.calls) == 2  # Initial unit plus the existing bounded repair.


def test_merged_result_must_pass_full_validator(tmp_path):
    p, content, _, _, _ = setup(tmp_path)
    with pytest.raises(ValueError, match='Merged facet reassessment'):
        reassess_facets(p, 'market_reassessment', '', content, lambda r: ['full validation failed'], BASE)


def test_under_limit_keeps_single_full_request(tmp_path):
    p, content, _, _, check = setup(tmp_path)
    p.gateway.preflight = lambda *args, **kwargs: None
    p.gateway.responses = iter([deepcopy(content['previous_assessment'])])
    assert reassess_facets(p, 'market_reassessment', '', content, check, BASE) == content['previous_assessment']
    assert [c[0] for c in p.gateway.calls] == ['market_reassessment']


def test_split_supports_at_most_three_distinct_facets(tmp_path):
    p, content, _, _, check = setup(tmp_path, ('costs', 'costs', 'adoption', 'alternatives'))
    with pytest.raises(ValueError, match='1..3 distinct'):
        reassess_facets(p, 'market_reassessment', '', content, check, BASE)
    assert not p.gateway.calls
