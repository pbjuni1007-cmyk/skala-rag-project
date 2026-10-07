from copy import deepcopy
import json
from pathlib import Path
import pytest
from rag.evidence import validate_assessment, validate_perspective
from rag.graph import StructuredValidationError
from rag.repair import plan_repairs, apply_patch, source_verbatim_quote
from rag.schemas import Assessment
from test_pipeline_improvements import perspective_assessment, chunks, pipeline


def fixture():
    value = perspective_assessment('domain')
    value['claims'][3]['references'].append({'chunk_id': 'infinigen:missing', 'quote': 'A nonexistent source reference.'})
    errors = validate_perspective(value, chunks(), 'domain')
    unit, = plan_repairs(value, errors, {}, chunks())
    return value, unit


def test_unknown_reference_gets_typed_target_and_only_relevant_whole_chunks():
    value, unit = fixture()
    assert unit.kind == 'reference' and unit.reference_index == 1
    assert unit.content['chunks'] == [chunks()[1]]
    assert unit.content['claim'] == value['claims'][3]
    assert unit.schema.model_json_schema()['$defs']['ReferencePatch']['additionalProperties'] is False


def test_replacement_preserves_every_other_field_and_original_input():
    value, unit = fixture(); before = deepcopy(value)
    fixed = apply_patch(value, unit, {'patches': [{'target_id': unit.target_id,
        'chunk_id': 'InfiniGen', 'quote': chunks()[1]['text']}]})
    expected = deepcopy(value)
    expected['claims'][3]['references'][1] = {'chunk_id': 'InfiniGen', 'quote': chunks()[1]['text']}
    assert fixed == expected and value == before
    assert not validate_perspective(fixed, chunks(), 'domain')


@pytest.mark.parametrize('change', [
    {'chunk_id': 'invented'}, {'chunk_id': 'KIVI'}, {'quote': 'short'},
    {'quote': 'This quotation never appeared in the source.'}, {'target_id': 'claims:0:reference:0'},
    {'text': 'Change the claim too'},
])
def test_bad_reference_patches_are_rejected(change):
    value, unit = fixture()
    item = {'target_id': unit.target_id, 'chunk_id': 'InfiniGen', 'quote': chunks()[1]['text'], **change}
    with pytest.raises(ValueError):
        apply_patch(value, unit, {'patches': [item]})


def test_duplicate_replacement_targets_are_rejected():
    value, unit = fixture()
    item = {'target_id': unit.target_id, 'chunk_id': 'InfiniGen', 'quote': chunks()[1]['text']}
    with pytest.raises(ValueError):
        apply_patch(value, unit, {'patches': [item, item]})


def test_existing_chunk_quote_error_keeps_fast_citation_path():
    value = perspective_assessment('domain')
    value['claims'][3]['references'][0]['quote'] = 'Quotation absent from existing chunk.'
    unit, = plan_repairs(value, validate_perspective(value, chunks(), 'domain'), {}, chunks())
    assert unit.kind == 'citation' and unit.content['chunks'] == [chunks()[1]]
    assert 'chunk_id' not in unit.schema.model_json_schema()['$defs']['CitationPatch']['properties']


def test_structured_repair_runs_merged_original_validator(tmp_path):
    from rag.schemas import Assessment
    value, unit = fixture()
    patch = {'patches': [{'target_id': unit.target_id, 'chunk_id': 'InfiniGen', 'quote': chunks()[1]['text']}]}
    p = pipeline(tmp_path, [value, patch]); seen=[]
    def validate(v):
        seen.append(deepcopy(v))
        return validate_perspective(v, chunks(), 'domain')
    fixed = p.structured('domain', Assessment, 'test', {'chunks': chunks()}, validate)
    assert len(seen) == 2 and len(fixed['claims']) == 6
    assert not validate_perspective(fixed, chunks(), 'domain')
    assert len(p.gateway.calls) == 2


def test_unique_pdf_wrap_restores_actual_source_not_rewritten_quote():
    from rag.repair import source_verbatim_quote
    source = 'Before. This preserves pre- serving model accuracy. After.'
    quote = 'This preserves preserving model accuracy.'
    assert source_verbatim_quote(quote, source) == 'This preserves pre- serving model accuracy.'


def test_pdf_wrap_alignment_rejects_ambiguity_and_substantive_changes():
    from rag.repair import source_verbatim_quote
    quote = 'The accuracy is preserved at 20 percent.'
    source = 'The accu- racy is preserved at 20 percent.'
    assert source_verbatim_quote(quote, source + ' ' + source) == quote
    assert source_verbatim_quote(quote, source.replace('20', '30')) == quote
    assert source_verbatim_quote(quote, source.replace('preserved', 'lost')) == quote


def test_reference_patch_saves_verbatim_pdf_wrap_and_passes_original_validator():
    value, unit = fixture()
    unit.content['chunks'][0]['text'] = 'The method preserves model accu- racy under the tested conditions.'
    patched = apply_patch(value, unit, {'patches': [{'target_id': unit.target_id,
        'chunk_id': 'InfiniGen', 'quote': 'The method preserves model accuracy under the tested conditions.'}]})
    assert patched['claims'][3]['references'][1]['quote'] == unit.content['chunks'][0]['text']


def test_missing_space_after_pdf_wrap_hyphen_restores_source():
    from rag.repair import source_verbatim_quote
    source = 'The text compares Llama- based models and quanti- zation.'
    assert source_verbatim_quote('The text compares Llama-based models and quanti-zation.', source) == source


def test_typography_restoration_preserves_all_other_data_and_unknown_ids():
    from rag.repair import restore_verbatim_references
    value = {'claims': [{'text': 'Unchanged', 'conditions': '20 percent', 'references': [
        {'chunk_id': 'a', 'quote': 'Llama-based models preserve accuracy.'},
        {'chunk_id': 'missing', 'quote': 'Llama-based models preserve accuracy.'}]}]}
    before = deepcopy(value)
    fixed = restore_verbatim_references(value, [{'id': 'a', 'text': 'Llama- based models preserve accuracy.'}])
    expected = deepcopy(value)
    expected['claims'][0]['references'][0]['quote'] = 'Llama- based models preserve accuracy.'
    assert fixed == expected and value == before


def test_typography_restoration_avoids_paid_correction_but_runs_validator(tmp_path):
    from rag.schemas import Assessment
    value = perspective_assessment('domain')
    source_chunks = chunks()
    source_chunks[0]['text'] += ' The model preserves accu- racy.'
    value['claims'][0]['references'].append({'chunk_id': 'KIVI', 'quote': 'The model preserves accuracy.'})
    p = pipeline(tmp_path, [value])
    fixed = p.structured('domain', Assessment, 'test', {'chunks': source_chunks},
                         lambda v: validate_perspective(v, source_chunks, 'domain'))
    assert fixed['claims'][0]['references'][-1]['quote'] == 'The model preserves accu- racy.'
    assert len(p.gateway.calls) == 1


@pytest.fixture
def main01_citations():
    return json.loads((Path(__file__).parent / 'fixtures/research/main01_citations.json').read_text())


def citation_assessment(reference):
    value = perspective_assessment('domain')
    value['claims'] = [value['claims'][0]]
    value['claims'][0]['references'] = [deepcopy(reference)]
    return value


def test_main01_wrong_existing_chunk_can_rebind_to_supplied_literal_source(tmp_path, main01_citations):
    data = main01_citations
    source_chunks = data['chunks']
    value = citation_assessment(data['wrongly_bound'])
    before = deepcopy(value)
    errors = validate_assessment(value, source_chunks, 'KIVI')
    assert errors == ['claim 0: quotation does not exist in supplied chunk kivi:p9:t500']

    unit, = plan_repairs(value, errors, {}, source_chunks)
    assert unit.kind == 'reference'
    supporting = next(c for c in source_chunks if c['id'] == data['supporting_chunk_id'])
    assert unit.content['chunks'] == [supporting]
    patch = {'patches': [{'target_id': unit.target_id, 'chunk_id': supporting['id'],
                          'quote': data['wrongly_bound']['quote']}]}
    p = pipeline(tmp_path, [value, patch])
    fixed = p.structured('research_kivi_0', Assessment, 'test', {'chunks': source_chunks},
                         lambda answer: validate_assessment(answer, source_chunks, 'KIVI'))
    expected = deepcopy(value)
    expected['claims'][0]['references'][0] = {
        'chunk_id': supporting['id'],
        'quote': source_verbatim_quote(data['wrongly_bound']['quote'], supporting['text']),
    }
    assert fixed == expected and value == before
    assert 'resid- ual' in fixed['claims'][0]['references'][0]['quote']
    assert not validate_assessment(fixed, source_chunks, 'KIVI')
    assert len(p.gateway.calls) == 1


@pytest.mark.parametrize('excluded', ['not_supplied', 'other_technology'])
def test_main01_reference_cannot_rebind_outside_supplied_technology(main01_citations, excluded):
    data = main01_citations
    value = citation_assessment(data['wrongly_bound'])
    source_chunks = deepcopy(data['chunks'])
    supporting = next(c for c in source_chunks if c['id'] == data['supporting_chunk_id'])
    if excluded == 'not_supplied':
        source_chunks.remove(supporting)
    else:
        supporting['technology'] = 'InfiniGen'
    unit, = plan_repairs(value, validate_assessment(value, source_chunks, 'KIVI'), {}, source_chunks)
    assert unit.kind == 'citation'
    assert all(c['id'] != supporting['id'] for c in unit.content['chunks'])


@pytest.mark.parametrize('change', ['different_literal', 'other_chunk', 'invented'])
def test_main01_rebind_preserves_the_original_quote_and_verified_candidates(main01_citations, change):
    data = main01_citations
    value = citation_assessment(data['wrongly_bound'])
    source_chunks = data['chunks']
    unit, = plan_repairs(value, validate_assessment(value, source_chunks, 'KIVI'), {}, source_chunks)
    item = {'target_id': unit.target_id, 'chunk_id': data['supporting_chunk_id'],
            'quote': data['wrongly_bound']['quote']}
    if change == 'different_literal':
        item['quote'] = 'The effect of residual length.'
    elif change == 'other_chunk':
        item['chunk_id'] = data['wrongly_bound']['chunk_id']
        item['quote'] = 'Table 5: Ablation study of KIVI by changing group size G and residual length R.'
    else:
        item['chunk_id'] = 'kivi:invented'
    with pytest.raises(ValueError):
        apply_patch(value, unit, {'patches': [item]})


@pytest.mark.parametrize('failure', ['paraphrased', 'non_contiguous'])
def test_main01_nonliteral_quotes_still_fail_after_one_correction(tmp_path, main01_citations, failure):
    data = main01_citations
    value = citation_assessment(data[failure])
    source_chunks = data['chunks']
    source = next(c for c in source_chunks if c['id'] == data[failure]['chunk_id'])
    assert source_verbatim_quote(data[failure]['quote'], source['text']) == data[failure]['quote']
    check = lambda answer: validate_assessment(answer, source_chunks, 'KIVI')
    errors = check(value)
    assert len(errors) == 1 and 'quotation does not exist' in errors[0]
    unit, = plan_repairs(value, errors, {}, source_chunks)
    patch = {'patches': [{'target_id': unit.target_id, 'quote': data[failure]['quote']}]}
    p = pipeline(tmp_path, [value, patch])
    with pytest.raises(StructuredValidationError, match='quotation does not exist'):
        p.structured('research_kivi_0', Assessment, 'test', {'chunks': source_chunks}, check)
    assert len(p.gateway.calls) == 2


@pytest.fixture
def main05_citations():
    return json.loads((Path(__file__).parent / 'fixtures/research/main05_citations.json').read_text())


def test_main05_unique_rebind_needs_no_generated_quotation(tmp_path, main05_citations):
    data = main05_citations
    value = citation_assessment(data['wrongly_bound'])
    before = deepcopy(value)
    patch = deepcopy(data['expanded_patch'])
    patch['patches'][0]['target_id'] = 'claims:0:reference:0'
    p = pipeline(tmp_path, [value, patch])
    seen = []
    def check(answer):
        seen.append(deepcopy(answer))
        return validate_assessment(answer, data['chunks'], 'KIVI')
    fixed = p.structured('domain', Assessment, 'test', {'chunks': data['chunks']}, check)
    expected = deepcopy(value)
    expected['claims'][0]['references'][0]['chunk_id'] = data['supporting_chunk_id']
    assert fixed == expected and value == before
    assert len(p.gateway.calls) == 1
    assert seen == [value, expected]
    receipt = json.loads((tmp_path / 'repairs/domain.json').read_text())
    assert receipt['deterministic_units'] == ['claims:0:reference:0']
    assert receipt['full_validation'] == 'passed'


def test_ambiguous_rebind_keeps_bounded_model_selection(tmp_path, main05_citations):
    data = main05_citations
    source_chunks = deepcopy(data['chunks'])
    alternative = deepcopy(next(c for c in source_chunks if c['id'] == data['supporting_chunk_id']))
    alternative['id'] += ':overlap'
    alternative['page'] += 1  # The same sentence at a genuinely different location.
    source_chunks.append(alternative)
    value = citation_assessment(data['wrongly_bound'])
    patch = {'patches': [{'target_id': 'claims:0:reference:0', 'chunk_id': alternative['id'],
                          'quote': data['wrongly_bound']['quote']}]}
    p = pipeline(tmp_path, [value, patch])
    fixed = p.structured('domain', Assessment, 'test', {'chunks': source_chunks},
                         lambda answer: validate_assessment(answer, source_chunks, 'KIVI'))
    assert fixed['claims'][0]['references'][0] == {
        'chunk_id': alternative['id'], 'quote': data['wrongly_bound']['quote']}
    assert len(p.gateway.calls) == 2


@pytest.fixture
def main07_citations():
    return json.loads((Path(__file__).parent / 'fixtures/research/main07_citations.json').read_text())


@pytest.mark.parametrize('case_index', [0, 1, 2])
def test_main07_recovers_source_location_and_case_without_model_repair(tmp_path, main07_citations, case_index):
    data = main07_citations
    case = data['cases'][case_index]
    value = citation_assessment(case['reference'])
    value['claims'][0]['technology'] = case['technology']
    before = deepcopy(value)
    source_chunks = data['chunks']
    p = pipeline(tmp_path, [value])
    fixed = p.structured('stakeholder', Assessment, 'test', {'chunks': source_chunks},
                         lambda answer: validate_assessment(answer, source_chunks, case['technology']))
    expected = deepcopy(value)
    supporting = next(c for c in source_chunks if c['id'] == case['expected_chunk_id'])
    expected['claims'][0]['references'][0] = {'chunk_id': supporting['id'],
        'quote': source_verbatim_quote(case['reference']['quote'], supporting['text'])}
    assert fixed == expected and value == before
    assert not validate_assessment(fixed, source_chunks, case['technology'])
    assert len(p.gateway.calls) == 1


def test_sentence_initial_case_restores_only_a_unique_sentence_start():
    quote = 'we use the same settings for this experiment.'
    actual = 'We use the same settings for this experiment.'
    assert source_verbatim_quote(quote, 'Earlier result. ' + actual) == actual
    assert source_verbatim_quote(quote, actual + ' ' + actual) == quote
    assert source_verbatim_quote(quote, 'The label says ' + actual) == quote


def test_overlapping_repeated_quote_keeps_rebind_ambiguous():
    quote = 'test test test'
    value = citation_assessment({'chunk_id': 'wrong', 'quote': quote})
    source_chunks = [{'id': ident, 'source_id': 'kivi', 'technology': 'KIVI',
                      'page': 1, 'char_start': 0, 'char_end': len(text), 'text': text}
                     for ident, text in [('wrong', 'unrelated source text'),
                                         ('short', quote), ('long', quote + ' test')]]
    unit, = plan_repairs(value, validate_assessment(value, source_chunks, 'KIVI'), {}, source_chunks)
    assert unit.kind == 'reference' and len(unit.content['chunks']) == 2


def test_overlapping_pdf_wrap_matches_do_not_select_one_source_span():
    quote = 'testing testing testing'
    source = 'test-\ning test-\ning test-\ning test-\ning'
    assert source_verbatim_quote(quote, source) == quote


@pytest.mark.parametrize('source', ['We use 30 tokens.', 'WE use 20 tokens.',
    'We Use 20 tokens.', 'We use 20 tokens!', 'They use 20 tokens.'])
def test_initial_case_restoration_does_not_change_numbers_words_or_other_case(source):
    quote = 'we use 20 tokens.'
    assert source_verbatim_quote(quote, source) == quote


@pytest.mark.parametrize('difference', ['source_id', 'page', 'char_start', 'missing_offset', 'repeated'])
def test_overlap_rebind_requires_one_verified_absolute_source_span(main07_citations, difference):
    data = main07_citations
    case = data['cases'][1]
    value = citation_assessment(case['reference'])
    source_chunks = deepcopy(data['chunks'])
    other = next(c for c in source_chunks if c['id'] == 'kivi:p7:t750')
    if difference == 'missing_offset':
        other.pop('char_start')
    elif difference == 'repeated':
        other['text'] += ' ' + case['reference']['quote']
        other['char_end'] = other['char_start'] + len(other['text'])
    elif difference == 'source_id':
        other['source_id'] = 'different-version'
    else:
        other[difference] += 1
        if difference == 'char_start':
            other['char_end'] += 1
    unit, = plan_repairs(value, validate_assessment(value, source_chunks, 'KIVI'), {}, source_chunks)
    assert unit.kind == 'reference' and len(unit.content['chunks']) == 2


def test_main07_expanded_quote_is_still_rejected_after_overlap_resolution(main07_citations):
    data = main07_citations
    case = data['cases'][1]
    value = citation_assessment(case['reference'])
    unit, = plan_repairs(value, validate_assessment(value, data['chunks'], 'KIVI'), {}, data['chunks'])
    expanded = ('From Table 5, we observe that group sizes 32 and 64 yield similar results, whereas '
                + case['reference']['quote'])
    with pytest.raises(ValueError, match='preserve the original quotation'):
        apply_patch(value, unit, {'patches': [{'target_id': unit.target_id,
            'chunk_id': case['expected_chunk_id'], 'quote': expanded}]})


def test_unique_rebind_preserves_other_correction_and_final_validation(tmp_path, main05_citations):
    data = main05_citations
    value = citation_assessment(data['wrongly_bound'])
    original_id = data['wrongly_bound']['chunk_id']
    value['claims'][0]['references'].append({'chunk_id': original_id, 'quote': 'This quote is absent.'})
    source = next(c for c in data['chunks'] if c['id'] == original_id)
    quote = source['text'][:100]
    patch = {'patches': [{'target_id': 'claims:0:reference:1', 'quote': quote}]}
    p = pipeline(tmp_path, [value, patch])
    fixed = p.structured('domain', Assessment, 'test', {'chunks': data['chunks']},
                         lambda answer: validate_assessment(answer, data['chunks'], 'KIVI'))
    assert fixed['claims'][0]['references'] == [
        {'chunk_id': data['supporting_chunk_id'], 'quote': data['wrongly_bound']['quote']},
        {'chunk_id': original_id, 'quote': quote}]
    assert [call[0] for call in p.gateway.calls] == ['domain', 'domain_repair_1']
    assert not validate_assessment(fixed, data['chunks'], 'KIVI')
