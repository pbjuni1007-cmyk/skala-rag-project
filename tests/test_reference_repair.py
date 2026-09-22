from copy import deepcopy
import pytest
from rag.evidence import validate_perspective
from rag.repair import plan_repairs, apply_patch
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
