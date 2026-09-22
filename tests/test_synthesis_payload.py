from copy import deepcopy
from rag.payloads import synthesis_payload


def test_reference_table_round_trip_preserves_all_prose_and_reference_order():
    ref = {'chunk_id': 'source:1', 'quote': 'The full original quotation.'}
    claims = [{'id': 'a', 'text': 'Judgment', 'conditions': 'All conditions', 'caveats': 'All caveats', 'references': [ref, ref]},
              {'id': 'b', 'text': 'Another judgment', 'references': [ref, {**ref, 'quote': 'A different full quotation.'}]}]
    before = deepcopy(claims)
    result = synthesis_payload(claims, ['Keep the complete conflict'])
    assert len(result['reference_table']) == 2
    restored = []
    for claim in result['claims']:
        item = deepcopy(claim)
        item['references'] = [result['reference_table'][key] for key in item.pop('reference_ids')]
        restored.append(item)
    assert restored == before and claims == before
    assert result['conflicts'] == ['Keep the complete conflict']


def test_different_chunk_identity_or_quote_whitespace_is_not_silently_merged():
    refs = [{'chunk_id': 'one', 'quote': 'same words'}, {'chunk_id': 'two', 'quote': 'same words'},
            {'chunk_id': 'one', 'quote': 'same  words'}]
    result = synthesis_payload([{'references': refs}], [])
    assert len(result['reference_table']) == 3
