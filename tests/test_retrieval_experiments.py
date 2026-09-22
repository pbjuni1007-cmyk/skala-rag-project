import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('retrieval_experiments', ROOT / 'scripts/retrieval_experiments.py')
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False, truncation=False):
        assert truncation is False
        return list(text) + ([0, 0] if add_special_tokens else [])

    def __call__(self, text, **kwargs):
        assert kwargs.get('truncation', False) is False
        return {'offset_mapping': [(i, i + 1) for i in range(len(text))]}


def page(text='abcdefghij', number=1):
    return {'source_id': 'test', 'technology': 'test', 'page': number, 'text': text}


def test_truncation_is_error():
    with pytest.raises(ValueError, match='truncate'):
        r.guard(CharacterTokenizer(), ['1234'], {'limit': 5})


def test_chunk_offsets_zero_overlap_exact_page_coverage():
    p = page('abc  def\nghij')
    chunks = r.token_chunks([p], CharacterTokenizer(), 4, 0, {'limit': 9, 'document_prefix': 'x: '})
    assert ''.join(c['text'] for c in chunks) == p['text']
    assert all(a['end'] == b['start'] for a, b in zip(chunks, chunks[1:]))
    assert all(c['text'] == p['text'][c['start']:c['end']] for c in chunks)
    assert r.duplicate_ratio(chunks) == 0


def test_overlap_is_bounded_and_terminates():
    chunks = r.token_chunks([page()], CharacterTokenizer(), 4, 2, {'limit': 9, 'document_prefix': 'x: '})
    assert [c['start'] for c in chunks] == [0, 2, 4, 6, 8]
    assert 0 < r.duplicate_ratio(chunks) < 1
    with pytest.raises(ValueError):
        r.token_chunks([page()], CharacterTokenizer(), 4, 4, {'limit': 9, 'document_prefix': ''})


def test_span_union_complete_but_gap_not_complete():
    p = page()
    evidence = {'source_id': 'test', 'page': 1, 'start': 1, 'end': 9}
    assert r.span_recovered(evidence, [r.make_chunk(p, 0, 5), r.make_chunk(p, 5, 10)])
    assert not r.span_recovered(evidence, [r.make_chunk(p, 0, 4), r.make_chunk(p, 5, 10)])
    assert not r.span_recovered(evidence, [r.make_chunk(page(number=2), 0, 10)])


def test_condition_requires_all_mapped_spans():
    p = page()
    q = {'evidence': [{'source_id': 'test', 'page': 1, 'start': 0, 'end': 3}, {'source_id': 'test', 'page': 1, 'start': 8, 'end': 10}], 'conditions': [[0, 1]]}
    assert r.metrics(q, [r.make_chunk(p, 0, 3)])['condition_recovery'] == 0
    assert r.metrics(q, [r.make_chunk(p, 0, 3), r.make_chunk(p, 8, 10)])['complete'] == 1


def test_budget_counts_separators_and_never_truncates():
    p = page('0123456789012345')
    ranked = [r.make_chunk(p, 0, 7), r.make_chunk(p, 7, 14), r.make_chunk(p, 14, 16)]
    selected = r.budget_context(ranked, CharacterTokenizer(), 11)
    assert [c['start'] for c in selected] == [0, 14]
    assert r.tokens(CharacterTokenizer(), r.context_text(selected)) == 11


def test_common_body_identical_and_every_prefix_fits():
    p = page('one two three four five')
    tok = CharacterTokenizer()
    specs = [(tok, {'limit': 12, 'document_prefix': 'x:'}), (tok, {'limit': 10, 'document_prefix': ''})]
    chunks = r.common_chunks([p], specs)
    assert ''.join(c['text'] for c in chunks) == p['text']
    for t, spec in specs:
        r.guard(t, [spec['document_prefix'] + c['text'] for c in chunks], spec)


def result(cid, family='operational', complete=(1, 0), condition=(1, 0), seconds=1, split='selection', model='e5'):
    return {'config': {'id': cid, 'family': family, 'model': model}, 'split': split,
            'rows': [{'id': str(i), 'language': 'en', 'required': i == 0, 'query_seconds': [seconds],
                      'metrics': {'budget2000': {'complete': v, 'condition_recovery': condition[i]}}} for i, v in enumerate(complete)]}


def test_required_per_query_regression_blocks_even_same_average():
    baseline = result(r.BASELINE)
    candidate = result('e5-200-0', complete=(0, 1), condition=(0, 1), seconds=.1)
    assert not r.no_required_regression(candidate, baseline)
    chosen = r.choose([baseline, candidate, result('e5-common', 'common')])
    assert chosen['selected'] == r.BASELINE


def test_holdout_cannot_leak_into_selection():
    with pytest.raises(ValueError, match='Holdout'):
        r.choose([result(r.BASELINE, split='holdout')])


def test_common_model_gate_binds_operational_selection():
    rows = [result(r.BASELINE), result('e5-common', 'common'),
            result('bge-small-common', 'common', complete=(0, 1), model='bge-small'),
            result('bge-small-380-50', complete=(1, 1), condition=(1, 1), model='bge-small')]
    assert r.choose(rows)['selected'] == r.BASELINE


def test_validation_falls_back_for_condition_regression():
    baseline = result(r.BASELINE, complete=(1, 0), condition=(1, .5), split='holdout')
    candidate = result('candidate', complete=(1, 0), condition=(1, .25), split='holdout')
    assert not r.holdout_passes(candidate, baseline)
    candidate['rows'][1]['metrics']['budget2000']['condition_recovery'] = .5
    assert r.holdout_passes(candidate, baseline)


def test_original_dataset_source_hashes_and_distinct_holdout():
    dataset = r.read(ROOT / 'eval/retrieval-technical.json')
    pages = r.read(ROOT / 'data/pages.json')
    r.validate_dataset(dataset, pages)
    assert dataset['original_pages_sha256'] == r.digest(ROOT / 'data/pages.json')
    old = r.read(ROOT / 'eval/embedding-comparison.json')['questions']
    selection = [q for q in dataset['questions'] if q['split'] == 'selection']
    assert [q['queries'] for q in selection] == [q['queries'] for q in old]
    holdout = [q for q in dataset['questions'] if q['split'] == 'holdout']
    assert len(holdout) == 8
    assert len({q['queries']['en'] for q in dataset['questions']}) == 24
    assert all(q['distinct_from_selection'] for q in holdout)
    assert sum(len({e['page'] for e in q['evidence']}) > 1 for q in holdout) >= 3
    for q in holdout:
        assert not any(q['queries'] == old_q['queries'] for old_q in old)
    broken = copy.deepcopy(dataset)
    broken['questions'][0]['evidence'][0]['start'] += 1
    with pytest.raises(ValueError, match='source mismatch'):
        r.validate_dataset(broken, pages)


def test_table_figure_holdout_has_required_qualifiers():
    qs = {q['id']: q for q in r.read(ROOT / 'eval/retrieval-technical.json')['questions']}
    quotes = lambda qid: ' '.join(e['quote'] for e in qs[qid]['evidence'])
    assert all(s in quotes('H1') for s in ('2048', '80%', '30.99 33.84'))
    assert all(s in quotes('H3') for s in ('beyond 4', 'ratio of 0.3', 'doubles'))
    assert all(s in quotes('H5') for s in ('38.36', '26.46', '36.01', 'full precision'))
    assert 'future work' in quotes('H7')
    assert all(s in quotes('H8') for s in ('1920', '128', '27.36', '41.99'))


def test_baseline_matches_production_corpus_build_offsets_and_trailing_chunk(tmp_path):
    import numpy as np
    from rag.corpus import Corpus
    from types import SimpleNamespace
    p = page('a' * 701)
    tok = CharacterTokenizer()
    class Encoder:
        tokenizer = tok
        def encode(self, bodies, **kwargs):
            return np.zeros((len(bodies), 2), dtype=np.float32)
    corpus = Corpus.__new__(Corpus)
    corpus.load_encoder = lambda: None
    corpus.encoder = Encoder()
    corpus.pages = [p]
    corpus.config = {'chunk_tokens': 380, 'overlap_tokens': 50}
    corpus.embedding = {'prefixes': ['query: ', 'passage: '], 'max_tokens': 512}
    corpus.settings = SimpleNamespace(integer=lambda name, default: default)
    corpus.index = tmp_path
    corpus.build()
    experiment = r.token_chunks([p], tok, 380, 50, {'document_prefix': 'passage: ', 'limit': 512})
    assert [(c['start'], c['end'], c['text']) for c in experiment] == [(c['char_start'], c['char_end'], c['text']) for c in corpus.chunks]
    assert [c['start'] for c in experiment] == [0, 330, 660]


def test_m3_long_special_tokens_fit_without_changing_operational_guard():
    p = page('x' * 1100)
    spec = {'document_prefix': '', 'limit': 1024}
    with pytest.raises(ValueError, match='truncate'):
        r.token_chunks([p], CharacterTokenizer(), 1024, 50, spec)
    chunks = r.token_chunks([p], CharacterTokenizer(), 1024, 50, spec, trim_to_limit=True)
    assert chunks[0]['end'] == 1022
    r.guard(CharacterTokenizer(), [c['text'] for c in chunks], spec)


def test_korean_secondary_results_do_not_choose_or_veto_candidate():
    baseline = result(r.BASELINE)
    candidate = result('e5-200-0', complete=(1, 1), condition=(1, 1))
    for output, korean_complete in [(baseline, 1), (candidate, 0)]:
        secondary = copy.deepcopy(output['rows'][0])
        secondary['language'] = 'ko'
        secondary['metrics']['budget2000']['complete'] = korean_complete
        secondary['metrics']['budget2000']['condition_recovery'] = korean_complete
        output['rows'].append(secondary)
    common = copy.deepcopy(baseline)
    common['config'] = {'id': 'e5-common', 'family': 'common', 'model': 'e5'}
    assert r.choose([baseline, candidate, common])['selected'] == 'e5-200-0'
    baseline['split'] = candidate['split'] = 'holdout'
    assert r.holdout_passes(candidate, baseline)
