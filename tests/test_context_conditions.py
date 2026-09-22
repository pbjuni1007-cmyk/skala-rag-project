from copy import deepcopy
import json
from pathlib import Path

import pytest
from rag.context import assessment_evidence, build_research_context, companion_candidates, policy_identity
from scripts.evaluate_context import verify_dataset, recover


class Corpus:
    def __init__(self, chunks, adjacent=()):
        self.chunks = chunks
        self.adjacent = adjacent
        self.config = {}

    def evidence_tokens(self, text):
        return len(text.split())

    def adjacent_candidates(self, ranked):
        return list(self.adjacent)


def chunk(i, text='full original words', source='a', page=1):
    return dict(id=f'{source}:{i}', source_id=source, technology='Tech', page=page,
                char_start=i * 100, char_end=i * 100 + len(text), text=text)


def assessment(*ids):
    return {'Tech': {'claims': [{'references': [{'chunk_id': i} for i in ids]}]}}


def test_research_preserves_entire_legacy_and_full_source_chunks():
    chunks = [chunk(0), chunk(1, 'Figure 5: full caption'), chunk(2)]
    corpus = Corpus(chunks, [chunks[2]])
    before = deepcopy(chunks)
    result = build_research_context(corpus, [chunks[0]], 'Tech',
                                    {'evidence_token_budget': 10, 'adjacent_token_budget': 3})
    assert result[:2] == [chunks[2], chunks[0]]
    assert chunks[1] in result
    assert sum(corpus.evidence_tokens(c['text']) for c in result) <= 10
    assert all(any(c is original for original in chunks) for c in result)
    assert chunks == before
    assert len({c['id'] for c in result}) == len(result)


def test_neighbors_cross_pages_but_never_cross_sources_or_expand_recursively():
    chunks = [chunk(0), chunk(1, page=2), chunk(2, page=2), chunk(3, source='other')]
    assert companion_candidates(Corpus(chunks), [chunks[0]]) == [chunks[1]]


def test_assessment_keeps_every_citation_without_mutation_and_adds_setup():
    chunks = [chunk(0), chunk(1, 'We use hardware and batch size'), chunk(2)]
    corpus = Corpus(chunks)
    value = assessment(chunks[0]['id'], chunks[0]['id'])
    saved = deepcopy(value)
    result = assessment_evidence(corpus, value, {'assessment_evidence_token_budget': 9, 'adjacent_token_budget': 6})
    assert result == chunks[:2]
    assert value == saved
    assert chunks[0] is result[0]


def test_cited_overflow_or_missing_id_fails_without_silent_loss():
    corpus = Corpus([chunk(0)])
    with pytest.raises(ValueError, match='Required cited'):
        assessment_evidence(corpus, assessment('a:0'), {'assessment_evidence_token_budget': 2, 'adjacent_token_budget': 0})
    with pytest.raises(ValueError, match='absent'):
        assessment_evidence(corpus, assessment('missing'))


def test_zero_companion_budget_and_policy_identity():
    corpus = Corpus([chunk(0), chunk(1)])
    assert assessment_evidence(corpus, assessment('a:0'), {'assessment_evidence_token_budget': 6, 'adjacent_token_budget': 0}) == corpus.chunks[:1]
    assert policy_identity({}) != policy_identity({'adjacent_token_budget': 0})
    assert policy_identity({}) != policy_identity({'assessment_evidence_token_budget': 6400})


def test_diagnostic_source_spans_and_known_item_label():
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads((root/'eval/context-conditions.json').read_text())
    verify_dataset(dataset, json.loads((root/'data/pages.json').read_text()))
    assert 'holdout' in dataset['usage'].lower()
    rows = {q['id']: q for q in dataset['questions']}
    k = ' '.join(e['quote'] for e in rows['K3-expanded']['evidence'])
    i = ' '.join(e['quote'] for e in rows['I1-expanded']['evidence'])
    assert 'ShareGPT' in k and '2bit' in k and '16bit' in k
    assert 'partial' in i and 'CPU' in i
    bad = deepcopy(dataset)
    bad['questions'][0]['evidence'][0]['quote'] += ' fabricated'
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_dataset(bad, json.loads((root/'data/pages.json').read_text()))


def test_union_recovery_requires_all_characters():
    evidence = dict(source_id='a', page=1, start=0, end=10)
    a = dict(source_id='a', page=1, char_start=0, char_end=6)
    b = dict(source_id='a', page=1, char_start=5, char_end=10)
    assert recover(evidence, [a, b])
    assert not recover(evidence, [a])


def test_lexical_setup_fallback_is_same_source_bounded_and_pairs_lengths():
    chunks = [chunk(0), chunk(1, 'Figure 1: exact caption'), chunk(2),
              chunk(3, 'workload workload workload batch size'),
              chunk(4, 'We use input tokens and output tokens with batch size'),
              chunk(5, 'input prompt length and output length for a workload'),
              chunk(6, 'input tokens output tokens batch size', source='other')]
    result = companion_candidates(Corpus(chunks), [chunks[0]])
    assert result[0] is chunks[1]
    assert result[1] is chunks[4]
    assert chunks[5] in result
    assert result.index(chunks[3]) > result.index(chunks[5])  # paired fields outrank repetitions
    assert chunks[6] not in result
    assert all(any(c is original for original in chunks) for c in result)


def test_baseline_seeds_reach_neighbors_without_recursive_expansion():
    chunks = [chunk(0), chunk(1), chunk(2, 'Figure 1: exact caption'), chunk(3)]
    corpus = Corpus(chunks, [chunks[3]])
    result = build_research_context(corpus, [chunks[0]], 'Tech',
                                    {'evidence_token_budget': 10, 'adjacent_token_budget': 3})
    assert chunks[2] in result
    assert chunks[0] in result


def test_actual_diagnostic_preserves_original_gold_and_verifies_new_workload_span():
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root/'eval/context-conditions.json').read_text())
    actual = json.loads((root/'eval/context-actual-conditions.json').read_text())
    assert actual['questions'][:len(original['questions'])] == original['questions']
    verify_dataset(actual, json.loads((root/'data/pages.json').read_text()))
    quote = actual['questions'][-1]['evidence'][0]['quote']
    assert 'ShareGPT' in quote and '161' in quote and '338' in quote
    assert actual['queries'] != original['queries']
