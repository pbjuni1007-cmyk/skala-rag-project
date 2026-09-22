"""Replay saved query pairs against one frozen corpus without model calls."""
import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.context import build_research_context
from scripts.evaluate_context import recover, verify_dataset

FACETS = {'mechanism', 'limitation', 'conditions', 'maturity'}


def validate_pairs(cases):
    ids = set()
    if not cases:
        raise ValueError('No observed rewrite pairs')
    for case in cases:
        if case['id'] in ids:
            raise ValueError('Duplicate rewrite case')
        ids.add(case['id'])
        for key in ('before', 'after'):
            queries = case[key]
            if len(queries) != 4 or {q['facet'] for q in queries} != FACETS:
                raise ValueError('A pair must retain all four facets')
            if any(q['technology'] != case['technology'] or not q['query'].strip() for q in queries):
                raise ValueError('Technology and nonempty queries must be preserved')


def compare_pairs(corpus, config, cases, dataset):
    validate_pairs(cases)
    results = []
    for case in cases:
        views = {}
        for view in ('before', 'after'):
            groups = [corpus.search(q['query'], case['technology'], config['top_k']) for q in case[view]]
            ranked = [c for row in zip_longest(*groups) for c in row if c]
            chunks = build_research_context(corpus, ranked, case['technology'], config)
            questions = [q for q in dataset['questions'] if q['technology'] == case['technology']]
            if not questions:
                raise ValueError('No reference evidence for technology')
            scores = [{'id': q['id'], 'complete': all(recover(e, chunks) for e in q['evidence']),
                       'spans': [recover(e, chunks) for e in q['evidence']]} for q in questions]
            views[view] = {'chunk_ids': [c['id'] for c in chunks],
                           'context_tokens': sum(corpus.evidence_tokens(c['text']) for c in chunks),
                           'complete_count': sum(q['complete'] for q in scores), 'total': len(scores),
                           'condition_count': sum(sum(q['spans']) for q in scores),
                           'condition_total': sum(len(q['spans']) for q in scores), 'questions': scores}
        before, after = views['before'], views['after']
        results.append({'id': case['id'], 'technology': case['technology'], **views,
            'new_chunk_ids': sorted(set(after['chunk_ids']) - set(before['chunk_ids'])),
            'complete_delta': after['complete_count'] - before['complete_count'],
            'condition_delta': after['condition_count'] - before['condition_count']})
    unique = {}
    for case, result in zip(cases, results):
        key = hashlib.sha256(json.dumps([case['before'], case['after']], sort_keys=True).encode()).hexdigest()
        unique.setdefault(key, result)
    return {'unique_pairs': len(unique),
            'unique_improved': sum(r['complete_delta'] > 0 for r in unique.values()),
            'unique_unchanged': sum(r['complete_delta'] == 0 for r in unique.values()),
            'unique_regressed': sum(r['complete_delta'] < 0 for r in unique.values()),
            'scope': 'controlled replay of observed queries; known-item diagnostic, not independent holdout',
            'paid_model_calls': 0, 'results': results,
            'improved': sum(r['complete_delta'] > 0 for r in results),
            'unchanged': sum(r['complete_delta'] == 0 for r in results),
            'regressed': sum(r['complete_delta'] < 0 for r in results)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, default=Path('eval/rewrite-pairs.json'))
    p.add_argument('--dataset', type=Path, default=Path('eval/context-conditions.json'))
    p.add_argument('--config', type=Path, default=Path('config/run.yaml'))
    p.add_argument('--pages', type=Path, default=Path('data/pages.json'))
    p.add_argument('--cache', type=Path, default=Path('.cache/huggingface/hub'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import yaml
    from scripts.revalidate_retrieval import load_offline_corpus
    config = yaml.safe_load(args.config.read_text())
    # The common loader builds the current chunking without fetching sources.
    corpus = load_offline_corpus(config=config, pages_path=args.pages, cache_path=args.cache, index_path=args.output.parent / 'rewrite-index')
    dataset = json.loads(args.dataset.read_text())
    verify_dataset(dataset, json.loads(args.pages.read_text()))
    result = compare_pairs(corpus, config, json.loads(args.pairs.read_text())['cases'], dataset)
    result['config'] = config
    result['source_hashes'] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                             [('pairs', args.pairs), ('dataset', args.dataset), ('pages', args.pages), ('runner', Path(__file__))]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('improved','unchanged','regressed','paid_model_calls')}))


if __name__ == '__main__':
    main()
