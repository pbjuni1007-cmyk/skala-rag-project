#!/usr/bin/env python3
"""Frozen, source-offset retrieval experiments. No LLM, key, or network access.

freeze -> prepare -> run-all -> select -> validate. Only validate reads holdout
queries. Common literal comparisons gate model eligibility; operational runs gate
configuration selection. M3-1024 and adjacent-page results are diagnostic only.
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import importlib.metadata
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

for _thread_env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_env] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'

BASELINE = 'e5-380-50'
SELECTION_RULE = 'required-query complete non-regression; complete, conditions, latency; holdout fallback'


def read(path):
    return json.loads(Path(path).read_text())


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    return digest_bytes(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def tokens(tokenizer, text, special=False):
    return len(tokenizer.encode(text, add_special_tokens=special, truncation=False))


def guard(tokenizer, texts, spec):
    lengths = [tokens(tokenizer, t, True) for t in texts]
    if max(lengths, default=0) > spec['limit']:
        raise ValueError('Input would truncate')
    return lengths


def validate_dataset(dataset, pages):
    lookup = {(p['source_id'], p['page']): p for p in pages}
    ids = set()
    for q in dataset['questions']:
        if q['id'] in ids or q['split'] not in ('selection', 'holdout'):
            raise ValueError('Duplicate ID or invalid split')
        ids.add(q['id'])
        if not q['evidence'] or not q['conditions'] or not q['queries']:
            raise ValueError('Empty ground truth')
        for e in q['evidence']:
            p = lookup[e['source_id'], e['page']]
            if not 0 <= e['start'] < e['end'] <= len(p['text']):
                raise ValueError('Invalid evidence offsets')
            if p['text'][e['start']:e['end']] != e['quote']:
                raise ValueError('Ground truth source mismatch')
            if e['quote_sha256'] != digest_bytes(e['quote'].encode()) or e['page_sha256'] != digest_bytes(p['text'].encode()):
                raise ValueError('Ground truth hash mismatch')
            if p['technology'] != q['technology']:
                raise ValueError('Ground truth technology mismatch')
        for condition in q['conditions']:
            if not condition or any(type(i) is not int or i < 0 or i >= len(q['evidence']) for i in condition):
                raise ValueError('Unmapped condition')
        if set(i for group in q['conditions'] for i in group) != set(range(len(q['evidence']))):
            raise ValueError('Every evidence span must be scored')
    expected = {'technical': (16, 8), 'enterprise': (8, 8)}[dataset['corpus']]
    actual = tuple(sum(q['split'] == s for q in dataset['questions']) for s in ('selection', 'holdout'))
    if expected != actual:
        raise ValueError(f'Expected split {expected}, got {actual}')
    if dataset.get('review_status') != 'human_review_pending':
        raise ValueError('Do not silently upgrade human review status')


def make_chunk(page, a, b):
    return {'id': f"{page['source_id']}-p{page['page']}-{a}-{b}", 'source_id': page['source_id'],
            'technology': page['technology'], 'page': page['page'], 'start': a, 'end': b,
            'text': page['text'][a:b]}


def token_chunks(pages, tokenizer, size, overlap, spec, trim_to_limit=False):
    """Corpus.build literal spans, including its trailing overlapping chunk.

    Operational grids use each model's tokenizer and exactly the production
    offset/stride algorithm. Only exploratory M3-1024 trims its endpoint to
    reserve special-token space inside the 1024 total input limit.
    """
    if not 0 <= overlap < size:
        raise ValueError('Overlap must be smaller than chunk size')
    chunks = []
    for page in pages:
        offsets = tokenizer(page['text'], add_special_tokens=False, truncation=False, return_offsets_mapping=True)['offset_mapping']
        for start in range(0, len(offsets), size - overlap):
            end = min(start + size, len(offsets))
            while end > start:
                body = page['text'][offsets[start][0]:offsets[end-1][1]]
                if tokens(tokenizer, spec['document_prefix'] + body, True) <= spec['limit']:
                    break
                if not trim_to_limit:
                    raise ValueError('Operational chunk would truncate; do not silently alter baseline spans')
                end -= 1
            if end == start:
                raise ValueError('One source token cannot fit without truncation')
            chunks.append(make_chunk(page, offsets[start][0], offsets[end-1][1]))
    return chunks


def common_chunks(pages, tokenizers):
    """Identical literal bodies; each full prefixed input fits every model."""
    chunks = []
    for page in pages:
        boundaries = sorted(set([0, len(page['text'])] + [m.start() for m in re.finditer(r'\S+', page['text'])]))
        start = 0
        while start < len(boundaries) - 1:
            lo, hi, best = start + 1, len(boundaries) - 1, start
            while lo <= hi:
                mid = (lo + hi) // 2
                body = page['text'][boundaries[start]:boundaries[mid]]
                if all(tokens(tok, s['document_prefix'] + body, True) <= s['limit'] for tok, s in tokenizers):
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            if best == start:
                raise ValueError('Source word exceeds common model limit')
            chunks.append(make_chunk(page, boundaries[start], boundaries[best]))
            start = best
    return chunks


def covered_length(intervals):
    total, stop = 0, -1
    for a, b in sorted(intervals):
        total += max(0, b - max(a, stop))
        stop = max(stop, b)
    return total


def span_recovered(evidence, chunks):
    intervals = [(max(evidence['start'], c['start']), min(evidence['end'], c['end'])) for c in chunks
                 if c['source_id'] == evidence['source_id'] and c['page'] == evidence['page']
                 and c['start'] < evidence['end'] and c['end'] > evidence['start']]
    return covered_length(intervals) == evidence['end'] - evidence['start']


def duplicate_ratio(chunks):
    total = sum(c['end'] - c['start'] for c in chunks)
    groups = {}
    for c in chunks:
        groups.setdefault((c['source_id'], c['page']), []).append((c['start'], c['end']))
    unique = sum(covered_length(v) for v in groups.values())
    return 1 - unique / total if total else 0


def metrics(question, chunks):
    recovered = [span_recovered(e, chunks) for e in question['evidence']]
    conditions = [all(recovered[i] for i in group) for group in question['conditions']]
    return {'complete': int(all(recovered)), 'condition_recovery': statistics.mean(conditions),
            'duplicate_ratio': duplicate_ratio(chunks), 'span_recovered': recovered}


def context_text(chunks):
    return '\n\n'.join(c['text'] for c in chunks)


def budget_context(ranked, tokenizer, budget=2000):
    """Whole spans only. Joining separators are included in the shared E5 budget."""
    selected = []
    for chunk in ranked:
        if tokens(tokenizer, context_text(selected + [chunk])) <= budget:
            selected.append(chunk)
    return selected


def adjacent_order(ranked):
    """Top hit first, best-ranked chunks on both adjacent pages next, then rest."""
    if not ranked:
        return []
    first = ranked[0]
    adjacent = [c for c in ranked[1:] if c['source_id'] == first['source_id'] and abs(c['page'] - first['page']) == 1]
    ids = {c['id'] for c in adjacent + [first]}
    return [first] + adjacent + [c for c in ranked if c['id'] not in ids]


def snapshot(cache, spec):
    return Path(cache) / ('models--' + spec['repo'].replace('/', '--')) / 'snapshots' / spec['revision']


def load_inputs(args):
    protocol = read(args.output / 'protocol.json')
    if protocol['runner_sha256'] != digest(__file__):
        raise ValueError('Runner changed after freeze')
    for field, path in [('pages', args.pages), ('dataset', args.dataset), ('models', args.models)]:
        if digest(path) != protocol[field + '_sha256']:
            raise ValueError(f'Frozen {field} changed')
    return protocol, read(args.pages), read(args.dataset), read(args.models)


def freeze(args):
    pages, dataset, models = read(args.pages), read(args.dataset), read(args.models)
    validate_dataset(dataset, pages)
    if dataset.get('original_pages_sha256') != digest(args.pages):
        raise ValueError('Dataset was authored for different pages')
    if set(models) != {'e5', 'bge-small', 'bge-m3'}:
        raise ValueError('Three explicit model specs required')
    for spec in models.values():
        if not re.fullmatch(r'[a-f0-9]{40}', spec.get('revision', '')):
            raise ValueError('Models require exact verified 40-character revisions')
        if spec.get('trust_remote_code') is not False or spec.get('mode') != 'dense':
            raise ValueError('Only dense models without remote code are allowed')
    protocol = {'schema_version': 1, 'runner_sha256': digest(__file__), 'pages_sha256': digest(args.pages), 'dataset_sha256': digest(args.dataset),
                'models_sha256': digest(args.models), 'source_pages_sha256': {f"{p['source_id']}:{p['page']}": digest_bytes(p['text'].encode()) for p in pages},
                'baseline': BASELINE, 'selection_rule': SELECTION_RULE, 'selection_metric_view': 'budget2000', 'primary_language': 'en', 'secondary_language': 'ko',
                'corpus': dataset['corpus'], 'review_status': 'human_review_pending', 'threads': 2,
                'query_repeats': 3, 'context_tokenizer': 'e5', 'context_budget': 2000,
                'top_k': 5, 'common_gate': 'each required selection query non-regression against common E5',
                'holdout_gate': 'candidate complete and condition means and each required query must not regress',
                'adjacent_policy': 'top hit then ranked adjacent-page chunks; diagnostic only',
                'enterprise_policy': 'separate corpus, never authorizes operational model change',
                'grid': [[s, o] for s in (200, 300, 380) for o in (0, 25, 50)],
                'operational_chunking': 'Corpus.build offset/stride algorithm including trailing chunks; each model uses its own tokenizer',
                'm3_1024': 'separate exploratory run; endpoint trimmed for special tokens within total limit 1024; never selection eligible'}
    write(args.output / 'protocol.json', protocol, exclusive=True)


def tokenizer_for(cache, spec):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(snapshot(cache, spec)), local_files_only=True, trust_remote_code=False, use_fast=True)


def prepare(args):
    protocol, pages, dataset, models = load_inputs(args)
    toks = {key: tokenizer_for(args.cache, spec) for key, spec in models.items()}
    bounded = {key: dict(spec, limit=512) for key, spec in models.items()}
    shared = common_chunks(pages, [(toks[k], s) for k, s in bounded.items()])
    configs = []
    for key, spec in bounded.items():
        configs.append({'id': key + '-common', 'model': key, 'family': 'common', 'limit': 512, 'chunks': shared})
        grid = protocol['grid'] if key == 'e5' else [(380, 50)]
        for size, overlap in grid:
            configs.append({'id': f'{key}-{size}-{overlap}', 'model': key, 'family': 'operational', 'limit': 512, 'size': size, 'overlap': overlap,
                            'chunks': token_chunks(pages, toks[key], size, overlap, spec)})
        if key == 'bge-m3':
            configs.append({'id': 'bge-m3-1024-50', 'model': key, 'family': 'exploratory', 'limit': 1024, 'size': 1024, 'overlap': 50,
                            'chunks': token_chunks(pages, toks[key], 1024, 50, dict(spec, limit=1024), trim_to_limit=True)})
    manifest = []
    for config in configs:
        chunks = config.pop('chunks')
        guard(toks[config['model']], [models[config['model']]['document_prefix'] + c['text'] for c in chunks], dict(models[config['model']], limit=config['limit']))
        path = args.output / 'chunks' / (config['id'] + '.json')
        write(path, chunks, exclusive=True)
        manifest.append({**config, 'chunk_count': len(chunks), 'chunks_sha256': digest(path)})
    write(args.output / 'manifest.json', {'protocol_sha256': digest(args.output / 'protocol.json'), 'configs': manifest}, exclusive=True)


def check_manifest(args):
    manifest = read(args.output / 'manifest.json')
    if manifest['protocol_sha256'] != digest(args.output / 'protocol.json'):
        raise ValueError('Frozen protocol changed')
    for config in manifest['configs']:
        if config['chunks_sha256'] != digest(args.output / 'chunks' / (config['id'] + '.json')):
            raise ValueError('Frozen chunks changed')
    return manifest


def result_path(args, config_id, split='selection'):
    return args.output / 'results' / f'{config_id}-{split}.json'


def run_model(args):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    protocol, pages, dataset, models = load_inputs(args)
    manifest = check_manifest(args)
    split = args.split
    questions = [q for q in dataset['questions'] if q['split'] == split]
    configs = [c for c in manifest['configs'] if c['model'] == args.model]
    selection_hash = None
    if split == 'holdout':
        selection = read(args.output / 'selection.json')
        selection_hash = digest(args.output / 'selection.json')
        if selection['protocol_sha256'] != digest(args.output / 'protocol.json'):
            raise ValueError('Selection protocol mismatch')
        allowed = {BASELINE, selection['selected']}
        configs = [c for c in configs if c['id'] in allowed]
        lock = read(args.output / 'validation-lock.json')
        if lock['selection_sha256'] != selection_hash:
            raise ValueError('Holdout selection changed')
    elif (args.output / 'validation-lock.json').exists():
        raise ValueError('No tuning after holdout starts')
    if not configs:
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    spec = models[args.model]
    start = time.perf_counter()
    model = SentenceTransformer(str(snapshot(args.cache, spec)), device='cpu', local_files_only=True, trust_remote_code=False)
    model.max_seq_length = 512
    load_seconds = time.perf_counter() - start
    e5_tok = tokenizer_for(args.cache, models['e5'])
    batch = 1 if args.model == 'bge-m3' else 16
    model.encode([spec['query_prefix'] + 'warm up'], batch_size=1, normalize_embeddings=True, show_progress_bar=False, prompt='')
    for config in configs:
        spec = dict(models[args.model], limit=config['limit'])
        model.max_seq_length = spec['limit']
        path = result_path(args, config['id'], split)
        if path.exists():
            prior_result = read(path)
            if prior_result['protocol_sha256'] != digest(args.output / 'protocol.json') or prior_result['manifest_sha256'] != digest(args.output / 'manifest.json') or prior_result['selection_sha256'] != selection_hash:
                raise ValueError('Existing result provenance mismatch')
            if digest(args.output / 'vectors' / (config['id'] + '.npy')) != prior_result['vectors_sha256']:
                raise ValueError('Existing index changed')
            continue
        chunks = read(args.output / 'chunks' / (config['id'] + '.json'))
        doc_texts = [spec['document_prefix'] + c['text'] for c in chunks]
        lengths = guard(model.tokenizer, doc_texts, spec)
        vector_path = args.output / 'vectors' / (config['id'] + '.npy')
        if split == 'selection':
            start = time.perf_counter()
            vectors = model.encode(doc_texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False, prompt='', convert_to_numpy=True)
            index_seconds = time.perf_counter() - start
            if vectors.shape[1] != spec['dimension']:
                raise ValueError('Unexpected dense vector dimension')
            vector_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(vector_path, vectors)
        else:
            prior = read(result_path(args, config['id']))
            if digest(vector_path) != prior['vectors_sha256']:
                raise ValueError('Index changed before holdout')
            vectors = np.load(vector_path, allow_pickle=False)
            index_seconds = prior['index_build_seconds']
        rows = []
        for q in questions:
            for language, query in q['queries'].items():
                query_text = spec['query_prefix'] + query
                guard(model.tokenizer, [query_text], spec)
                times = []
                for _ in range(protocol['query_repeats']):
                    start = time.perf_counter()
                    vector = model.encode([query_text], batch_size=1, normalize_embeddings=True, show_progress_bar=False, prompt='')[0]
                    scores = vectors @ vector
                    indices = sorted((i for i, c in enumerate(chunks) if c['technology'] == q['technology']), key=lambda i: (-float(scores[i]), i))
                    times.append(time.perf_counter() - start)
                ranked = [dict(chunks[i], cosine=float(scores[i])) for i in indices]
                contexts = {'top5': ranked[:5], 'budget2000': budget_context(ranked, e5_tok),
                            'adjacent2000': budget_context(adjacent_order(ranked), e5_tok)}
                rows.append({'id': q['id'], 'language': language, 'required': q.get('required', False),
                             'query_seconds': times, 'metrics': {view: metrics(q, context) for view, context in contexts.items()},
                             'contexts': {view: {'chunk_ids': [c['id'] for c in context], 'e5_tokens': tokens(e5_tok, context_text(context))} for view, context in contexts.items()},
                             'raw_ranking': [{'id': c['id'], 'cosine': c['cosine']} for c in ranked]})
        write(path, {'config': config, 'split': split, 'protocol_sha256': digest(args.output / 'protocol.json'),
                     'manifest_sha256': digest(args.output / 'manifest.json'), 'selection_sha256': selection_hash,
                     'environment': {'python': sys.version, 'platform': platform.platform(), 'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'sentence-transformers', 'numpy')}},
                     'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                     'summary': {language: {view: {'complete': statistics.mean(r['metrics'][view]['complete'] for r in rows if r['language'] == language), 'condition_recovery': statistics.mean(r['metrics'][view]['condition_recovery'] for r in rows if r['language'] == language), 'duplicate_ratio': statistics.mean(r['metrics'][view]['duplicate_ratio'] for r in rows if r['language'] == language)} for view in ('top5', 'budget2000', 'adjacent2000')} for language in sorted({r['language'] for r in rows})},
                     'model': spec, 'batch_size': batch, 'threads': 2, 'device': 'cpu', 'model_load_seconds': load_seconds,
                     'index_build_seconds': index_seconds, 'vector_bytes': int(vectors.nbytes),
                     'index_total_bytes': vector_path.stat().st_size + (args.output / 'chunks' / (config['id'] + '.json')).stat().st_size,
                     'query_timing_scope': 'warm query encode + dense dot product + ranking; excludes context packing', 'index_file_bytes': vector_path.stat().st_size,
                     'vectors_sha256': digest(vector_path), 'max_document_tokens': max(lengths, default=0), 'rows': rows}, exclusive=True)


def row_map(result):
    rows = result['rows']
    mapped = {(r['id'], r['language']): r for r in rows}
    if len(mapped) != len(rows):
        raise ValueError('Duplicate scored query')
    return mapped


def score(result, view='budget2000', language='en'):
    rows = [r for r in result['rows'] if r['language'] == language]
    if not rows:
        raise ValueError('Primary language results missing')
    return (statistics.mean(r['metrics'][view]['complete'] for r in rows),
            statistics.mean(r['metrics'][view]['condition_recovery'] for r in rows),
            -statistics.median(t for r in rows for t in r['query_seconds']))


def no_required_regression(candidate, baseline, view='budget2000'):
    cand, base = row_map(candidate), row_map(baseline)
    if cand.keys() != base.keys():
        raise ValueError('Mismatched scored queries')
    return all(cand[key]['metrics'][view]['complete'] >= row['metrics'][view]['complete'] for key, row in base.items() if row['required'] and row['language'] == 'en')


def choose(selection_results):
    if any(r['split'] != 'selection' for r in selection_results):
        raise ValueError('Holdout must never enter selection')
    by_id = {r['config']['id']: r for r in selection_results}
    baseline = by_id[BASELINE]
    common_base = by_id['e5-common']
    eligible_models = {'e5'} | {r['config']['model'] for r in selection_results if r['config']['family'] == 'common' and no_required_regression(r, common_base)}
    eligible = [r for r in selection_results if r['config']['family'] == 'operational' and r['config']['model'] in eligible_models and no_required_regression(r, baseline)]
    selected = max(eligible, key=lambda r: (score(r), r['config']['id'] == BASELINE))
    return {'selected': selected['config']['id'], 'eligible_models': sorted(eligible_models), 'eligible_configs': [r['config']['id'] for r in eligible],
            'selection_scores': {r['config']['id']: score(r) for r in selection_results}, 'baseline': BASELINE}


def holdout_passes(candidate, baseline):
    if candidate['split'] != 'holdout' or baseline['split'] != 'holdout':
        raise ValueError('Final gate needs holdout results')
    c, b = score(candidate), score(baseline)
    return no_required_regression(candidate, baseline) and c[0] >= b[0] and c[1] >= b[1]


def select(args):
    load_inputs(args)
    manifest = check_manifest(args)
    if (args.output / 'validation-lock.json').exists():
        raise ValueError('Holdout already opened')
    results = [read(result_path(args, c['id'])) for c in manifest['configs']]
    for result in results:
        if result['protocol_sha256'] != digest(args.output / 'protocol.json') or result['manifest_sha256'] != digest(args.output / 'manifest.json'):
            raise ValueError('Results do not match frozen protocol')
    selection = choose(results)
    write(args.output / 'selection.json', {**selection, 'protocol_sha256': digest(args.output / 'protocol.json'),
          'result_hashes': {r['config']['id']: digest(result_path(args, r['config']['id'])) for r in results}}, exclusive=True)


def child(args, model, split):
    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'run', '--cache', str(args.cache), '--pages', str(args.pages),
                    '--dataset', str(args.dataset), '--models', str(args.models), '--output', str(args.output), '--model', model, '--split', split],
                   check=True, env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1'})


def validate(args):
    protocol, _, _, _ = load_inputs(args)
    check_manifest(args)
    selection = read(args.output / 'selection.json')
    for config_id, expected in selection['result_hashes'].items():
        if digest(result_path(args, config_id)) != expected:
            raise ValueError('Selection evidence changed')
    lock_path = args.output / 'validation-lock.json'
    lock = {'selection_sha256': digest(args.output / 'selection.json')}
    if (args.output / 'decision.json').exists():
        raise ValueError('Validation is complete; holdout cannot be reopened')
    if lock_path.exists():
        if read(lock_path) != lock:
            raise ValueError('Cannot resume validation with changed selection')
    else:
        write(lock_path, lock, exclusive=True)
    selected = selection['selected']
    configs = {c['id']: c for c in check_manifest(args)['configs']}
    for model in sorted({configs[c]['model'] for c in (BASELINE, selected)}):
        child(args, model, 'holdout')
    baseline = read(result_path(args, BASELINE, 'holdout'))
    candidate = read(result_path(args, selected, 'holdout'))
    passed = holdout_passes(candidate, baseline)
    write(args.output / 'decision.json', {'selected_before_validation': selected, 'holdout_passed': passed,
          'retained': selected if passed else BASELINE, 'baseline_holdout_score': score(baseline), 'candidate_holdout_score': score(candidate),
          'operational_change_authorized': protocol['corpus'] == 'technical' and passed,
          'review_status': 'human_review_pending', 'corpus': protocol['corpus'],
          'note': 'Human semantic review remains pending; enterprise evidence alone cannot change production.'}, exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'prepare', 'run', 'run-all', 'select', 'validate'])
    parser.add_argument('--cache', type=Path, default=Path('.cache/embedding-benchmark'))
    parser.add_argument('--pages', type=Path, default=Path('data/pages.json'))
    parser.add_argument('--dataset', type=Path, default=Path('eval/retrieval-technical.json'))
    parser.add_argument('--models', type=Path, default=Path('eval/retrieval-models.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/retrieval-experiments'))
    parser.add_argument('--model', choices=['e5', 'bge-small', 'bge-m3'])
    parser.add_argument('--split', choices=['selection', 'holdout'], default='selection')
    args = parser.parse_args()
    if args.command == 'run-all':
        if args.split != 'selection':
            parser.error('Use validate for holdout')
        for model in ('e5', 'bge-small', 'bge-m3'):
            child(args, model, 'selection')
    elif args.command == 'run':
        if not args.model:
            parser.error('run requires --model')
        run_model(args)
    else:
        globals()[args.command](args)


if __name__ == '__main__':
    main()
