#!/usr/bin/env python3
"""Offline retrieval regression over previously exposed questions, NOT a new holdout.

No LLM, source download, environment-file loading, or automatic baseline promotion.
CLI: evaluate -> review candidate -> approve-baseline --candidate-sha256 SHA -> check.
The app can call ensure_retrieval_regression with its already-built Corpus.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.context import build_research_context, policy_identity
from rag.corpus import Corpus, EMBEDDING_PROFILES
from rag.settings import Settings
from scripts.retrieval_experiments import metrics, validate_dataset

CODE_FILES = ('rag/corpus.py', 'rag/context.py', 'rag/graph.py',
              'scripts/retrieval_experiments.py', 'scripts/revalidate_retrieval.py')
K_VALUES = (3, 5, 8, 10)
NOTE = ('Known-question regression, including previously exposed selection and holdout questions; '
        'not independent generalization evidence. English is the gate; Korean is diagnostic. '
        'Single-query probes use the production search and research-context policy; '
        'this does not measure generated query quality or downstream answer correctness.')


class RetrievalRegressionError(ValueError):
    pass


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def seal(value):
    value = {k: v for k, v in value.items() if k != 'record_sha256'}
    return {**value, 'record_sha256': sha_bytes(canonical(value))}


def verify(value):
    if value.get('record_sha256') != seal(value)['record_sha256']:
        raise RetrievalRegressionError('Evaluation record changed or lacks integrity hash')


def identity(corpus, config, dataset_path, root=Path('.')):
    """Bind resolved model, effective policy, source text, questions and executing code."""
    dataset = read(dataset_path)
    validate_dataset(dataset, corpus.pages)
    if not dataset['questions'] or any('en' not in q['queries'] for q in dataset['questions']):
        raise RetrievalRegressionError('Every regression question requires English ground truth')
    top_k = config.get('top_k')
    if type(top_k) is not int or top_k < 1:
        raise RetrievalRegressionError('top_k must be a positive integer')
    packages = {}
    for name in ('numpy', 'sentence-transformers', 'transformers', 'torch'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'unavailable'
    return {'schema': 1, 'embedding': json.loads(canonical(corpus.embedding)),
            'config': {k: config[k] for k in ('chunk_tokens', 'overlap_tokens', 'top_k')},
            'context': policy_identity(config), 'k_values': sorted(set(K_VALUES + (top_k,))),
            'pages_sha256': sha_bytes(canonical(corpus.pages)),
            'questions_sha256': sha_bytes(Path(dataset_path).read_bytes()),
            'chunks_sha256': sha_bytes(canonical(corpus.chunks)),
            'vectors_sha256': sha_bytes(corpus.vectors.tobytes()),
            'code_sha256': {name: sha_bytes((Path(root) / name).read_bytes()) for name in CODE_FILES},
            'packages': packages}


def evaluate(corpus, config, dataset_path, *, root=Path('.'), current_identity=None):
    ident = current_identity or identity(corpus, config, dataset_path, root)
    dataset = read(dataset_path)
    rows = []
    for question in dataset['questions']:
        for language, query in question['queries'].items():
            for k in ident['k_values']:
                start = time.perf_counter()
                hits = corpus.search(query, question['technology'], k)
                context = build_research_context(corpus, hits, question['technology'], config)
                elapsed = time.perf_counter() - start
                convert = lambda cs: [dict(c, start=c['char_start'], end=c['char_end']) for c in cs]
                rows.append({'id': question['id'], 'language': language, 'k': k,
                             'required': question.get('required', False),
                             'original_split': question['split'], 'query': query,
                             'retrieval_seconds': elapsed,
                             'top_k': metrics(question, convert(hits)),
                             'context': metrics(question, convert(context)),
                             'context_tokens': sum(corpus.evidence_tokens(c['text']) for c in context),
                             'hit_ids': [c['id'] for c in hits], 'context_ids': [c['id'] for c in context]})
    summary = {}
    for language in sorted({r['language'] for r in rows}):
        summary[language] = {}
        for k in ident['k_values']:
            group = [r for r in rows if r['language'] == language and r['k'] == k]
            summary[language][str(k)] = {'questions': len(group), **{
                view: {metric: statistics.mean(r[view][metric] for r in group)
                       for metric in ('complete', 'condition_recovery', 'duplicate_ratio')}
                for view in ('top_k', 'context')},
                'mean_context_tokens': statistics.mean(r['context_tokens'] for r in group),
                'mean_retrieval_seconds': statistics.mean(r['retrieval_seconds'] for r in group)}
    return seal({'schema': 1, 'status': 'review_required', 'note': NOTE,
                 'measured_at': datetime.now(timezone.utc).isoformat(), 'identity': ident,
                 'fingerprint': sha_bytes(canonical(ident)), 'rows': rows, 'summary': summary})


def approve_baseline(candidate_path, baseline_path, expected_sha256):
    """Explicit promotion, bound to the exact candidate bytes the caller reviewed."""
    if Path(baseline_path).exists():
        raise RetrievalRegressionError('Baseline exists; use a new explicit versioned baseline path')
    if sha_bytes(Path(candidate_path).read_bytes()) != expected_sha256:
        raise RetrievalRegressionError('Candidate hash does not match the reviewed artifact')
    candidate = read(candidate_path)
    verify(candidate)
    if candidate.get('status') != 'review_required' or not candidate.get('rows'):
        raise RetrievalRegressionError('Only an explicitly reviewed measurement candidate can become baseline')
    baseline = seal({**candidate, 'status': 'baseline', 'approved_candidate_sha256': expected_sha256,
                     'approval_note': 'Explicit CLI baseline freeze; source semantics remain human_review_pending'})
    write(baseline_path, baseline)
    return baseline


def compare(candidate, baseline):
    verify(candidate)
    verify(baseline)
    if baseline.get('status') != 'baseline' or not baseline.get('approved_candidate_sha256'):
        raise RetrievalRegressionError('An explicitly frozen baseline is required')
    for key in ('pages_sha256', 'questions_sha256'):
        if baseline['identity'][key] != candidate['identity'][key]:
            raise RetrievalRegressionError('Sources/questions changed; review and freeze a new baseline')
    def operational_rows(value):
        k = value['identity']['config']['top_k']
        return {r['id']: r for r in value['rows'] if r['language'] == 'en' and r['k'] == k}
    current, prior = operational_rows(candidate), operational_rows(baseline)
    if not current or set(current) != set(prior):
        raise RetrievalRegressionError('Regression question identities differ')
    regressions = []
    for qid, row in current.items():
        for view in ('top_k', 'context'):
            for metric in ('complete', 'condition_recovery'):
                before, after = prior[qid][view][metric], row[view][metric]
                if after + 1e-12 < before:
                    regressions.append({'id': qid, 'required': row['required'], 'view': view,
                                        'metric': metric, 'baseline': before, 'current': after})
    # Protect each question, not merely a mean that could hide regressions.
    return {'status': 'failed' if regressions else 'passed', 'regressions': regressions,
            'rule': 'Every English question: top-k and operational-context complete/condition non-regression'}


def ensure_retrieval_regression(corpus, config, *, baseline_path, result_path,
                               dataset_path=Path('eval/retrieval-technical.json'), root=Path('.')):
    """Raise before paid generation when baseline is missing, stale, or regressed.

    Stale cached evaluations are recomputed; stale baselines for changed ground truth
    require explicit review. A matching passed result reuses the original evidence.
    """
    if Path(result_path).resolve() == Path(baseline_path).resolve():
        raise RetrievalRegressionError('Baseline and result paths must be different')
    if not Path(baseline_path).is_file():
        raise RetrievalRegressionError('No reviewed baseline; evaluate and explicitly freeze one first')
    baseline = read(baseline_path)
    verify(baseline)
    baseline_sha = sha_bytes(Path(baseline_path).read_bytes())
    ident = identity(corpus, config, dataset_path, root)
    fingerprint = sha_bytes(canonical(ident))
    if Path(result_path).exists():
        cached = read(result_path)
        verify(cached)
        if cached.get('fingerprint') == fingerprint and cached.get('baseline_sha256') == baseline_sha:
            decision = compare(cached, baseline)
            if decision['status'] != 'passed':
                raise RetrievalRegressionError('Cached retrieval regression failed')
            return {**cached, 'reused': True}
    candidate = evaluate(corpus, config, dataset_path, root=root, current_identity=ident)
    try:
        decision = compare(candidate, baseline)
    except RetrievalRegressionError as exc:
        write(result_path, seal({**candidate, 'status': 'blocked', 'baseline_sha256': baseline_sha,
                                'reason': str(exc)}))
        raise
    result = seal({**candidate, **decision, 'baseline_sha256': baseline_sha})
    write(result_path, result)
    if result['status'] != 'passed':
        raise RetrievalRegressionError(f'Retrieval regression failed: {len(result["regressions"])} metric drops; see result file')
    return {**result, 'reused': False}


class OfflineCorpus(Corpus):
    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer
        model = self.settings.get('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small')
        if model not in EMBEDDING_PROFILES:
            raise RetrievalRegressionError('Model lacks a production embedding profile')
        profile = EMBEDDING_PROFILES[model]
        revision = self.settings.get('EMBEDDING_REVISION') or profile['revision']
        cache = Path(self.settings.get('HF_HOME')) / 'hub'
        snapshot = cache / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
        if not snapshot.is_dir():
            raise RetrievalRegressionError('Local model snapshot missing; downloads are forbidden')
        self.encoder = SentenceTransformer(str(snapshot), device='cpu', local_files_only=True, trust_remote_code=False)
        if self.encoder.max_seq_length < profile['max_tokens']:
            raise RetrievalRegressionError('Encoder limit differs from production profile')
        self.embedding = {'model': model, 'revision': revision, 'dimension': self.encoder.get_embedding_dimension(),
                          'max_tokens': profile['max_tokens'], 'normalized': True,
                          'prefixes': [profile['query_prefix'], profile['passage_prefix']]}
        e5 = EMBEDDING_PROFILES['intfloat/multilingual-e5-small']
        budget_snapshot = cache / 'models--intfloat--multilingual-e5-small' / 'snapshots' / e5['revision']
        self.budget_tokenizer = AutoTokenizer.from_pretrained(str(budget_snapshot), local_files_only=True, trust_remote_code=False)



def load_offline_corpus(config, *, pages_path=Path('data/pages.json'),
                        cache_path=Path('.cache/huggingface/hub'),
                        index_path=Path('.cache/retrieval-regression'),
                        model='intfloat/multilingual-e5-small', revision=''):
    """No credentials or network: caller supplies resolved operating model/revision."""
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    torch.set_num_threads(2)
    settings = Settings({'EMBEDDING_MODEL': model, 'EMBEDDING_REVISION': revision,
                         'HF_HOME': str(Path(cache_path).parent), 'RAG_INDEX_DIR': str(index_path)})
    corpus = OfflineCorpus(settings, config)
    corpus.pages = read(pages_path)
    corpus.build()  # Production chunking/index; never prepare_sources/download.
    return corpus


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('evaluate', 'check', 'approve-baseline'))
    parser.add_argument('--config', type=Path, default=Path('config/run.yaml'))
    parser.add_argument('--pages', type=Path, default=Path('data/pages.json'))
    parser.add_argument('--dataset', type=Path, default=Path('eval/retrieval-technical.json'))
    parser.add_argument('--cache', type=Path, default=Path('.cache/huggingface/hub'))
    parser.add_argument('--index', type=Path, default=Path('.cache/retrieval-regression'))
    parser.add_argument('--model', default='intfloat/multilingual-e5-small')
    parser.add_argument('--revision', default='')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate-sha256')
    args = parser.parse_args(argv)
    try:
        if args.command == 'approve-baseline':
            if not args.baseline or not args.candidate_sha256:
                parser.error('approve-baseline requires --baseline and --candidate-sha256')
            approve_baseline(args.output, args.baseline, args.candidate_sha256)
            print(json.dumps({'status': 'baseline_frozen', 'path': str(args.baseline)}))
            return 0
        if args.command == 'check' and not args.baseline:
            parser.error('check requires --baseline')
        import yaml
        config = yaml.safe_load(args.config.read_text())
        corpus = load_offline_corpus(config, pages_path=args.pages, cache_path=args.cache,
                                     index_path=args.index, model=args.model, revision=args.revision)
        if args.command == 'evaluate':
            result = evaluate(corpus, config, args.dataset)
            write(args.output, result)
            print(json.dumps({'status': result['status'], 'candidate_sha256': sha_bytes(args.output.read_bytes()),
                              'summary': result['summary']}, ensure_ascii=False))
        else:
            result = ensure_retrieval_regression(corpus, config, baseline_path=args.baseline,
                                                result_path=args.output, dataset_path=args.dataset)
            print(json.dumps({'status': result['status'], 'reused': result['reused']}))
        return 0
    except (ValueError, OSError) as exc:
        print(json.dumps({'status': 'blocked', 'reason': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
