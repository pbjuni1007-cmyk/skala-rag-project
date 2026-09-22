from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts import revalidate_retrieval as r

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'eval/retrieval-technical.json'


@pytest.fixture
def inputs():
    config = yaml.safe_load((ROOT / 'config/run.yaml').read_text())
    corpus = SimpleNamespace(pages=r.read(ROOT / 'data/pages.json'), chunks=[],
                             vectors=np.array([[1.0, 0.0]]),
                             embedding={'model': 'test-model', 'revision': 'fixed'})
    return corpus, config


def candidate(ident, score=1):
    rows = [{'id': qid, 'language': 'en', 'k': ident['config']['top_k'], 'required': True,
             'top_k': {'complete': score, 'condition_recovery': score},
             'context': {'complete': score, 'condition_recovery': score}}
            for qid in ('K1', 'I1')]
    return r.seal({'status': 'review_required', 'identity': ident,
                   'fingerprint': r.sha_bytes(r.canonical(ident)), 'rows': rows})


def baseline(tmp_path, ident):
    path, target = tmp_path / 'candidate.json', tmp_path / 'baseline.json'
    r.write(path, candidate(ident))
    r.approve_baseline(path, target, r.sha_bytes(path.read_bytes()))
    return target


@pytest.mark.parametrize('mutation', ['model', 'revision', 'chunk', 'overlap', 'top_k', 'context', 'vectors', 'code'])
def test_identity_tracks_operating_changes(inputs, tmp_path, mutation):
    corpus, config = inputs
    before = r.identity(corpus, config, DATASET, ROOT)
    if mutation in ('model', 'revision'):
        corpus.embedding[mutation] = 'changed'
    elif mutation in ('chunk', 'overlap'):
        config[{'chunk': 'chunk_tokens', 'overlap': 'overlap_tokens'}[mutation]] += 1
    elif mutation == 'top_k':
        config['top_k'] = 8
    elif mutation == 'context':
        config['assessment_evidence_token_budget'] += 1
    elif mutation == 'vectors':
        corpus.vectors[0, 0] = .5
    else:
        for name in r.CODE_FILES:
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes() + b'\n# changed\n')
    after = r.identity(corpus, config, DATASET, tmp_path if mutation == 'code' else ROOT)
    assert before != after


def test_no_baseline_never_measures_or_passes(inputs, tmp_path, monkeypatch):
    corpus, config = inputs
    monkeypatch.setattr(r, 'evaluate', lambda *a, **kw: pytest.fail('must not auto-bootstrap'))
    with pytest.raises(r.RetrievalRegressionError, match='No reviewed baseline'):
        r.ensure_retrieval_regression(corpus, config, baseline_path=tmp_path/'none', result_path=tmp_path/'out', dataset_path=DATASET, root=ROOT)


def test_baseline_requires_exact_reviewed_hash_and_never_overwrites(inputs, tmp_path):
    corpus, config = inputs
    value = candidate(r.identity(corpus, config, DATASET, ROOT))
    path = tmp_path/'candidate.json'
    r.write(path, value)
    with pytest.raises(r.RetrievalRegressionError, match='hash'):
        r.approve_baseline(path, tmp_path/'baseline.json', 'incorrect')
    r.approve_baseline(path, tmp_path/'baseline.json', r.sha_bytes(path.read_bytes()))
    with pytest.raises(r.RetrievalRegressionError, match='exists'):
        r.approve_baseline(path, tmp_path/'baseline.json', r.sha_bytes(path.read_bytes()))


def test_unchanged_reuses_and_changed_topk_remeasures(inputs, tmp_path, monkeypatch):
    corpus, config = inputs
    target = baseline(tmp_path, r.identity(corpus, config, DATASET, ROOT))
    calls = []
    def measured(*args, **kwargs):
        calls.append(1)
        return candidate(kwargs['current_identity'])
    monkeypatch.setattr(r, 'evaluate', measured)
    kwargs = dict(baseline_path=target, result_path=tmp_path/'result.json', dataset_path=DATASET, root=ROOT)
    assert r.ensure_retrieval_regression(corpus, config, **kwargs)['reused'] is False
    assert r.ensure_retrieval_regression(corpus, config, **kwargs)['reused'] is True
    config['top_k'] = 8
    assert r.ensure_retrieval_regression(corpus, config, **kwargs)['reused'] is False
    assert len(calls) == 2


def test_drop_is_persisted_and_cached_failure_cannot_pass(inputs, tmp_path, monkeypatch):
    corpus, config = inputs
    target = baseline(tmp_path, r.identity(corpus, config, DATASET, ROOT))
    monkeypatch.setattr(r, 'evaluate', lambda *a, **kw: candidate(kw['current_identity'], score=0))
    kwargs = dict(baseline_path=target, result_path=tmp_path/'result.json', dataset_path=DATASET, root=ROOT)
    for _ in range(2):
        with pytest.raises(r.RetrievalRegressionError, match='failed'):
            r.ensure_retrieval_regression(corpus, config, **kwargs)
    result = r.read(tmp_path/'result.json')
    assert result['status'] == 'failed' and len(result['regressions']) == 8


def test_tampered_cached_results_are_not_reused(inputs, tmp_path, monkeypatch):
    corpus, config = inputs
    target = baseline(tmp_path, r.identity(corpus, config, DATASET, ROOT))
    monkeypatch.setattr(r, 'evaluate', lambda *a, **kw: candidate(kw['current_identity']))
    kwargs = dict(baseline_path=target, result_path=tmp_path/'result.json', dataset_path=DATASET, root=ROOT)
    r.ensure_retrieval_regression(corpus, config, **kwargs)
    changed = r.read(tmp_path/'result.json'); changed['rows'][0]['context']['complete'] = 0
    r.write(tmp_path/'result.json', changed)
    with pytest.raises(r.RetrievalRegressionError, match='integrity'):
        r.ensure_retrieval_regression(corpus, config, **kwargs)


def test_changed_questions_require_baseline_review(inputs, tmp_path):
    corpus, config = inputs
    ident = r.identity(corpus, config, DATASET, ROOT)
    base = r.read(baseline(tmp_path, ident))
    changed = deepcopy(ident); changed['questions_sha256'] = 'new'
    with pytest.raises(r.RetrievalRegressionError, match='Sources/questions changed'):
        r.compare(candidate(changed), base)


def test_evaluate_runs_every_k_and_uses_operational_context(inputs, monkeypatch):
    corpus, config = inputs
    corpus.chunks = [dict(p, id=str(i), char_start=0, char_end=len(p['text'])) for i,p in enumerate(corpus.pages)]
    calls, context_calls = [], []
    def search(query, technology, k):
        calls.append(k)
        return [c for c in corpus.chunks if c['technology'] == technology][:k]
    corpus.search = search
    corpus.evidence_tokens = lambda text: len(text.split())
    def context(c, hits, tech, cfg):
        assert c is corpus and cfg is config
        context_calls.append(tech)
        return hits
    monkeypatch.setattr(r, 'build_research_context', context)
    result = r.evaluate(corpus, config, DATASET, root=ROOT)
    assert set(calls) == {3,5,8,10}
    assert len(calls) == len(context_calls) == 24 * 2 * 4
    assert result['status'] == 'review_required'
    assert 'not independent' in result['note']
    assert set(result['summary']['en']) == {'3','5','8','10'}


def test_missing_snapshot_never_downloads(tmp_path):
    corpus = r.OfflineCorpus(r.Settings({'HF_HOME': str(tmp_path)}), {})
    with pytest.raises(r.RetrievalRegressionError, match='downloads are forbidden'):
        corpus.load_encoder()


def test_app_blocks_before_paid_gateway_when_gate_fails(monkeypatch, tmp_path):
    import app
    from scripts.revalidate_retrieval import RetrievalRegressionError
    from rag.settings import Settings
    import rag.corpus
    import rag.llm
    calls = []
    class FakeCorpus:
        def __init__(self, *args): pass
        def prepare_sources(self, *args): return []
        def build(self): return {}
    monkeypatch.setattr(rag.corpus, 'Corpus', FakeCorpus)
    monkeypatch.setattr(Settings, 'load', classmethod(lambda cls: Settings({})))
    monkeypatch.setattr(app.sys, 'argv', ['app.py'])
    def blocked(*args, **kwargs):
        calls.append('gate')
        raise RetrievalRegressionError('review needed')
    monkeypatch.setattr('scripts.revalidate_retrieval.ensure_retrieval_regression', blocked)
    monkeypatch.setattr(rag.llm, 'Gateway', lambda *args: calls.append('paid'))
    with pytest.raises(RetrievalRegressionError):
        app.main()
    assert calls == ['gate']
