import json
from pathlib import Path
from urllib.request import Request
import pytest
from rag.source_discovery import Discovery, CheckedRedirect, allowed, sha, links
from rag.corpus import Corpus
from rag.settings import Settings
from rag.graph import token_context, BASE

POLICY = {'allow': [{'host': 'official.test', 'paths': ['/docs/']}]}
CFG = dict(depth=1, per_source=3, total=6, file_bytes=5000000, aggregate_bytes=20000000,
           timeout=30, sources={'official': POLICY})


def test_discovery_ignores_navigation_images_and_binary_downloads():
    text = b'<nav><a href="install">Install</a></nav><main><a href="cache">Cache</a></main> ![plot](plot.png) [paper](paper.pdf) [archive](code.zip)'
    assert links(text, 'https://official.test/docs/start') == [
        'https://official.test/docs/cache', 'https://official.test/docs/paper.pdf']


def test_discovery_start_cache_uses_its_own_hash_when_base_snapshot_is_older(tmp_path):
    source = {'url': 'https://official.test/docs/start', 'raw_sha256': sha(b'old registered snapshot')}
    (tmp_path / 'official.discovery-start.raw').write_bytes(b'[new](new.md)')
    (tmp_path / 'official.discovery-start.json').write_text(json.dumps({
        'url': source['url'], 'final_url': source['url'], 'raw_sha256': sha(b'[new](new.md)')}))
    d = Discovery(CFG, tmp_path, lambda url, *a: (b'new official evidence', url))
    sources = d.collect({'official': source})
    assert len(sources) == 1
    assert next(iter(sources.values()))['url'].endswith('new.md')

@pytest.mark.parametrize('url', ['http://official.test/docs/a', 'https://evil.test/docs/a',
    'https://official.test:444/docs/a', 'https://user@official.test/docs/a',
    'https://official.test/docs2/a', 'https://official.test/docs/%2e%2e/secret',
    'https://official.test/docs/a?secret=1', 'https://official.test:bad/docs/a'])
def test_boundary_rejects_initial_and_redirect_before_follow(url):
    assert not allowed(url, POLICY)
    with pytest.raises(ValueError):
        CheckedRedirect(POLICY).redirect_request(Request('https://official.test/docs/start'), None, 302, '', {}, url)


def collect(tmp_path, fetcher, **changes):
    root = tmp_path / 'web'
    root.mkdir(exist_ok=True)
    raw_path = root / 'start.raw'
    raw_path.write_bytes(b'[one](a.md) [duplicate](a.md) [two](b.md) [three](c.md) [four](d.md)')
    source = dict(url='https://official.test/docs/start', raw_path=str(raw_path), raw_sha256=sha(raw_path.read_bytes()), final_url='https://official.test/docs/start', authors='official')
    discovery = Discovery({**CFG, **changes}, root, fetcher)
    return discovery, discovery.collect({'official': source})


def test_bounded_deduplicated_raw_text_hash_and_cache(tmp_path):
    calls = []
    def fetcher(url, policy, limit, timeout):
        calls.append(url)
        return b'Authentic source evidence. [deeper](deep.md)', url
    first, sources = collect(tmp_path, fetcher)
    assert len(sources) == len(calls) == 3
    assert not any('deep' in url for url in calls)
    for source in sources.values():
        assert Path(source['raw_path']).read_bytes().startswith(b'Authentic')
        assert source['raw_sha256'] == source['text_sha256']
        assert source['discovered_from'].endswith('start') and source['accessed_at']
    second, cached = collect(tmp_path, lambda *a: pytest.fail('cache must avoid network'))
    assert all(s['cache_status'] == 'hit' for s in cached.values())
    path = Path(next(iter(cached.values()))['raw_path'])
    path.write_bytes(b'tampered')
    third, recovered = collect(tmp_path, lambda *a: pytest.fail('corrupt cache must not fetch silently'))
    assert len(recovered) == 2
    assert any(e['status'] == 'failed' for e in third.events)


def test_oversize_and_redirect_results_are_not_saved(tmp_path):
    discovery, sources = collect(tmp_path, lambda url, *a: (b'x' * 11, url), file_bytes=10)
    assert not sources
    discovery, sources = collect(tmp_path, lambda url, *a: (b'text', 'https://evil.test/docs/a'))
    assert not sources


def test_global_limit_and_frozen_identity(tmp_path):
    _, sources = collect(tmp_path, lambda url, *a: (b'evidence', url), total=2)
    assert len(sources) == 2
    corpus = Corpus(Settings({}), dict(chunk_tokens=380, overlap_tokens=50))
    corpus.sources = sources
    old = corpus.retrieval_identity()
    corpus.config['chunk_tokens'] = 200
    assert old != corpus.retrieval_identity()
    corpus.freeze_sources()
    with pytest.raises(ValueError, match='frozen'):
        corpus.prepare_sources()


def test_neighbors_reserved_deduplicated_and_token_limited():
    chunks = [dict(id=str(i), text='word ' * 3) for i in range(5)]
    selected = token_context(chunks, [chunks[4], chunks[4]], lambda text: len(text.split()), 9, 3)
    assert [c['id'] for c in selected] == ['4', '0', '1']
    assert sum(len(c['text'].split()) for c in selected) == 9
    assert '검색 미확인과 원문 부재는 다르다' in BASE
    assert '표·그림의 수치와 그 실험을 설명하는 앞뒤 문맥을 연결' in BASE
    assert '논문에서 확인한 실험과 목표 업무의 검증 상태를 구분' in BASE


def test_target_condition_neighbor_pages_have_reserved_candidate_priority():
    corpus = Corpus(Settings({}), {})
    corpus.chunks = [dict(id=f'p{p}', source_id='infinigen', page=p, text=f'Full source page {p}') for p in range(1, 13)]
    hits = [corpus.chunks[p - 1] for p in (10, 11, 2, 9)]
    adjacent = corpus.adjacent_candidates(hits)
    selected = token_context(hits, adjacent, lambda _: 1, 4, 2)
    assert {11, 12} <= {c['page'] for c in adjacent}
    assert sum(1 for c in selected) <= 4
    corpus.chunks = [dict(id=f'k{p}', source_id='kivi', page=p, text=f'Full source page {p}') for p in range(1, 9)]
    adjacent = corpus.adjacent_candidates([corpus.chunks[5], corpus.chunks[6]])
    assert {7, 8} <= {c['page'] for c in adjacent}


def test_mutation_of_frozen_source_registry_is_rejected_before_search():
    corpus = Corpus(Settings({}), {})
    corpus.freeze_sources()
    corpus.sources['late_source'] = {'url': 'https://official.test/docs/late'}
    with pytest.raises(ValueError, match='Frozen'):
        corpus.web_search('late')


def test_failed_overlimit_reads_keep_reservations_within_aggregate(tmp_path):
    consumed = []
    def overlimit(url, policy, limit, timeout):
        consumed.append(limit + 1)
        raise ValueError('over limit after partial transfer')
    d = Discovery(CFG, tmp_path, overlimit)
    for _ in range(10):
        with pytest.raises(ValueError):
            d._fetch('https://official.test/docs/a', POLICY)
    assert sum(consumed) == d.used == 20_000_000
    assert len(consumed) == 4


def test_generation_reads_frozen_snapshot_not_mutated_disk(tmp_path):
    _, sources = collect(tmp_path, lambda url, *a: (b'original evidence', url), total=1)
    corpus = Corpus(Settings({}), {})
    corpus.sources = sources
    corpus.freeze_sources()
    source = next(iter(sources.values()))
    path = Path(source['local_path'])
    snapshot = json.loads(path.read_text())
    snapshot['text'] = 'mutated evidence'
    snapshot['text_sha256'] = sha(snapshot['text'].encode())
    path.write_text(json.dumps(snapshot))
    Path(source['raw_path']).write_bytes(b'mutated raw')
    assert corpus.web_search('original')[0]['text'] == 'original evidence'
    assert corpus.web_search('mutated') == []


def test_initial_cached_raw_hash_and_size_are_checked(tmp_path):
    raw_path = tmp_path / 'initial.raw'
    raw_path.write_bytes(b'[link](a.md)')
    source = {'url': 'https://official.test/docs/start', 'raw_path': str(raw_path), 'raw_sha256': 'wrong'}
    for cfg in (CFG, {**CFG, 'file_bytes': 2}):
        d = Discovery(cfg, tmp_path, lambda *a: pytest.fail('bad cache must not use network'))
        assert not d.collect({'official': source})
        assert d.events[-1]['status'] == 'failed'


def test_start_redirect_relative_links_are_identical_on_cache_reuse(tmp_path):
    source = {'url': 'https://official.test/docs/start'}
    def fetcher(url, *args):
        if url == source['url']:
            return b'[child](child.md)', 'https://official.test/docs/v2/start'
        assert url == 'https://official.test/docs/v2/child.md'
        return b'official child evidence', url
    first = Discovery(CFG, tmp_path, fetcher).collect({'official': source})
    second = Discovery(CFG, tmp_path, lambda *a: pytest.fail('verified cache must avoid fetch')).collect({'official': source})
    assert {s['url'] for s in first.values()} == {s['url'] for s in second.values()} == {'https://official.test/docs/v2/child.md'}


@pytest.mark.parametrize('registered', [False, True])
def test_start_cache_rejects_out_of_scope_final_url(tmp_path, registered):
    raw = b'[child](child.md)'
    path = tmp_path / 'official.discovery-start.raw'
    path.write_bytes(raw)
    metadata = {'url': 'https://official.test/docs/start', 'raw_sha256': sha(raw), 'final_url': 'https://evil.test/docs/start'}
    if registered:
        source = {**metadata, 'raw_path': str(path)}
        path.rename(tmp_path / 'registered.raw')
        source['raw_path'] = str(tmp_path / 'registered.raw')
    else:
        source = {'url': metadata['url']}
        (tmp_path / 'official.discovery-start.json').write_text(json.dumps(metadata))
    d = Discovery(CFG, tmp_path, lambda *a: pytest.fail('invalid final URL must not fetch'))
    assert d.collect({'official': source}) == {}
    assert d.events[-1]['status'] == 'failed'


@pytest.mark.parametrize('registered', [False, True])
def test_legacy_start_cache_without_final_url_refetches_before_resolving(tmp_path, registered):
    raw = b'[incorrect legacy link](wrong.md)'
    path = tmp_path / ('registered.raw' if registered else 'official.discovery-start.raw')
    path.write_bytes(raw)
    source = {'url': 'https://official.test/docs/start'}
    metadata = {**source, 'raw_sha256': sha(raw)}
    if registered:
        source.update(metadata, raw_path=str(path))
    else:
        (tmp_path / 'official.discovery-start.json').write_text(json.dumps(metadata))
    calls = []
    def fetcher(url, *args):
        calls.append(url)
        return (b'[child](child.md)', 'https://official.test/docs/v2/start') if url == source['url'] else (b'evidence', url)
    first = Discovery(CFG, tmp_path, fetcher).collect({'official': source})
    assert calls == [source['url'], 'https://official.test/docs/v2/child.md']
    second = Discovery(CFG, tmp_path, lambda *a: pytest.fail('upgraded cache must avoid fetch')).collect({'official': source})
    assert {s['url'] for s in first.values()} == {s['url'] for s in second.values()}


def test_registered_raw_cache_uses_verified_final_url(tmp_path):
    path = tmp_path / 'registered.raw'
    raw = b'[child](child.md)'
    path.write_bytes(raw)
    source = dict(url='https://official.test/docs/start', final_url='https://official.test/docs/v2/start',
                  raw_path=str(path), raw_sha256=sha(raw))
    calls = []
    d = Discovery(CFG, tmp_path, lambda url, *a: (calls.append(url) or b'evidence', url))
    d.collect({'official': source})
    assert calls == ['https://official.test/docs/v2/child.md']


def test_refresh_refetches_start_links_and_final_url(tmp_path):
    source = {'url': 'https://official.test/docs/start'}
    def initial(url, *args):
        return (b'[old](old.md)', 'https://official.test/docs/v1/start') if url == source['url'] else (b'old evidence', url)
    Discovery(CFG, tmp_path, initial).collect({'official': source})
    calls = []
    def refreshed(url, *args):
        calls.append(url)
        return (b'[new](new.md)', 'https://official.test/docs/v2/start') if url == source['url'] else (b'new evidence', url)
    result = Discovery(CFG, tmp_path, refreshed).collect({'official': source}, refresh=True)
    assert calls == [source['url'], 'https://official.test/docs/v2/new.md']
    metadata = json.loads((tmp_path / 'official.discovery-start.json').read_text())
    assert metadata['final_url'] == 'https://official.test/docs/v2/start'
    assert metadata['raw_sha256'] == sha(b'[new](new.md)')
    cached = Discovery(CFG, tmp_path, lambda *a: pytest.fail('cache must remain offline')).collect({'official': source})
    assert {s['url'] for s in result.values()} == {s['url'] for s in cached.values()}
