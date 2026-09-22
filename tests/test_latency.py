"""Deterministic concurrency/telemetry contracts; never contact a provider."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from threading import Barrier, Event, Lock, enumerate as threads
import urllib.error

import pytest

from rag.llm import APIError, Gateway
from test_gateway import settings, completed, no_network_or_retry_wait
from test_facet_reassessment import setup
from rag.reassessment import reassess_facets
from rag.graph import BASE


@pytest.mark.parametrize('purpose,effort', [
    ('research_queries', 'low'), ('market_rewrite', 'low'),
    ('retrieval_review_kivi_0', 'low'), ('research_kivi_0', 'medium'),
    ('market', 'medium'), ('stakeholder', 'medium'), ('domain', 'medium'),
    ('market_reassessment_facet_costs', 'medium'),
    ('synthesis_report', 'max'), ('synthesis_gaps_0', 'max'),
    ('synthesis_report_repair', 'low'), ('synthesis_gaps_0_reference_repair', 'low'),
    ('unrecognized', 'max'),
])
def test_effective_role_routing(settings, tmp_path, purpose, effort):
    gateway = Gateway(settings, 'routing', tmp_path)
    assert gateway.build_payload(purpose, 'instruction', 'input')['reasoning']['effort'] == effort
    assert settings.public()['LLM_REASONING_PROFILE'] == 'balanced'
    balanced_hash = gateway.payload_hash(gateway.build_payload(purpose, 'i', 'x'))
    settings.values['LLM_REASONING_PROFILE'] = 'fixed'
    fixed = gateway.build_payload(purpose, 'i', 'x')
    assert fixed['reasoning']['effort'] == 'max'
    assert (balanced_hash == gateway.payload_hash(fixed)) == (effort == 'max')


def test_nested_pools_share_global_generation_cap_and_keep_telemetry(settings, tmp_path, monkeypatch, capsys):
    gateway = Gateway(settings, 'nested', tmp_path / 'out')
    lock, first_wave, release = Lock(), Event(), Event()
    active = peak = 0
    def request(path, payload):
        nonlocal active, peak
        if path == 'responses/input_tokens':
            return {'input_tokens': 100}
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 3:
                first_wave.set()
        assert release.wait(5)
        with lock:
            active -= 1
        result = completed()
        result['usage']['output_tokens_details'] = {'reasoning_tokens': 150}
        return result
    monkeypatch.setattr(gateway, 'request', request)
    def perspective(index):
        with ThreadPoolExecutor(max_workers=3) as inner:
            return list(inner.map(lambda facet: gateway.generate(f'domain_{index}_{facet}', 'SECRET INSTRUCTION', str(facet)), range(3)))
    with ThreadPoolExecutor(max_workers=3) as pool:
        work = [pool.submit(perspective, i) for i in range(3)]
        try:
            assert first_wave.wait(5), 'three actual POSTs must overlap'
            assert peak == 3
        finally:
            release.set()
        assert all(len(f.result()) == 3 for f in work)
    assert peak == 3
    receipts = [json.loads(p.read_text()) for p in (tmp_path / 'out/calls').glob('*.json')]
    assert len(receipts) == 9
    for receipt in receipts:
        assert receipt['attempt'] == 1 and not receipt['retryable']
        assert receipt['effective_reasoning_effort'] == 'medium'
        assert receipt['generation_elapsed_seconds'] >= 0
        assert receipt['queue_wait_seconds'] >= 0
        assert datetime.fromisoformat(receipt['finished_at']) >= datetime.fromisoformat(receipt['started_at'])
        assert receipt['usage']['output_tokens_details']['reasoning_tokens'] == 150
    output = capsys.readouterr().out
    assert 'SECRET' not in output and 'fake-test-key' not in output
    assert output.count('generation_started') == 9
    assert output.count('generation_finished') == 9
    assert not any(t.name == 'rag-call-progress' for t in threads())


def test_identical_preflight_count_is_single_flight(settings, tmp_path, monkeypatch):
    gateway = Gateway(settings, 'counts', tmp_path)
    counted, release = Event(), Event()
    calls = []
    def request(path, payload):
        calls.append(path)
        counted.set()
        assert release.wait(5)
        return {'input_tokens': 100}
    monkeypatch.setattr(gateway, 'request', request)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(gateway.preflight, 'research', 'i', 'same', reserve_input=i) for i in range(5)]
        assert counted.wait(5)
        release.set()
        results = [f.result() for f in futures]
    assert calls == ['responses/input_tokens']
    assert gateway.input_reserves[results[0]['request_hash']] == 4


def test_facets_overlap_after_all_preflights_and_drain_success_after_failure(tmp_path):
    p, content, units, seen, check = setup(tmp_path)
    barrier = Barrier(2)
    generate = p.gateway.generate
    def request(purpose, *args):
        assert len(seen) >= 3
        assert [r[2] for r in seen[:3]] == [0, 0, 0]
        barrier.wait(timeout=5)
        if purpose.endswith('costs'):
            raise APIError('controlled failure')
        return generate(purpose, *args)
    p.gateway.generate = request
    with pytest.raises(APIError, match='controlled'):
        reassess_facets(p, 'market_reassessment', '', content, check, BASE)
    saved = json.loads((tmp_path / 'reassessments/market_reassessment-split.json').read_text())
    assert saved['status'] == 'failed'
    assert list(saved['completed_facets']) == ['adoption']
    assert saved['original_assessment'] == content['previous_assessment']


@pytest.mark.parametrize('failure', [TimeoutError('SECRET'), urllib.error.URLError('SECRET'),
    urllib.error.HTTPError('https://secret.invalid', 503, 'SECRET', {}, None)])
def test_ambiguous_failure_has_one_post_and_safe_failed_receipt(settings, tmp_path, monkeypatch, capsys, failure):
    gateway = Gateway(settings, 'failed', tmp_path / 'out')
    posts = []
    def request(path, payload):
        if path == 'responses/input_tokens':
            return {'input_tokens': 100}
        posts.append(path)
        raise failure
    monkeypatch.setattr(gateway, 'request', request)
    with pytest.raises(APIError):
        gateway.generate('research', 'SECRET prompt', 'SECRET source')
    assert posts == ['responses']
    receipt = next((tmp_path / 'out/calls').glob('*.json')).read_text()
    assert 'SECRET' not in receipt + capsys.readouterr().out
    assert json.loads(receipt)['reservation_retained']
    assert gateway.budget.summary()['unsettled_calls'] == 1


def test_reuse_retains_original_duration_and_separate_cache_hit_time(settings, tmp_path, monkeypatch):
    first = Gateway(settings, 'first', tmp_path / 'first')
    monkeypatch.setattr(first, 'request', lambda path, payload: {'input_tokens': 100} if path.endswith('input_tokens') else completed())
    first.generate('research', 'i', 'x')
    original = json.loads(next((tmp_path / 'first/calls').glob('*.json')).read_text())
    second = Gateway(settings, 'second', tmp_path / 'second', cache_dir=tmp_path / 'first')
    second.generate('research', 'i', 'x')
    reused = json.loads((tmp_path / 'second/calls/research-reused.json').read_text())
    assert reused['generation_elapsed_seconds'] == original['generation_elapsed_seconds']
    assert reused['started_at'] == original['started_at']
    assert reused['cache_hit_elapsed_seconds'] >= 0


def test_heartbeat_observes_30_second_interval_and_is_joined(settings, tmp_path, monkeypatch, capsys):
    # Simulate the first elapsed interval without sleeping or altering the socket timeout.
    from threading import Event as RealEvent
    tick = RealEvent()
    waits = []
    class FastFirstInterval:
        def __init__(self):
            self.done = RealEvent()
            self.first = True
        def wait(self, duration):
            waits.append(duration)
            if self.first:
                self.first = False
                return False
            tick.set()
            return self.done.wait(duration)
        def set(self):
            self.done.set()
    monkeypatch.setattr('rag.llm.Event', FastFirstInterval)
    gateway = Gateway(settings, 'heartbeat', tmp_path)
    def request(path, payload):
        if path == 'responses/input_tokens':
            return {'input_tokens': 100}
        assert tick.wait(5)
        return completed()
    monkeypatch.setattr(gateway, 'request', request)
    gateway.generate('research', 'SECRET', 'SECRET')
    assert waits == [30, 30]
    output = capsys.readouterr().out
    assert 'generation_waiting' in output and 'SECRET' not in output
    assert not any(t.name == 'rag-call-progress' for t in threads())
