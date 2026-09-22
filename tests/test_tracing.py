"""Metadata tracing contracts with no credentials or remote requests."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
import json
import time

import pytest

from rag.settings import Settings
from rag.tracing import trace_run, span, submit, traced_node
from rag.llm import Gateway, APIError
from test_gateway import settings, completed, no_network_or_retry_wait


@pytest.fixture
def client(monkeypatch):
    class Fake:
        def __init__(self, **kwargs):
            self.records = []
            self.kwargs = kwargs
        def create_run(self, **kwargs):
            self.records.append(('create', kwargs))
        def update_run(self, run_id, **kwargs):
            self.records.append(('update', {'id': run_id, **kwargs}))
        def close(self, **kwargs):
            pass
    fake = Fake()
    def make(**kwargs):
        fake.kwargs = kwargs
        return fake
    monkeypatch.setattr('langsmith.Client', make)
    return fake


def enabled():
    return Settings({'LANGSMITH_TRACING': 'true', 'LANGSMITH_API_KEY': 'KEY_SENTINEL'})


def test_off_suppresses_ambient_tracing(monkeypatch):
    from langgraph.graph import StateGraph, START, END
    from langchain_core.tracers.context import tracing_v2_callback_var
    def forbidden(**kwargs):
        pytest.fail('off must not construct a Client')
    monkeypatch.setattr('langsmith.Client', forbidden)
    monkeypatch.setenv('LANGSMITH_TRACING', 'true')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'true')
    token = tracing_v2_callback_var.set(object())
    try:
        graph = StateGraph(dict)
        graph.add_node('safe', traced_node('safe', lambda state: state))
        graph.add_edge(START, 'safe'); graph.add_edge('safe', END)
        with trace_run(Settings({'LANGSMITH_TRACING': 'false'}), 'off'):
            assert graph.compile().invoke({'secret': 'RAW_SENTINEL'})['secret'] == 'RAW_SENTINEL'
    finally:
        tracing_v2_callback_var.reset(token)


def test_parallel_parent_and_content_boundary(client):
    barrier = Barrier(2)
    def child():
        with span('generation', 'llm', purpose='research', effort='medium',
                  prompt='RAW_SENTINEL', usage={'input_tokens': 4, 'output_tokens': 6, 'total_tokens': 10}) as data:
            barrier.wait(timeout=3)
            data['response'] = 'RAW_SENTINEL'
    with trace_run(enabled(), 'test-run'):
        with span('research'):
            with ThreadPoolExecutor(2) as pool:
                futures = [submit(pool, child) for _ in range(2)]
                for future in futures:
                    future.result()
        with pytest.raises(ValueError):
            with span('failed'):
                raise ValueError('RAW_SENTINEL KEY_SENTINEL')
    payload = json.dumps(client.records, default=str)
    assert 'SENTINEL' not in payload
    creates = [data for kind, data in client.records if kind == 'create']
    root = next(x for x in creates if x['name'] == 'rag_run')
    research = next(x for x in creates if x['name'] == 'research')
    assert research['parent_run_id'] == root['id']
    calls = [x for x in creates if x['name'] == 'generation']
    assert len(calls) == 2
    assert all(x['parent_run_id'] == research['id'] and x['trace_id'] == root['id'] for x in calls)
    assert all(x['extra']['metadata']['usage_metadata']['total_tokens'] == 10 for x in calls)
    assert client.kwargs['hide_inputs'] and client.kwargs['hide_outputs']


def test_gateway_payload_usage_and_cache(client, settings, tmp_path, monkeypatch):
    gateway = Gateway(settings, 'gateway', tmp_path)
    payloads = []
    def request(path, payload):
        if path.endswith('input_tokens'):
            return {'input_tokens': 100}
        payloads.append(payload)
        return completed('RAW_SENTINEL')
    monkeypatch.setattr(gateway, 'request', request)
    original = gateway.build_payload('market', 'RAW_SENTINEL', 'RAW_SENTINEL')
    public = settings.public()
    settings.values.update(enabled().values)
    assert settings.public() == public
    with trace_run(settings, 'gateway'):
        assert gateway.generate('market', 'RAW_SENTINEL', 'RAW_SENTINEL') == 'RAW_SENTINEL'
        gateway.cached[gateway.payload_hash(original)] = {'text': 'RAW_SENTINEL'}
        assert gateway.generate('market', 'RAW_SENTINEL', 'RAW_SENTINEL') == 'RAW_SENTINEL'
    assert payloads == [original]
    assert 'SENTINEL' not in json.dumps(client.records, default=str)
    updates = [d for kind, d in client.records if kind == 'update']
    usage = [d['extra']['metadata']['usage_metadata'] for d in updates if 'usage_metadata' in d.get('extra', {}).get('metadata', {})]
    assert len(usage) == 1 and usage[0]['input_tokens'] == 100


def test_slow_telemetry_does_not_block_pipeline(client):
    release = Event()
    client.create_run = lambda **kwargs: release.wait(3)
    started = time.monotonic()
    try:
        with trace_run(enabled(), 'slow'):
            with span('node'):
                pass
        assert time.monotonic() - started < 2.8
    finally:
        release.set()


def test_failed_telemetry_preserves_original_error(client):
    def fail(**kwargs):
        raise RuntimeError('KEY_SENTINEL')
    client.create_run = fail
    with pytest.raises(ValueError, match='original'):
        with trace_run(enabled(), 'failed'):
            raise ValueError('original')


def test_real_sdk_serialization_and_graph_are_metadata_only(monkeypatch):
    from langsmith import Client
    from langgraph.graph import StateGraph, START, END
    import requests
    sent = []
    def transport(self, method, path, **kwargs):
        request = kwargs.get('request_kwargs', {})
        assert request.get('headers', {}).get('x-api-key') == 'KEY_SENTINEL'
        sent.append((method, path, request.get('data')))
        response = requests.Response()
        response.status_code = 200
        response._content = b'{}'
        return response
    monkeypatch.setattr(Client, 'request_with_retries', transport)
    monkeypatch.setenv('LANGSMITH_METADATA', '{"private":"RAW_SENTINEL"}')
    monkeypatch.setenv('LANGSMITH_TRACING', 'true')
    graph = StateGraph(dict)
    graph.add_node('research', traced_node('research', lambda state: state))
    graph.add_edge(START, 'research'); graph.add_edge('research', END)
    with trace_run(enabled(), 'sdk-test'):
        assert graph.compile().invoke({'document': 'RAW_SENTINEL'})['document'] == 'RAW_SENTINEL'
    assert len(sent) == 4  # root and explicit node only; no automatic graph traces
    encoded = json.dumps(sent, default=lambda x: x.decode() if isinstance(x, bytes) else str(x))
    assert 'SENTINEL' not in encoded
    assert all(method in ('POST', 'PATCH') for method, _, _ in sent)


def test_inherited_callback_cannot_receive_source_text():
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables.config import var_child_runnable_config
    from langgraph.graph import StateGraph, START, END
    received = []
    class Collector(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, **kwargs):
            received.append(inputs)
    inherited = {'callbacks': [Collector()]}
    token = var_child_runnable_config.set(inherited)
    try:
        graph = StateGraph(dict)
        graph.add_node('node', lambda state: state)
        graph.add_edge(START, 'node'); graph.add_edge('node', END)
        with trace_run(Settings({}), 'off'):
            graph.compile().invoke({'secret': 'RAW_SENTINEL'})
        assert not received
        assert var_child_runnable_config.get() is inherited
    finally:
        var_child_runnable_config.reset(token)


def test_handled_pipeline_failure_marks_trace_failed(client):
    with trace_run(enabled(), 'handled') as trace:
        result = traced_node('research', lambda: {'run_status': 'incomplete'})()
        trace['status'] = result['run_status']
    updates = [data for kind, data in client.records if kind == 'update']
    assert len(updates) == 2
    assert all(data['error'] == 'execution_failed' for data in updates)
