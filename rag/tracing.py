"""Optional metadata-only tracing; telemetry never owns pipeline execution."""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps
from queue import Queue, Full, Empty
from threading import Thread, Event
import math
import re
import sys

_SESSION = ContextVar('rag_trace_session', default=None)
_PARENT = ContextVar('rag_trace_parent', default=None)
_NUMBERS = {'attempt', 'queue_wait_seconds', 'generation_elapsed_seconds', 'http_status'}
_LABELS = {'purpose', 'model', 'effort', 'run_id', 'request_id', 'role',
           'next_action', 'reason_code', 'error_code', 'result_status'}
_ACTIONS = {'research', 'market', 'stakeholder', 'domain', 'writer', 'evaluator', 'publish', 'stop'}
_WORKER_ROLES = {'research', 'market', 'stakeholder', 'domain', 'writer', 'evaluator'}
_REASONS = {'initial_research', 'missing_view', 'evidence_gap', 'evidence_ready',
            'report_ready', 'quality_rework', 'quality_passed', 'limit_exceeded', 'fatal_error'}
_ERRORS = {'retrieval_error', 'invalid_response', 'api_error', 'budget_exceeded',
           'input_budget_exceeded', 'uncertain_request', 'artifact_mismatch', 'render_error'}
_CONTROL_CODES = {'role': _WORKER_ROLES | {'publish', 'supervisor'}, 'next_action': _ACTIONS,
                  'reason_code': _REASONS, 'error_code': _ERRORS,
                  'result_status': {'ok', 'insufficient', 'failed'}}


def safe_metadata(values):
    result = {}
    for key, value in values.items():
        if key == 'attempt' and type(value) is int and value >= 1:
            result[key] = value
        elif key in _NUMBERS and key != 'attempt' and type(value) in (int, float) and math.isfinite(value) and value >= 0:
            result[key] = value
        elif key in _LABELS and isinstance(value, str):
            if key in _CONTROL_CODES and value in _CONTROL_CODES[key]:
                result[key] = value
            elif key in {'run_id', 'request_id'} and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', value):
                result[key] = value
            elif key not in _CONTROL_CODES and key not in {'run_id', 'request_id'} and re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', value):
                result[key] = value
        elif key in {'evidence_sufficient', 'passed'} and type(value) is bool:
            result[key] = value
        elif key == 'status' and value in ('research_ok', 'joined', 'report_drafted', 'revision_requested',
                                            'validated', 'human_review_pending', 'incomplete', 'failed', 'completed'):
            result[key] = value
        elif key == 'cache_hit' and type(value) is bool:
            result[key] = value
    usage = values.get('usage')
    if isinstance(usage, dict):
        clean = {k: usage[k] for k in ('input_tokens', 'output_tokens', 'total_tokens')
                 if type(usage.get(k)) is int and usage[k] >= 0}
        for source, target, key, dest in (
            ('input_tokens_details', 'input_token_details', 'cached_tokens', 'cache_read'),
            ('output_tokens_details', 'output_token_details', 'reasoning_tokens', 'reasoning')):
            details = usage.get(source)
            if isinstance(details, dict) and type(details.get(key)) is int and details[key] >= 0:
                clean[target] = {dest: details[key]}
        if clean:
            result['usage_metadata'] = clean
    return result


class Session:
    def __init__(self, settings):
        from langsmith import Client
        from urllib3.util.retry import Retry
        self.client = Client(api_key=settings.get('LANGSMITH_API_KEY'),
            api_url=settings.get('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com'),
            workspace_id=settings.get('LANGSMITH_WORKSPACE_ID') or None,
            auto_batch_tracing=False, timeout_ms=1000, retry_config=Retry(total=0),
            hide_inputs=True, hide_outputs=True, omit_traced_runtime_info=True)
        self.project = settings.get('LANGSMITH_PROJECT', 'skala-rag-project')
        self.queue = Queue(maxsize=512)
        self.warned = False
        self.stopping = Event()
        self.worker = Thread(target=self._send, daemon=True, name='rag-tracing')
        self.worker.start()

    def warn(self):
        if not self.warned:
            self.warned = True
            print('LangSmith tracing unavailable or incomplete; report generation continues.', file=sys.stderr)

    def _send(self):
        try:
            while True:
                try:
                    item = self.queue.get(timeout=0.05)
                except Empty:
                    if self.stopping.is_set():
                        return
                    continue
                try:
                    if item is None:
                        return
                    tree, method = item
                    getattr(tree, method)()
                except Exception:
                    self.warn()
                finally:
                    self.queue.task_done()
        finally:
            try:
                self.client.close(timeout=0.1)
            except Exception:
                self.warn()

    def enqueue(self, tree, method):
        try:
            self.queue.put_nowait((tree, method))
        except Full:
            self.warn()

    def close(self):
        # A slow telemetry server cannot hold up report delivery or process exit.
        self.stopping.set()
        self.worker.join(timeout=2.0)
        if self.worker.is_alive():
            self.warn()


@contextmanager
def span(name, run_type='chain', **metadata):
    session = _SESSION.get()
    tree = None
    token = None
    values = dict(metadata)
    if session is not None:
        try:
            from langsmith.run_trees import RunTree
            parent = _PARENT.get()
            tree = RunTree(name=name, run_type=run_type, inputs={}, outputs={},
                           parent_run=parent, ls_client=session.client, replicas=[],
                           project_name=session.project, extra={'metadata': safe_metadata(values)})
            token = _PARENT.set(tree)
            session.enqueue(tree, 'post')
        except Exception:
            session.warn()
    failed = False
    try:
        yield values
    except BaseException:
        failed = True
        raise
    finally:
        if tree is not None:
            try:
                tree.end(outputs={}, error='execution_failed' if failed or values.get('status') in ('incomplete', 'failed') else None,
                         metadata=safe_metadata(values))
                session.enqueue(tree, 'patch')
            except Exception:
                session.warn()
        if token is not None:
            _PARENT.reset(token)


@contextmanager
def trace_run(settings, run_id):
    # Disable automatic LangChain serialization, including ambient CLI tracing.
    from langsmith import tracing_context
    from langchain_core.tracers.context import tracing_v2_callback_var
    from langchain_core.runnables.config import var_child_runnable_config
    session = None
    enabled = settings.get('LANGSMITH_TRACING', 'false').lower()
    if enabled == 'true' and settings.get('LANGSMITH_API_KEY'):
        try:
            session = Session(settings)
        except Exception:
            print('LangSmith configuration unavailable; tracing disabled.', file=sys.stderr)
    elif enabled != 'false':
        print('LangSmith key missing or toggle invalid; tracing disabled.', file=sys.stderr)
    session_token = _SESSION.set(session)
    parent_token = _PARENT.set(None)
    callback_token = tracing_v2_callback_var.set(None)
    config_token = var_child_runnable_config.set({})
    try:
        with tracing_context(enabled=False, parent=False):
            with span('rag_run', run_id=run_id) as metadata:
                yield metadata
    finally:
        var_child_runnable_config.reset(config_token)
        tracing_v2_callback_var.reset(callback_token)
        _PARENT.reset(parent_token)
        _SESSION.reset(session_token)
        if session is not None:
            session.close()


def traced_node(name, function):
    @wraps(function)
    def run(*args, **kwargs):
        with span(name) as metadata:
            result = function(*args, **kwargs)
            if isinstance(result, dict):
                metadata['status'] = result.get('run_status')
            return result
    return run


def trace_decision(decision):
    """Record only the contract's routing fields, never reason or feedback text."""
    with span('supervisor_decision', role='supervisor',
              run_id=decision.get('run_id'), request_id=decision.get('request_id'),
              attempt=decision.get('attempt'), next_action=decision.get('next_action'),
              reason_code=decision.get('reason_code'),
              evidence_sufficient=decision.get('evidence_sufficient')):
        pass


def trace_agent_call(role, request, function):
    """Run a contract worker in a metadata-only span and return its response."""
    if role not in _WORKER_ROLES:
        raise ValueError('Unsupported traced Agent role')

    def field(value, name):
        return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

    with span(role, role=role, run_id=request.get('run_id'),
              request_id=request.get('request_id'), attempt=request.get('attempt')) as metadata:
        result = function(request)
        status = field(result, 'status')
        if isinstance(status, str) and status in _CONTROL_CODES['result_status']:
            metadata['result_status'] = status
            metadata['status'] = 'failed' if status == 'failed' else 'completed'
            if role == 'evaluator':
                metadata['passed'] = field(result, 'passed')
            error = field(result, 'error')
            if error is not None:
                metadata['error_code'] = field(error, 'code')
        return result


def submit(pool, function, *args, **kwargs):
    return pool.submit(copy_context().run, function, *args, **kwargs)
