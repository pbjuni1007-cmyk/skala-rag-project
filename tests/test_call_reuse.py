from copy import deepcopy
import hashlib
import json

import pytest
from app import check_call_reuse


def fixture(tmp_path):
    sources = {'paper': {'sha256': 'paper-sha', 'text_sha256': 'text-sha', 'version': 'v1', 'url': 'https://example.test', 'status': 'ok', 'raw_sha256': 'raw'}}
    identity = {'config': {'top_k': 5}, 'settings': {'model': 'fixed'}, 'index_hash': 'index', 'lock_sha256': 'lock', 'code_sha256': 'old-code',
                'sources': {k: {n: v.get(n) for n in ('sha256','text_sha256','version','url','status')} for k,v in sources.items()}}
    index = {'index_hash': 'index', 'embedding': {'model': 'e5', 'revision': 'fixed'}}
    manifest = {'config': identity['config'], 'settings': identity['settings'], 'index': index, 'lock_sha256': 'lock', 'code_sha256': 'old-code',
                'requirements_sha256': 'req', 'fingerprint': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'sources.json').write_text(json.dumps(sources))
    return deepcopy(identity), deepcopy(index), deepcopy(sources)


def test_code_only_reuse_keeps_strict_resume_identity_different(tmp_path):
    identity,index,sources = fixture(tmp_path)
    identity['code_sha256'] = 'new-code'
    original = json.loads((tmp_path/'manifest.json').read_text())
    assert original['fingerprint'] != hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    check_call_reuse(tmp_path, identity, 'req', index, sources)


@pytest.mark.parametrize('field',['config','settings','index_hash','lock_sha256','sources','requirements','embedding','raw'])
def test_reuse_rejects_changed_inputs(tmp_path, field):
    identity,index,sources = fixture(tmp_path)
    requirement = 'req'
    if field == 'requirements': requirement = 'changed'
    elif field == 'embedding': index['embedding']['revision'] = 'changed'
    elif field == 'raw': sources['paper']['raw_sha256'] = 'changed'
    else: identity[field] = 'changed'
    with pytest.raises(ValueError): check_call_reuse(tmp_path, identity, requirement, index, sources)


@pytest.mark.parametrize('field',['config','settings','index','lock_sha256','code_sha256','fingerprint','requirements_sha256'])
def test_missing_manifest_fields_cannot_match(tmp_path, field):
    identity,index,sources = fixture(tmp_path)
    p=tmp_path/'manifest.json';m=json.loads(p.read_text());del m[field];p.write_text(json.dumps(m))
    with pytest.raises((ValueError,KeyError)): check_call_reuse(tmp_path, identity, 'req', index, sources)


def test_previous_source_manifest_mix_is_rejected(tmp_path):
    identity,index,sources = fixture(tmp_path)
    changed=deepcopy(sources);changed['paper']['text_sha256']='other'
    (tmp_path/'sources.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='prior manifest'):check_call_reuse(tmp_path, identity, 'req', index, sources)


def test_discovery_policy_change_rejects_reuse_even_with_same_documents(tmp_path):
    identity, index, sources = fixture(tmp_path)
    identity['retrieval'] = {'discovery': {'total': 6}, 'chunk_tokens': 380}
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['retrieval'] = deepcopy(identity['retrieval'])
    manifest['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(manifest))
    check_call_reuse(tmp_path, identity, 'req', index, sources)
    identity['retrieval']['discovery']['total'] = 3
    with pytest.raises(ValueError, match='inputs'):
        check_call_reuse(tmp_path, identity, 'req', index, sources)


def test_legacy_run_cannot_satisfy_new_discovery_contract(tmp_path):
    identity, index, sources = fixture(tmp_path)
    identity['retrieval'] = {'discovery': {'total': 6}}
    with pytest.raises(ValueError, match='inputs'):
        check_call_reuse(tmp_path, identity, 'req', index, sources)


def test_public_contract_loads_without_private_documents(tmp_path, monkeypatch):
    from app import report_contract_hash
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'config').mkdir()
    contract = tmp_path / 'config/report-contract.yaml'
    contract.write_text('version: 1\n')
    first = report_contract_hash()
    contract.write_text('version: 2\n')
    assert report_contract_hash() != first


def test_report_contract_participates_in_reuse_identity(tmp_path):
    identity, index, sources = fixture(tmp_path)
    identity['report_contract_sha256'] = 'req'
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['report_contract_sha256'] = 'req'
    manifest['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(manifest))
    check_call_reuse(tmp_path, identity, 'req', index, sources)
    identity['report_contract_sha256'] = 'changed'
    with pytest.raises(ValueError, match='inputs'):
        check_call_reuse(tmp_path, identity, 'changed', index, sources)
