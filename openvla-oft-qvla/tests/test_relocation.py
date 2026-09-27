import functools
import pytest
from integration.relocation import normalize_paths
from training import checkpoint
from test_integration import config, ToyPolicy


def test_only_paths_normalize_hashes_and_protocol_do_not():
    mapping = {'/old/workspace': '/new/workspace'}
    old = {'checkpoint': '/old/workspace/models/fp', 'hash': 'abc', 'method': 'pivot_q'}
    new = {'checkpoint': '/new/workspace/models/fp', 'hash': 'abc', 'method': 'pivot_q'}
    assert normalize_paths(old, mapping) == new
    assert normalize_paths('/old/workspace-other/file', mapping) == '/old/workspace-other/file'
    assert normalize_paths({**old, 'hash':'changed'}, mapping) != new


def test_adapter_load_after_relocation_keeps_identity_checks(tmp_path, monkeypatch):
    cfg = config('pivot_q')
    student = ToyPolicy(cfg, quantized=True, trainable=True)
    student.identity['checkpoint'] = '/old/workspace/models/fp'
    path = tmp_path/'adapter.pt'
    checkpoint.save(path, student, cfg)
    student.identity['checkpoint'] = '/new/workspace/models/fp'
    with pytest.raises(ValueError, match='identity'):
        checkpoint.read(path, student, cfg)
    monkeypatch.setattr(checkpoint, 'normalize_paths', functools.partial(normalize_paths, mapping={'/old/workspace':'/new/workspace'}))
    checkpoint.read(path, student, cfg)
    student.identity['weights'] = {'weight.safetensors':'different hash'}
    with pytest.raises(ValueError, match='identity'):
        checkpoint.read(path, student, cfg)
