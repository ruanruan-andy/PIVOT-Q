"""Compare relocated metadata without altering weights or weakening content hashes."""
from __future__ import annotations
import json
from pathlib import Path


def normalize_paths(value, mapping=None):
    if mapping is None:
        state = Path(__file__).resolve().parents[2] / 'env/runtime/relocation.json'
        mapping = json.loads(state.read_text()).get('path_mapping', {}) if state.exists() else {}
    if isinstance(value, dict):
        return {key: normalize_paths(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_paths(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_paths(item, mapping) for item in value)
    if isinstance(value, str):
        for old in sorted(mapping, key=len, reverse=True):
            if value == old or value.startswith(old + '/'):
                return mapping[old] + value[len(old):]
    return value
