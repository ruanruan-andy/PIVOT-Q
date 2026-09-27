#!/usr/bin/env python3
"""Configure workspace paths and trusted LIBERO compatibility links."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import yaml

CONFIGS = {
    'quantvla-groot': ['config/*.yaml'],
    'quantvla-pi05': ['pi05_quantvla/config/*.yaml', 'pi05_quantvla/pivot_q/config/*.yaml'],
    'omegavla-groot': ['omega_qvla/config/*.yaml', 'omega_qvla/pivot_q/config/*.yaml'],
    'omegavla-pi05': ['pi05_omegavla/config/*.yaml', 'pi05_quantvla/config/*.yaml'],
    'qvla-openvla-oft': ['openvla-oft-qvla/configs/*.yaml'],
}

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])

def upstream_links(root):
    manifest = json.loads((root/'upstream_patches/manifest.json').read_text())
    links = []
    for name, entry in manifest.items():
        repo = root/'third_party'/name
        if not (repo/'.git').exists():
            raise ValueError(f'Initialize submodules first: {repo}')
        head = git(repo, 'rev-parse', 'HEAD').decode().strip()
        if head != entry['base_commit']:
            raise ValueError(f'{name}: expected {entry["base_commit"]}, found {head}')
        for item in entry['files']:
            relative = item['path']
            source = root/'upstream_patches'/name/'files'/relative
            target = repo/relative
            if not source.is_file():
                raise FileNotFoundError(source)
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise ValueError(f'Unknown symlink; left untouched: {target}')
                continue
            official = git(repo, 'show', f'{head}:{relative}')
            if not target.is_file() or target.read_bytes() not in (official, source.read_bytes()):
                raise ValueError(f'Unknown source changes; left untouched: {target}')
            links.append((target, os.path.relpath(source, target.parent)))
    return links

def plan(root, settings):
    runtime = root/'env/runtime'
    state = runtime/'relocation.json'
    previous = json.loads(state.read_text()) if state.exists() else {}
    mapping = {k:str(root) for k in previous.get('path_mapping', {}) if k != str(root)}
    old = previous.get('workspace')
    if old and old != str(root):
        mapping[old] = str(root)
    mapping['/path/to/PIVOT-Q'] = str(root)
    files = {root/'config/eval.yaml'}
    for setting in settings:
        for pattern in CONFIGS[setting]:
            files.update(root.glob(pattern))
    changes = {}
    for path in sorted(files):
        text = path.read_text()
        data = yaml.safe_load(text) or {}
        old = data.get('paths', {}).get('root_dir')
        if old and old != str(root):
            mapping[old] = str(root)
        # Replace only root prefixes; experiment scalars and comments are retained.
        for origin in sorted(mapping, key=len, reverse=True):
            text = text.replace(origin, str(root))
        if text != path.read_text():
            changes[path] = text
    for dest, name in [
        ('config/sparse_anchor_libero/config.yaml', 'LIBERO'),
        ('third_party/QuantVLA/configs/libero_plus/config.yaml', 'LIBERO-plus'),
        ('openvla-oft-qvla/configs/libero_plus/config.yaml', 'LIBERO-plus'),
        ('openvla-oft-qvla/cache/libero/config.yaml', 'LIBERO'),
        ('openvla-oft-qvla/cache/libero_plus/config.yaml', 'LIBERO-plus'),
    ]:
        repo = root/'third_party'/name
        base = repo/'libero/libero'
        data = dict(assets=str(base/'assets'), bddl_files=str(base/'bddl_files'),
                    benchmark_root=str(base), datasets=str(repo/'datasets'),
                    init_states=str(base/'init_files'))
        changes[root/dest] = yaml.safe_dump(data, sort_keys=False)
    exports = {
        'PIVOT_Q_ROOT':str(root), 'PIVOT_Q_DIR':str(root),
        'QUANTVLA_DIR':str(root/'third_party/QuantVLA'),
        'OMEGA_DIR':str(root/'third_party/Omega-QVLA'),
        'QVLA_DIR':str(root/'third_party/QVLA'),
        'LEROBOT_DIR':str(root/'third_party/lerobot'),
        'LIBERO_DIR':str(root/'third_party/LIBERO'),
        'LIBERO_ROOT':str(root/'third_party/LIBERO'),
        'LIBERO_PLUS_DIR':str(root/'third_party/LIBERO-plus'),
        'LIBERO_PLUS_ROOT':str(root/'third_party/LIBERO-plus'),
        'LIBERO_CONFIG_PATH':str(root/'third_party/QuantVLA/configs/libero_plus'),
        'LIBERO_PLUS_CONFIG_PATH':str(root/'third_party/QuantVLA/configs/libero_plus'),
        'CLEAN_LIBERO_CONFIG_PATH':str(root/'config/sparse_anchor_libero'),
        'PIVOT_Q_CLEAN_CONFIG_PATH':str(root/'config/sparse_anchor_libero'),
        'PIVOT_Q_QUANTVLA_ROOT':str(root/'third_party/QuantVLA'),
        'OFT_DIR':str(root/'openvla-oft-qvla'),
        'HF_HOME':str(root/'.cache/huggingface'),
        'QT_QPA_PLATFORM':'offscreen', 'MUJOCO_GL':'egl',
        'PYTHONNOUSERSITE':'1', 'PYTHONDONTWRITEBYTECODE':'1',
    }
    try:
        conda = os.environ.get('CONDA_EXE', 'conda')
        base = subprocess.check_output([conda, 'info', '--base'], text=True).strip()
        exports['CONDA_SH'] = str(Path(base)/'etc/profile.d/conda.sh')
    except (OSError, subprocess.CalledProcessError):
        print('NOTE: Conda not detected; install/activate a documented environment before experiments.')
    changes[runtime/'paths.sh'] = '# Generated workspace paths\n' + ''.join(
        f'export {k}={shlex.quote(v)}\n' for k,v in exports.items())
    changes[state] = json.dumps({'workspace':str(root), 'path_mapping':mapping}, indent=2)+'\n'
    return {p:s for p,s in changes.items() if not p.exists() or p.read_text()!=s}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=Path(__file__).resolve().parents[1])
    parser.add_argument('--setting', action='append', choices=CONFIGS)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--paths-only', action='store_true',
                        help='Configure paths before installing/checking submodules.')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not (root/'pivot_q/train.py').is_file():
        parser.error(f'Not a PIVOT-Q checkout: {root}')
    changes = plan(root, args.setting or list(CONFIGS))
    links = [] if args.paths_only else upstream_links(root)
    for path in changes:
        print(f'CONFIGURE {path.relative_to(root)}')
    for path,target in links:
        print(f'LINK {path.relative_to(root)} -> {target}')
    if not args.apply:
        print('Preview only; add --apply to write changes.')
        return
    for path,text in changes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for path,target in links:
        path.unlink()
        path.symlink_to(target)
    print('Configured. Source env/runtime/paths.sh in each new terminal.')

if __name__ == '__main__':
    main()
