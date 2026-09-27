"""Explicit, readable protocol comparisons; no weight-content hashing."""
import json
from pathlib import Path
import yaml

RUNTIME_FIELDS = {'resume', 'dry_run', 'output_dir', 'env_ports', 'clean_env_ports',
                  'env_port', 'clean_env_port', 'max_rounds', 'device',
                  'collective_smoke_test', 'keep_last_checkpoints', 'save_every_steps'}

def behavioral_config(value):
    if isinstance(value, dict):
        ignored = RUNTIME_FIELDS | {'rollout_ports', 'clean_anchor_ports', 'master_port'}
        return {k:behavioral_config(v) for k,v in value.items() if k not in ignored}
    if isinstance(value, list):
        return [behavioral_config(v) for v in value]
    return value

def assert_same(previous, current, label):
    if previous != current:
        keys = sorted(k for k in set(previous) | set(current) if previous.get(k) != current.get(k))
        raise ValueError(f'{label} changed: {", ".join(keys)}; use the original configuration or a new output directory')

def file_value(path):
    path = Path(path)
    if path.suffix in {'.yaml', '.yml'}:
        value = yaml.safe_load(path.read_text())
        if isinstance(value, dict) and value.get('extends'):
            base = file_value(path.parent / value.pop('extends'))
            def merge(a, b):
                for key, item in b.items():
                    if isinstance(item, dict) and isinstance(a.get(key), dict):
                        merge(a[key], item)
                    else:
                        a[key] = item
                return a
            return merge(base, value)
        return value
    return json.loads(path.read_text())

def training_protocol(config):
    protocol = {k:v for k,v in config.items() if k not in RUNTIME_FIELDS}
    # Capture mutable protocol files rather than just their names.
    for key in ('manifest', 'omega_config'):
        if config.get(key):
            contents = file_value(config[key])
            protocol[key + '_contents'] = behavioral_config(contents) if key == 'omega_config' else contents
    if not config.get('omega_config'):
        protocol['quantization_config'] = file_value(Path(__file__).resolve().parents[1] /
            'pi05_quantvla/config/quantvla.yaml')['quantization']
    if config.get('checkpoint'):
        path = Path(config['checkpoint']) / 'config.json'
        if path.is_file():
            protocol['model_config'] = file_value(path)
    return json.loads(json.dumps(protocol, default=str))

def check_training_output(output, config):
    output = Path(output)
    protocol = training_protocol(config)
    record = output / 'training_protocol.json'
    occupied = record.exists() or (output / 'final_adapter').is_symlink() or any((output / name).exists() for name in
        ('run.json', 'metrics.jsonl', 'metrics', 'checkpoints', 'final_adapter'))
    if occupied:
        if not config.get('resume'):
            raise ValueError('Training output already exists; enable resume or choose a new output directory')
        if not record.is_file():
            raise ValueError('Legacy training output lacks training_protocol.json; compatibility cannot be established automatically')
        assert_same(json.loads(record.read_text()), protocol, 'Training protocol')
        for marker in (output / 'checkpoints').glob('step-*/complete.json'):
            saved = json.loads(marker.read_text()).get('protocol')
            if saved is None:
                raise ValueError(f'Legacy checkpoint lacks protocol: {marker}')
            assert_same(saved, protocol, 'Checkpoint protocol')
    return protocol

def write_training_protocol(output, protocol):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'training_protocol.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(protocol, indent=2) + '\n')
    temporary.replace(target)
