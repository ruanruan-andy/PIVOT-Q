#!/usr/bin/env python3
"""Aggregate only canonical, complete, error-free LIBERO-Plus evaluations."""
import argparse
import csv
import json
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
CATEGORIES = ('Camera Viewpoints', 'Robot Initial States', 'Language Instructions', 'Light Conditions', 'Background Textures', 'Sensor Noise', 'Objects Layout')
METRICS = ('Camera', 'Init.', 'Language', 'Lighting', 'Background', 'Noise', 'Layout', 'Avg.')
FAMILIES = {('quantvla', 'groot_n1_5'): 'groot_quantvla',
            ('quantvla', 'pi05'): 'pi05_quantvla',
            ('holoq_vla', 'groot_n1_5'): 'groot_holoqvla',
            ('holoq_vla', 'pi05'): 'pi05_holoqvla',
            ('qvla', 'openvla_oft'): 'openvla_oft_qvla'}

def validate_identity(directory, family, model, method, seed, suite, records, manifest):
    receipt = directory / 'core_command.json'
    if not receipt.is_file():
        raise ValueError('missing core_command.json; legacy run identity must be verified before aggregation')
    data = json.loads(receipt.read_text())
    setting = {'original': 'fp_original', 'quantized': 'quant_original'}.get(method, method)
    expected = dict(family=FAMILIES[(family, model)], setting=setting, seed=seed, suite=suite)
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('run identity differs from directory labels')
    if data.get('benchmark') != 'libero-plus':
        raise ValueError('benchmark identity must be libero-plus')
    for parent in (directory, directory / 'eval'):
        for name in ('run.json', 'protocol.json'):
            path = parent / name
            if not path.is_file():
                continue
            metadata = json.loads(path.read_text())
            cfg = metadata.get('config', {})
            recorded = [metadata.get('eval_seed'), metadata.get('policy_seed'),
                        cfg.get('policy_seed'), cfg.get('evaluation', {}).get('seed')]
            if name == 'protocol.json':
                recorded.append(metadata.get('seed'))
            if any(int(v) != seed for v in recorded if v is not None):
                raise ValueError('runtime evaluation seed conflicts with receipt')
    wanted = {(int(task), 0) for task in manifest['task_ids_by_suite'][suite]}
    actual = {(int(r.get('task_index', r.get('task_id'))),
               int(r.get('initial_state_id', r.get('episode_index', 0)))) for r in records}
    if actual != wanted or len(records) != len(wanted):
        raise ValueError('episode identities differ from manifest')
    if any(r.get('suite', suite) != suite for r in records):
        raise ValueError('episode suite differs from directory')

def display(value):
    return str(Decimal(str(value)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))

def collect(root, manifest=None):
    if manifest is None:
        manifest = json.loads((ROOT / 'manifests/libero_plus_first20.json').read_text())
    if manifest.get('initial_state_ids', [0]) != [0]:
        raise ValueError('seven-category aggregation requires initial state 0')
    rows, audit = [], []
    for family in ('quantvla', 'holoq_vla', 'qvla'):
      for model in sorted((root / family).glob('*')):
        if not model.is_dir():
            continue
        methods = [p for p in model.iterdir() if p.is_dir() and p.name not in {'ablations', 'ablation_phase'}]
        for group in ('ablations', 'ablation_phase'):
            methods.extend(p for p in (model / group).glob('*') if p.is_dir())
        for method in sorted(methods):
          for seed in sorted((method / 'eval').glob('seed-*')):
            actual_seed = 2026 if seed.name == 'seed-000' else int(seed.name[5:])
            totals = {c: [0, 0] for c in CATEGORIES}
            sources, issues = [], []
            for suite in SUITES:
                files = [p for p in (seed / suite / 'metrics/summary.json', seed / suite / 'eval/summary.json', seed / 'libero-plus' / suite / 'metrics/summary.json') if p.is_file()]
                if len(files) != 1:
                    issues.append(f'{suite}: {len(files)} summaries; expected one')
                    continue
                path = files[0]
                try:
                    data = json.loads(path.read_text())
                    cats = data.get('by_category') or data.get('categories') or {}
                    count = data.get('completed_episodes', data.get('episodes'))
                    errors = data.get('errors', sum(c.get('errors', 0) for c in cats.values()))
                    if count != 140 or errors:
                        issues.append(f'{suite}: episodes={count}, errors={errors}')
                    ep = path.parent / 'episodes.jsonl'
                    records = [json.loads(line) for line in ep.read_text().splitlines()] if ep.is_file() else []
                    validate_identity(path.parent.parent, family, model.name,
                        str(method.relative_to(model)), actual_seed, suite, records, manifest)
                    ids = {(r.get('task_index', r.get('task_id')), r.get('initial_state_id', 0)) for r in records}
                    if len(records) != 140 or len(ids) != 140:
                        issues.append(f'{suite}: episode records incomplete')
                    if any(r.get('error') or not isinstance(r.get('success'), bool) for r in records):
                        issues.append(f'{suite}: invalid episode/error')
                    for category in CATEGORIES:
                        item = cats.get(category, {})
                        n = item.get('episodes', 0)
                        k = item.get('successes')
                        if k is None and 'success_rate' in item:
                            k = round(n * item['success_rate'])
                        if n != 20 or k is None or not 0 <= k <= n:
                            issues.append(f'{suite}: invalid {category}')
                            continue
                        category_rows = [r for r in records if r.get('category') == category]
                        if len(category_rows) != n or sum(r.get('success') is True for r in category_rows) != k:
                            issues.append(f'{suite}: summary disagrees with episodes for {category}')
                        totals[category][0] += k
                        totals[category][1] += n
                    sources.append(str(path.relative_to(root)))
                except (ValueError, OSError, TypeError, KeyError) as exc:
                    issues.append(f'{suite}: {exc}')
            complete = not issues and all(n == 80 for _, n in totals.values())
            values = [100*k/n if complete else None for k,n in totals.values()]
            rows.append(dict(Setting=f'{family}/{model.name}', Method=str(method.relative_to(model)), Seed=actual_seed,
                **dict(zip(METRICS, values + [sum(values)/7 if complete else None])), Status='complete' if complete else 'incomplete',
                Episodes=sum(n for _,n in totals.values()), Source='; '.join(sources)))
            audit.append(dict(directory=str(seed.relative_to(root)), issues=issues, legacy_seed_alias=seed.name=='seed-000'))
    for row in rows:
        if sum((r['Setting'],r['Method'],r['Seed']) == (row['Setting'],row['Method'],row['Seed']) for r in rows) > 1:
            row.update(Status='duplicate seed alias', **{m: None for m in METRICS})
    return rows, audit

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs')
    parser.add_argument('--manifest', type=Path, default=ROOT/'manifests/libero_plus_first20.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026,2027,2028])
    args = parser.parse_args()
    rows, audit = collect(args.output_root, json.loads(args.manifest.read_text()))
    rendered = []
    for key in sorted({(r['Setting'],r['Method']) for r in rows}):
        group = sorted([r for r in rows if (r['Setting'],r['Method']) == key], key=lambda r:r['Seed'])
        rendered.extend(group)
        selected = [r for r in group if r['Seed'] in args.seeds and r['Status']=='complete']
        if len(selected)==len(args.seeds) and len(selected)>1:
            rendered.append(dict(Setting=key[0],Method=key[1],Seed='Mean ± Std.',
                **{m:f'{display(statistics.mean(r[m] for r in selected))} ± {display(statistics.stdev(r[m] for r in selected))}' for m in METRICS},
                Status='complete',Episodes=560,Source='seeds: '+', '.join(map(str,args.seeds))))
    out = args.output_root/'statistics'
    out.mkdir(parents=True,exist_ok=True)
    fields = ['Setting','Method','Seed',*METRICS,'Status','Episodes','Source']
    (out/'all_results.json').write_text(json.dumps(rendered,indent=2)+'\n')
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    with (out/'all_results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader(); writer.writerows(rendered)
    def markdown(items):
        cols=fields[:-1]
        lines=['| '+' | '.join(cols)+' |','|'+'---|'*len(cols)]
        for r in items:
            lines.append('| '+' | '.join('—' if r[c] is None else display(r[c]) if isinstance(r[c],float) else str(r[c]) for c in cols)+' |')
        return '\n'.join(lines)+'\n'
    (out/'all_results.md').write_text(markdown(rendered))
    (out/'by_setting').mkdir(exist_ok=True)
    for setting in sorted({r['Setting'] for r in rendered}):
        (out/'by_setting'/(setting.replace('/','_')+'.md')).write_text(markdown([r for r in rendered if r['Setting']==setting]))
    print(f'{len(rows)} seed rows, {sum(r["Status"]=="complete" for r in rows)} complete; {out}')

if __name__=='__main__':
    main()
