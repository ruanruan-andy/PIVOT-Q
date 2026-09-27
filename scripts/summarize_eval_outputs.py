"""Aggregate each independent evaluation root; preserve suite-level records."""
import argparse
import csv
import json
from pathlib import Path

SUITES = {'libero_spatial', 'libero_object', 'libero_goal', 'libero_10'}
CATEGORIES = ['Camera Viewpoints', 'Robot Initial States', 'Language Instructions',
              'Light Conditions', 'Background Textures', 'Sensor Noise', 'Objects Layout']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs',
                        help='outputs root or an explicit method/seed directory')
    args = parser.parse_args()
    groups = {}
    canonical = [args.root / family for family in ('quantvla', 'holoq_vla', 'qvla')]
    roots = [p for p in canonical if p.is_dir()]
    if not roots:
        if args.root.name == 'outputs':
            parser.error('No canonical quantizer directories found under outputs')
        roots = [args.root]
    for path in (p for root in roots for p in root.rglob('episodes.jsonl')):
        if path.parent.name == 'metrics' and path.parent.parent.name in SUITES:
            groups.setdefault(path.parents[2], []).append(path)
    for root, files in sorted(groups.items()):
        rows = []; seen = set()
        for path in sorted(files):
            for line in path.read_text().splitlines():
                if not line.strip(): continue
                row = json.loads(line)
                key = (row['suite'], row['task_name'])
                if key in seen: raise ValueError(f'Duplicate episode: {root}: {key}')
                if row['category'] not in CATEGORIES: raise ValueError(row['category'])
                seen.add(key); rows.append(row)
        stats = []
        for category in CATEGORIES + ['Avg']:
            selected = rows if category == 'Avg' else [r for r in rows if r['category'] == category]
            n = len(selected); wins = sum(bool(r['success']) for r in selected)
            stats.append(dict(category=category, episodes=n, successes=wins,
                              success_rate_pct=100*wins/n if n else None,
                              errors=sum(bool(r.get('error')) for r in selected)))
        complete = {p.parent.parent.name for p in files} == SUITES and all(r['episodes']==80 for r in stats[:-1])
        output = dict(complete=complete, expected_episodes=560, completed_episodes=len(rows),
                      aggregation='sum(successes)/sum(episodes); percentages are unrounded',
                      sources=[str(p) for p in sorted(files)], results=stats)
        # Dedicated filenames never overwrite evaluator-owned summary.json.
        (root/'aggregate_summary.json').write_text(json.dumps(output, indent=2)+'\n')
        with (root/'aggregate_summary.csv').open('w', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(stats[0])); writer.writeheader(); writer.writerows(stats)
        print(root, len(rows), 'complete' if complete else 'PARTIAL', stats[-1]['success_rate_pct'])

if __name__ == '__main__': main()
