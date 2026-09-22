"""Attach measured frame timings to the external report and existing CSVs."""
import csv
import json
from pathlib import Path
import numpy as np
from .fit_ood_v4 import save_json
from .profile_external import stats


def main():
    run = Path('runs/pant_fail_v7_20260921')
    report = Path('reports/ordered_duration_v7_20260921')
    extension = json.loads((run/'report_extension.json').read_text())
    profiles = {mode: json.loads((run/f'latency_{mode}.json').read_text()) for mode in ('monitor', 'rgb')}
    selected = {r['id'] for r in profiles['monitor']['episodes']}
    assert selected == {r['id'] for r in profiles['rgb']['episodes']}
    assert all(p['complete'] and p['total_frames'] == sum(e['length'] for e in extension['episodes'] if e['id'] in selected) for p in profiles.values())
    assert all(p['bundle_sha256'] == extension['evaluation']['bundle_sha256'] for p in profiles.values())
    aggregate = {}
    for e in extension['episodes']:
        if e['id'] not in selected:
            e.pop('latency', None)
            e['result'].pop('latency_ms', None)
            continue
        arrays = {}
        for mode in profiles:
            with np.load(run/f'{e["id"]}_latency_{mode}.npz') as times:
                arrays.update({key: times[key] for key in times.files})
        assert all(len(a) == e['length'] and np.isfinite(a).all() and (a >= 0).all() for a in arrays.values())
        e['result']['latency_ms'] = {key: a.tolist() for key, a in arrays.items()}
        e['latency'] = {key: stats(a) for key, a in arrays.items()}
        for key, a in arrays.items():
            aggregate.setdefault(key, []).append(a)
        csv_path = report/e['result']['csv']
        with csv_path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f); rows = list(reader); fields = list(reader.fieldnames)
        assert len(rows) == e['length']
        for key in arrays:
            if key not in fields:
                fields.append(key)
        for t, row in enumerate(rows):
            row.update({key: float(a[t]) for key, a in arrays.items()})
        with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    latency = dict(profiles=profiles, aggregate_ms={key: stats(np.concatenate(a)) for key, a in aggregate.items()},
                   measured_episodes=len(selected), measured_frames=profiles['monitor']['total_frames'],
                   selected_ids=sorted(selected),
                   primary_keys=['monitor_ms', 'rgb_total_ms'],
                   note='Every frame measured. Cached-feature inference and batch1 RGB path are separate actual passes. Component synchronization overhead is not part of uninstrumented monitor total. Do not sum percentiles.',
                   source_result='Main OOD results retain original causal stride1 batch8 visual cache; RGB benchmark output and its BF16 consistency comparison are separately saved.')
    extension['evaluation']['latency'] = latency
    save_json(extension, run/'report_extension.json')
    evaluation = json.loads((run/'evaluation.json').read_text()); evaluation['latency'] = latency
    save_json(evaluation, run/'evaluation.json')
    print(json.dumps(dict(complete=True, frames=latency['measured_frames'], columns=len(aggregate))))


if __name__ == '__main__':
    main()
