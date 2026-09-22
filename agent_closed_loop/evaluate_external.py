"""Evaluate new LeRobot recordings with a fixed portable V7 checkpoint.

No fitting, threshold selection, or synthetic teacher labels on these episodes.
Visual features must use the same stride-one causal Wan preprocessing.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps

from .fit_ood_v4 import episode_seed, load, save_json, sha
from .monitor import OnlineMonitor
from .temporal_evidence import EVIDENCE_NAMES
from .visual_features import resolve_video_episodes


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [{'start_frame': int(a), 'end_frame': int(b-1), 'frames': int(b-a),
             'start_seconds': float(a/30), 'end_seconds': float((b-1)/30)}
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def make_media(video, folder, report):
    folder.mkdir(parents=True, exist_ok=True)
    signature = dict(source_sha256=sha(video.video_path), frames=video.length,
                     start_time_seconds=video.start_time_seconds)
    stamp = folder/'media.json'
    if stamp.exists():
        saved = json.loads(stamp.read_text())
        if all(saved.get(k) == v for k, v in signature.items()):
            assert all((report/p).is_file() for p in saved['sprites'])
            return saved
    cols, rows, width, height = 10, 5, 384, 216
    sheet = Image.new('RGB', (cols*width, rows*height), '#101820')
    sheets, count = [], 0
    with av.open(str(video.video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.time is not None and frame.time + .5/video.fps < video.start_time_seconds:
                continue
            if count == video.length:
                break
            thumb = ImageOps.pad(frame.to_image(), (width, height), color='#101820')
            if count == 0:
                thumb.save(folder/'poster.jpg', quality=90)
            slot = count % (cols*rows)
            sheet.paste(thumb, ((slot % cols)*width, (slot//cols)*height))
            count += 1
            if count % (cols*rows) == 0 or count == video.length:
                target = folder/f'sprite_{len(sheets):03d}.jpg'
                sheet.save(target, quality=85)
                sheets.append(str(target.relative_to(report)))
                sheet = Image.new('RGB', (cols*width, rows*height), '#101820')
    assert count == video.length, (video.episode_index, count, video.length)
    result = dict(signature, source=str(video.video_path), fps=video.fps, sprites=sheets,
                  columns=cols, rows=rows, tile_width=width, tile_height=height,
                  poster=str((folder/'poster.jpg').relative_to(report)))
    save_json(result, stamp)
    return result


def prepare(args):
    info = json.loads((args.data_root/'meta/info.json').read_text())
    if info['fps'] != 30:
        raise ValueError('Current model requires original continuous 30 FPS input')
    files = sorted((args.data_root/'data').rglob('*.parquet'))
    table = pq.read_table(files, columns=['state.joints', 'state.gripper_w', 'action.joints',
                                         'action.gripper_w', 'episode_index', 'frame_index', 'timestamp'])
    states = np.concatenate([np.stack(table[k].to_pylist()) for k in ('state.joints', 'state.gripper_w')], -1).astype('float32')
    actions = np.concatenate([np.stack(table[k].to_pylist()) for k in ('action.joints', 'action.gripper_w')], -1).astype('float32')
    ids = table['episode_index'].to_numpy()
    videos = {v.episode_index: v for v in resolve_video_episodes(args.data_root, 'global_image')}
    records = []
    for index in args.episodes:
        v = videos[index]
        chosen = np.flatnonzero(ids == index)
        order = np.argsort(table['frame_index'].to_numpy()[chosen])
        chosen = chosen[order]
        np.testing.assert_array_equal(table['frame_index'].to_numpy()[chosen], np.arange(v.length))
        timestamp = table['timestamp'].to_numpy()[chosen]
        np.testing.assert_allclose(timestamp, np.arange(v.length)/30, atol=1e-5, rtol=1e-6)
        assert states[chosen].shape == (v.length, 14) and np.isfinite(states[chosen]).all()
        key = f'pant_fail_ep{index:06d}'
        np.savez_compressed(args.run/f'{key}_input.npz', states=states[chosen], timestamps=timestamp)
        media = make_media(v, args.report/'media'/key, args.report)
        records.append(dict(id=key, episode_index=index, source_root=str(args.data_root.resolve()), length=v.length,
                            fps=30, state_sha256=hashlib.sha256(states[chosen].tobytes()).hexdigest(),
                            action_equals_state=bool(np.array_equal(actions[chosen], states[chosen])), media=media))
    sources = {str(p): sha(p) for p in files + [args.data_root/'meta/info.json'] + sorted((args.data_root/'meta/episodes').rglob('*.parquet'))}
    existing = json.loads(Path('runs/ood_v4_2000_20260920/dim128/ood/manifest.json').read_text())
    feature_root = Path('runs/recovery_line_v2_2000/features')
    duplicates = []
    for new in records:
        for old in existing['records']:
            assert Path(old['source_root']).resolve() != args.data_root.resolve()
            if old['length'] != new['length']:
                continue
            data = load(feature_root/old['file'])['states'].numpy().astype('float32')
            if hashlib.sha256(data.tobytes()).hexdigest() == new['state_sha256']:
                duplicates.append(dict(new=new['id'], existing=old['file'], split=old['split']))
    result = dict(complete=True, records=records, selected_episodes=args.episodes,
                  dataset_total_episodes=info['total_episodes'], source_sha256=sources,
                  existing_state_duplicates=duplicates,
                  selection='All four episodes explicitly requested; fixed existing fold_all and threshold; no fitting')
    save_json(result, args.run/'input_manifest.json')
    print(json.dumps(dict(prepared=True, episodes=len(records), frames=sum(r['length'] for r in records), duplicates=duplicates)), flush=True)
    return result


def reference_statistics():
    source = Path('runs/ood_v4_2000_20260920/dim128/ood')
    manifest = json.loads((source/'manifest.json').read_text())
    array = np.load(source/'evidence.npy', mmap_mode='r')
    values = np.concatenate([array[r['offset']:r['offset']+r['length']] for r in manifest['records'] if r['split']=='calibration'])
    return {name: dict(zip(['p50', 'p95', 'p99'], np.quantile(values[:, j], [.5, .95, .99]).tolist()))
            for j, name in enumerate(EVIDENCE_NAMES)}


def evaluate(args, manifest):
    model = OnlineMonitor(args.bundle, device=args.device)
    assert model.kind == 'ordered_duration_v7'
    bundle_hash = sha(args.bundle)
    reference = reference_statistics()
    episodes, summaries = [], []
    keys = ['phase_probs', 'boundary_cdf', 'boundary_evidence', 'duration_log_bias',
            'confirmed_phase', 'observation_boundary_probs', 'change_sizes', 'accepted_phase',
            'risk_score', 'frame_risk_score', 'risk_threshold_ratio', 'alarm']
    start = time.monotonic()
    for record in manifest['records']:
        key = record['id']
        states = np.load(args.run/f'{key}_input.npz')['states']
        visual_path = args.run/'visual'/f'episode_{record["episode_index"]:06d}.npz'
        with np.load(visual_path) as cache:
            visual = cache['features']
            np.testing.assert_array_equal(cache['encoded_indices'], np.arange(record['length']))
        assert visual.shape == (record['length'], 48) and np.isfinite(visual).all()
        seed = episode_seed(record)
        model.reset(episode_seed=seed)
        outputs = []
        for t, (state, feature) in enumerate(zip(states, visual)):
            outputs.append(model.step(state, feature, debug=True))
            if (t+1) % 500 == 0:
                print(json.dumps(dict(episode=record['episode_index'], frame=t+1, elapsed=time.monotonic()-start)), flush=True)
        result = {name: np.asarray([row[name] for row in outputs]) for name in keys}
        evidence = np.stack([row['debug']['evidence'] for row in outputs])
        result['evidence'] = evidence
        result['latent'] = np.stack([row['debug']['latent'] for row in outputs])
        raw = result['frame_risk_score']
        c = np.r_[0., raw.cumsum()]; t = np.arange(len(raw)); left = np.maximum(0, t-29)
        result['smoothed_risk_score'] = (c[t+1]-c[left])/(t+1-left)
        np.testing.assert_allclose(result['phase_probs'].sum(-1), 1, atol=1e-6)
        np.testing.assert_array_equal(result['alarm'], result['risk_score'] > model.threshold)
        assert np.all((np.diff(result['confirmed_phase']) >= 0) & (np.diff(result['confirmed_phase']) <= 1))
        assert (np.diff(result['boundary_cdf'], axis=0) >= -1e-6).all()
        np.savez_compressed(args.run/f'{key}_predictions.npz', **result)
        spans = intervals(result['alarm'])
        peak = int(result['risk_score'].argmax())
        diagnostics = {name: dict(peak=float(evidence[:, j].max()), peak_frame=int(evidence[:, j].argmax()),
                                 fraction_above_calibration_p99=float((evidence[:, j] > reference[name]['p99']).mean()),
                                 value_at_risk_peak=float(evidence[peak, j])) for j, name in enumerate(EVIDENCE_NAMES[:3])}
        summary = dict(id=key, episode_index=record['episode_index'], frames=len(raw), duration_seconds=len(raw)/30,
                       alarm=bool(spans), alarm_frames=int(result['alarm'].sum()), alarm_fraction=float(result['alarm'].mean()),
                       first_alarm_frame=spans[0]['start_frame'] if spans else None,
                       first_alarm_seconds=spans[0]['start_seconds'] if spans else None,
                       peak_risk=float(result['risk_score'][peak]), peak_frame=peak,
                       peak_threshold_ratio=float(result['risk_threshold_ratio'][peak]), threshold=model.threshold,
                       alarm_intervals=spans, longest_alarm_frames=max((s['frames'] for s in spans), default=0),
                       final_confirmed_phase=int(result['confirmed_phase'][-1]), diagnostics=diagnostics)
        summaries.append(summary)
        json_result = {name: value.tolist() for name, value in result.items() if name not in ('latent',)}
        json_result.update(fold='all', split='external_test', threshold=model.threshold, csv=f'data/{key}.csv')
        csv_fields = ['frame', 'seconds', 'confirmed_phase', 'accepted_phase', *[f'P{k}' for k in range(1, 6)],
                      'frame_risk_score', 'smoothed_risk_score', 'risk_score', 'threshold', 'risk_threshold_ratio', 'alarm',
                      *[f'boundary_cdf_{k}' for k in range(1, 5)], *[f'duration_log_bias_{k}' for k in range(1, 5)], *EVIDENCE_NAMES]
        with (args.report/json_result['csv']).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields); writer.writeheader()
            for t in range(len(raw)):
                row = dict(frame=t, seconds=t/30, threshold=model.threshold)
                for name in ['confirmed_phase', 'accepted_phase', 'frame_risk_score', 'smoothed_risk_score', 'risk_score', 'risk_threshold_ratio', 'alarm']:
                    row[name] = result[name][t].item()
                row.update({f'P{k+1}': float(result['phase_probs'][t, k]) for k in range(5)})
                for name in ['boundary_cdf', 'duration_log_bias']:
                    row.update({f'{name}_{k+1}': float(result[name][t, k]) for k in range(4)})
                row.update({name: float(evidence[t, j]) for j, name in enumerate(EVIDENCE_NAMES)})
                writer.writerow(row)
        durations = np.diff(np.r_[0, np.flatnonzero(np.diff(result['confirmed_phase']) > 0)+1]).tolist()
        episodes.append(dict(id=key, task='test_lwy', task_label='test_lwy · pant_fail',
                             episode_index=record['episode_index'], episode_label=f'pant_fail · EP {record["episode_index"]}',
                             length=record['length'], fps=30, ood=True, tail_start=None, media=record['media'],
                             source_root=record['source_root'], evaluation_kind='external_test',
                             split_label='新增外部测试 · fold_all · 未参与训练/调阈值',
                             result=json_result, baseline=None, teacher_phase=None, teacher_probs=None,
                             durations=dict(v6=[], v7=durations), summary=summary))
        print(json.dumps(summary), flush=True)
    assert sha(args.bundle) == bundle_hash
    summary = dict(complete=True, bundle=str(args.bundle.resolve()), bundle_sha256=bundle_hash,
                   stage_kind=model.kind, threshold=model.threshold, selected_episodes=args.episodes,
                   source_root=str(args.data_root.resolve()), episodes=summaries, frames=sum(s['frames'] for s in summaries),
                   alarmed_episodes=sum(s['alarm'] for s in summaries), reference_statistics=reference,
                   evidence_names=EVIDENCE_NAMES, existing_state_duplicates=manifest['existing_state_duplicates'],
                   feature_semantics='global_image, Wan VAE 256, causal five-frame window, stride1, spatial/temporal mean48',
                   label_semantics='User-supplied failure recordings; no frame-level OOD or irrecoverability truth',
                   protocol='Existing fold_all bundle and fixed calibration threshold; no training or threshold fitting',
                   seconds=time.monotonic()-start)
    save_json(summary, args.run/'evaluation.json')
    save_json(dict(episodes=episodes, evaluation=summary), args.run/'report_extension.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/data/dataset/laudry_fold_dataset/test_lwy/pant_fail'))
    parser.add_argument('--episodes', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--run', type=Path, default=Path('runs/pant_fail_v7_20260921'))
    parser.add_argument('--report', type=Path, default=Path('reports/ordered_duration_v7_20260921'))
    parser.add_argument('--bundle', type=Path, default=Path('models/v7/fold_all.pt'))
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    args.run.mkdir(parents=True, exist_ok=True); (args.report/'data').mkdir(parents=True, exist_ok=True)
    manifest = prepare(args)
    if not args.prepare_only:
        evaluate(args, manifest)


if __name__ == '__main__':
    main()
