"""Per-frame measured latency: portable monitor, components, and RGB end to end.

Component profiling synchronizes stage boundaries. Uninstrumented monitor and
RGB totals are measured in separate passes; their percentiles are not sums.
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import av
import numpy as np
import torch
from torch.nn import functional as F
from .fit_ood_v4 import episode_seed, save_json, sha
from .monitor import OnlineMonitor
from .streaming_fusion import readout_disagreement
from .visual_features import letterbox_frame


def stats(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), maximum=float(a.max()), total_seconds=float(a.sum()/1000))


@torch.inference_mode()
def detailed_step(stream, state, visual):
    times = {}; device = stream.device
    def clock():
        torch.cuda.synchronize(device)
        return time.perf_counter()
    start = clock()
    def mark(name):
        nonlocal start
        now = clock(); times[name] = (now-start)*1000; start = now
    values = stream.normalize(state, visual); mark('normalization_ms')
    x = torch.tensor(values, device=device)[None]; mark('input_transfer_ms')
    enc = stream.encoder; cache = {} if stream.encoder_cache is None else stream.encoder_cache
    u = enc.state(x[:, :14])+enc.motion(x[:, 14:28])
    q = enc._heads(enc.q(u[:, None])); k = enc._heads(enc.k(x[:, None, 28:76])); v = enc._heads(enc.v(x[:, None, 28:76]))
    current_v = v[:, :, 0].reshape_as(u); mark('state_motion_qkv_ms')
    if 'k' in cache:
        k = torch.cat((cache['k'], k), 2)[:, :, -enc.config.window:]
        v = torch.cat((cache['v'], v), 2)[:, :, -enc.config.window:]
    lag = torch.arange(k.shape[2]-1, -1, -1, device=device)
    bias = enc.relative_bias[:, lag].to(q.dtype)[None, :, None]
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.)[:, :, 0].reshape_as(u)
    x = u+enc.fusion_gate.sigmoid()*enc.out(y)+enc.visual_gate.sigmoid()*current_v; mark('causal_attention_fusion_ms')
    histories = []
    for block, buf in zip(enc.blocks, cache.get('blocks', [None]*len(enc.blocks))):
        x, buf = block.step(x, buf); histories.append(buf)
    z = F.relu(enc.latent(x)); stream.encoder_cache = dict(k=k, v=v, blocks=histories); mark('causal_tcn_latent_ms')
    energy, _ = stream.line(z); mark('line_ms')
    disagreement = readout_disagreement(z, enc.phase, torch.tensor([stream.frame], device=device), stream.seed,
                                       samples=stream.norm['dropout_samples'], dropout=stream.norm['phase_dropout']); mark('readout_uncertainty_ms')
    history = stream.history.step(values[:14], z[0].cpu().numpy()); mark('repeat_history_ms')
    evidence = torch.cat((energy[:, None], disagreement[:, None], torch.tensor(history, device=device)[None]), -1)
    zn = ((z-stream.zmean)/stream.zscale).clamp(-15, 15)
    en = ((evidence-stream.emean)/stream.escale).clamp(-15, 15); mark('evidence_normalization_ms')
    logit, stream.risk_cache = stream.head.step(zn, en, stream.risk_cache)
    raw = float(logit.sigmoid()[0]); mark('ood_head_ms')
    stream.raw_history.append(raw); stream.smooth_history.append(float(np.mean(stream.raw_history)))
    risk = min(stream.smooth_history) if len(stream.smooth_history) == stream.preprocessing['persist_frames'] else 0.
    if stream.frame < stream.preprocessing['minimum_history']:
        risk = 0.
    risk = float(np.float32(risk)); alarm = risk > stream.threshold; mark('risk_smoothing_ms')
    stage, stream.stage_cache = stream.stage.step(zn, stream.stage_cache, pause=[alarm]); mark('ordered_subtask_ms')
    confirmed = int(stage['confirmed_phase'][0])
    result = {k: stage[k][0].cpu().numpy() for k in ('phase_probs', 'boundary_cdf', 'boundary_evidence', 'duration_log_bias', 'observation_boundary_probs', 'change_sizes')}
    result.update(confirmed_phase=confirmed, accepted_phase=0 if alarm else confirmed,
                  frame_index=stream.frame, frame_risk_score=raw, risk_score=risk,
                  threshold=stream.threshold, risk_threshold_ratio=risk/stream.threshold, alarm=alarm)
    stream.frame += 1; mark('output_ms')
    return result, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['monitor', 'rgb'], required=True)
    parser.add_argument('--run', type=Path, default=Path('runs/pant_fail_v7_20260921'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--episodes', type=int, nargs='+', default=[0], help='Only these episodes are timed; default EP0')
    args = parser.parse_args(); torch.set_num_threads(2)
    manifest = json.loads((args.run/'input_manifest.json').read_text())
    manifest['records'] = [r for r in manifest['records'] if r['episode_index'] in args.episodes]
    assert manifest['records'], 'No selected episodes'
    evaluation = json.loads((args.run/'evaluation.json').read_text())
    started = time.perf_counter(); monitor = OnlineMonitor(evaluation['bundle'], args.device)
    vae = None
    if args.mode == 'rgb':
        sys.path.insert(0, '/data/users/liuweiyuan/Code/SIEVE')
        from third_party.wan22_vae.vae2_2 import Wan2_2_VAE
        vae_path = '/data/users/liuweiyuan/Code/SIEVE/weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth'
        vae = Wan2_2_VAE(vae_pth=vae_path, dtype=torch.bfloat16, device=args.device)
    init_seconds = time.perf_counter()-started
    first = manifest['records'][0]
    state0 = np.load(args.run/f'{first["id"]}_input.npz')['states'][0]
    visual0 = np.load(args.run/'visual'/f'episode_{first["episode_index"]:06d}.npz')['features'][0]
    # Warm kernels only; all episode/state caches are reset before measurement.
    if vae is not None:
        with av.open(first['media']['source']) as container:
            rgb = next(container.decode(video=0)).to_ndarray(format='rgb24')
        clip = torch.stack([letterbox_frame(rgb, 256)]*5, dim=1)[None].to(args.device, dtype=vae.dtype)
        with torch.inference_mode(), torch.autocast('cuda', dtype=vae.dtype):
            for _ in range(8):
                vae.model.encode(clip, vae.scale)
    for _ in range(80):
        monitor.step(state0, visual0)
    torch.cuda.synchronize(args.device)
    rows = []
    for record in manifest['records']:
        state = np.load(args.run/f'{record["id"]}_input.npz')['states']
        visual = np.load(args.run/'visual'/f'episode_{record["episode_index"]:06d}.npz')['features']
        expected = np.load(args.run/f'{record["id"]}_predictions.npz')
        monitor.reset(episode_seed=episode_seed(record)); measured = []; differences = []
        if args.mode == 'monitor':
            for t in range(len(state)):
                torch.cuda.synchronize(args.device); start = time.perf_counter()
                out = monitor.step(state[t], visual[t]); torch.cuda.synchronize(args.device)
                measured.append(dict(monitor_ms=(time.perf_counter()-start)*1000))
                assert out['alarm'] == expected['alarm'][t] and out['confirmed_phase'] == expected['confirmed_phase'][t]
                np.testing.assert_allclose(out['phase_probs'], expected['phase_probs'][t], atol=1e-5, rtol=1e-5)
            monitor.reset(episode_seed=episode_seed(record))
            for t in range(len(state)):
                out, components = detailed_step(monitor, state[t], visual[t])
                measured[t].update(components)
                assert out['alarm'] == expected['alarm'][t] and out['confirmed_phase'] == expected['confirmed_phase'][t]
                np.testing.assert_allclose(out['phase_probs'], expected['phase_probs'][t], atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(out['risk_score'], expected['risk_score'][t], atol=1e-8, rtol=1e-5)
        else:
            images = deque(maxlen=5); visuals = []; predictions = []
            with av.open(record['media']['source'], options={'threads': '1'}) as container:
                frames = iter(container.decode(video=0))
                for t in range(len(state)):
                    torch.cuda.synchronize(args.device); start = time.perf_counter()
                    frame = next(frames); decode_end = time.perf_counter()
                    images.append(letterbox_frame(frame.to_ndarray(format='rgb24'), 256))
                    window = [images[0]]*(5-len(images))+list(images)
                    clip = torch.stack(window, dim=1)[None].to(args.device, dtype=vae.dtype)
                    torch.cuda.synchronize(args.device); prep_end = time.perf_counter()
                    with torch.inference_mode(), torch.autocast('cuda', dtype=vae.dtype):
                        latent = vae.model.encode(clip, vae.scale).float()
                        feature = latent[0].mean(dim=(1, 2, 3)).cpu().numpy()
                    torch.cuda.synchronize(args.device); visual_end = time.perf_counter()
                    out = monitor.step(state[t], feature); torch.cuda.synchronize(args.device); end = time.perf_counter()
                    measured.append(dict(decode_ms=(decode_end-start)*1000, image_prepare_transfer_ms=(prep_end-decode_end)*1000,
                                         vae_ms=(visual_end-prep_end)*1000, rgb_monitor_ms=(end-visual_end)*1000,
                                         rgb_total_ms=(end-start)*1000))
                    visuals.append(feature); predictions.append(out)
                    if (t+1) % 500 == 0:
                        print(json.dumps(dict(mode=args.mode, episode=record['episode_index'], frame=t+1, total_seconds=time.perf_counter()-started)), flush=True)
            arrays = {k: np.asarray([o[k] for o in predictions]) for k in ('phase_probs', 'confirmed_phase', 'accepted_phase', 'risk_score', 'frame_risk_score', 'alarm')}
            arrays['visual'] = np.stack(visuals)
            np.savez_compressed(args.run/f'{record["id"]}_rgb_predictions.npz', **arrays)
            differences = dict(max_visual_difference=float(np.abs(arrays['visual']-visual).max()),
                               max_risk_difference=float(np.abs(arrays['risk_score']-expected['risk_score']).max()),
                               alarm_difference_frames=int((arrays['alarm'] != expected['alarm']).sum()),
                               confirmed_difference_frames=int((arrays['confirmed_phase'] != expected['confirmed_phase']).sum()),
                               note='Actual batch1 RGB profiling versus batch8 visual-cache evaluation; BF16 batch-shape roundoff may differ')
        times = {key: np.array([row[key] for row in measured]) for key in measured[0]}
        np.savez_compressed(args.run/f'{record["id"]}_latency_{args.mode}.npz', **times)
        row = dict(id=record['id'], episode_index=record['episode_index'], frames=len(state),
                   times={key: stats(value) for key, value in times.items()}, batch1_consistency=differences)
        rows.append(row); print(json.dumps(row), flush=True)
    output = dict(complete=True, mode=args.mode, device=args.device, hardware=torch.cuda.get_device_name(args.device),
                  batch_size=1, threads=torch.get_num_threads(), initialization_seconds=init_seconds,
                  warmup=dict(monitor_calls=80, vae_calls=8 if vae else 0), episodes=rows,
                  bundle_sha256=sha(evaluation['bundle']), total_frames=sum(r['frames'] for r in rows),
                  timing='Wall clock with CUDA synchronization; full monitor, components and RGB measured in separate passes; percentiles not additive',
                  exclusions='One-time model loading/warmup, parquet loading, camera/robot transport, report/saving; RGB includes MP4 decode and color conversion',
                  selected_episodes=args.episodes, all_trajectory_frames_included=True, wall_seconds=time.perf_counter()-started)
    save_json(output, args.run/f'latency_{args.mode}.json')


if __name__ == '__main__':
    main()
