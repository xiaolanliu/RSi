"""Verify new-data input alignment and full streaming outputs independently."""
import json
from pathlib import Path
import numpy as np
import torch
from agent_closed_loop.fit_ood_v4 import encode_episode, episode_seed, sha
from agent_closed_loop.monitor import OnlineMonitor
from agent_closed_loop.temporal_evidence import time_evidence
from agent_closed_loop.streaming_fusion import readout_disagreement
from agent_closed_loop.recovery import intervention_trace
from agent_closed_loop.evaluate_duration_subtask import predict


def main():
    torch.set_num_threads(2)
    root = Path('runs/pant_fail_v7_20260921')
    manifest = json.loads((root/'input_manifest.json').read_text())
    evaluation = json.loads((root/'evaluation.json').read_text())
    assert sha(evaluation['bundle']) == evaluation['bundle_sha256']
    model = OnlineMonitor(evaluation['bundle'], device='cuda:1')
    assert model.threshold == evaluation['threshold']
    for path, digest in manifest['source_sha256'].items():
        assert sha(path) == digest
    rows = []
    for record in manifest['records']:
        state = np.load(root/f'{record["id"]}_input.npz')['states']
        visual = np.load(root/'visual'/f'episode_{record["episode_index"]:06d}.npz')['features']
        expected = np.load(root/f'{record["id"]}_predictions.npz')
        delta = np.zeros_like(state); delta[1:] = np.diff(state, axis=0)
        x = np.clip((np.concatenate((state, delta, visual), axis=1)-model.input_mean)/model.input_scale, -15, 15).astype('float32')
        with torch.inference_mode():
            z = encode_episode(model.encoder, x, 'cuda:1', chunk=173).to('cuda:1')
            energy, _ = model.line(z)
            disagreement = readout_disagreement(z, model.encoder.phase, torch.arange(len(x), device='cuda:1'), episode_seed(record),
                                               samples=model.norm['dropout_samples'], dropout=model.norm['phase_dropout'])
            history = time_evidence(x[:, :14], z.cpu().numpy(), model.norm['z_scale'].numpy(), model.norm['visual_importance'].numpy())
            evidence = torch.cat((energy[:, None], disagreement[:, None], torch.tensor(history, device='cuda:1')), -1)
            zn = ((z-model.zmean)/model.zscale).clamp(-15, 15)
            en = ((evidence-model.emean)/model.escale).clamp(-15, 15)
            raw = model.head(zn[None], en[None])[0].sigmoid().cpu().numpy()
            risk = intervention_trace(raw); risk[:model.preprocessing['minimum_history']] = 0
            alarms = risk > model.threshold
            stages = predict(model.stage, zn, alarms, chunk=173)
        np.testing.assert_allclose(z.cpu(), expected['latent'], atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(evidence.cpu(), expected['evidence'], atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(raw, expected['frame_risk_score'], atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(risk, expected['risk_score'], atol=3e-5, rtol=3e-5)
        np.testing.assert_array_equal(alarms, expected['alarm'])
        np.testing.assert_allclose(stages['phase_probs'], expected['phase_probs'], atol=6e-5, rtol=6e-5)
        np.testing.assert_array_equal(stages['confirmed_phase'], expected['confirmed_phase'])
        np.testing.assert_array_equal(np.where(alarms, 0, stages['confirmed_phase']), expected['accepted_phase'])
        rows.append(dict(id=record['id'], frames=len(x), passed=True,
                         max_risk_difference=float(np.max(np.abs(risk-expected['risk_score']))),
                         max_phase_difference=float(np.max(np.abs(stages['phase_probs'].numpy()-expected['phase_probs'])))))
        print(json.dumps(rows[-1]), flush=True)
    result = dict(passed=True, frames=sum(r['frames'] for r in rows), records=rows,
                  bundle_sha256=evaluation['bundle_sha256'], checks=['raw data hashes and frame alignment',
                  'full-frame independent chunk173 encoder, evidence, temporal risk and stage scan',
                  'exact alarm, confirmed and accepted decisions', 'fixed checkpoint and fixed threshold'])
    (root/'inference_validation.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
