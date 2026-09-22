"""Full episodes: actual streaming vs independently cached feature evaluation."""
import argparse,json,tempfile,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from agent_closed_loop.monitor import OnlineMonitor
from agent_closed_loop.fit_ood_v4 import load,save_json,sha,episode_seed,section
from agent_closed_loop.paths import artifact_path
from agent_closed_loop.evaluate_duration_subtask import predict
from agent_closed_loop.three_signal_ood import SIGNAL_NAMES


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--version',choices=['v8','v9','v10','v11'],default='v8');a=p.parse_args();torch.set_num_threads(2)
    runs=dict(v8='three_signal_v8_20260921',v9='action_dynamics_v9_20260921',v10='early_repetition_v10_20260921',v11='command_constraints_v11_20260921')
    root=Path('runs')/runs[a.version];evaluation=json.loads((root/'evaluation.json').read_text());saved=load(root/'predictions.pt')
    source=Path('runs/ood_v4_2000_20260920/dim128/ood');manifest=json.loads((source/'manifest.json').read_text())
    frames=np.load(artifact_path(manifest['data'])/'frames.npy',mmap_mode='r');latent=np.load(source/'latent.npy',mmap_mode='r')
    bundle=Path(f'models/{a.version}/fold_all.pt')
    with tempfile.TemporaryDirectory() as tmp:
        copy=Path(tmp)/'bundle.pt';shutil.copy2(bundle,copy);monitor=OnlineMonitor(copy,a.device)
    nearest=min((r for r in evaluation['records'] if r['split']=='validation'),key=lambda r:abs(r['peak_threshold_ratio']-1))
    chosen=[r for r in manifest['records'] if r['split']=='ood' or r['file']==nearest['file']]
    inputs=[]
    for r in chosen:
        inputs.append((r['file'],np.array(frames[r['source_offset']:r['source_offset']+r['length'],:76]),
                       np.array(section(latent,r)),saved[r['file']],False,episode_seed(r)))
    external=Path('runs/pant_fail_v7_20260921')
    for i in range(4):
        name=f'pant_fail_ep{i:06d}';state=np.load(external/f'{name}_input.npz')['states']
        visual=np.load(external/'visual'/f'episode_{i:06d}.npz')['features']
        z=np.load(external/f'{name}_predictions.npz')['latent']
        inputs.append((name,(state,visual),z,saved[name],True,0))
    rows=[]
    for name,values,z,expected,raw,seed in inputs:
        zn=((torch.tensor(z,device=a.device)-monitor.zmean)/monitor.zscale).clamp(-15,15)
        with torch.inference_mode():stage=predict(monitor.stage,zn,expected['alarm'].numpy())
        monitor.reset(episode_seed=seed)
        # New runtime must never compute readout disagreement.
        with patch('agent_closed_loop.monitor.readout_disagreement',side_effect=AssertionError('extra criterion')):
            out=([monitor.step(s,v,debug=True) for s,v in zip(*values)] if raw else
                 [monitor.step_normalized(v,debug=True) for v in values])
        e=np.stack([r['debug']['evidence'] for r in out]);score=np.array([r['risk_score'] for r in out]);hard=np.array([r['confirmed_phase'] for r in out])
        assert e.shape==(len(out),3) and list(out[0]['ood_signals'])==monitor.signal_names
        np.testing.assert_allclose(e,expected['evidence'],rtol=1e-4,atol=8e-5)
        np.testing.assert_allclose(score,expected['risk_score'],rtol=1e-4,atol=8e-5)
        np.testing.assert_array_equal([r['alarm'] for r in out],expected['alarm'])
        np.testing.assert_array_equal(hard,stage['confirmed_phase'])
        np.testing.assert_allclose([r['phase_probs'] for r in out],stage['phase_probs'],rtol=1e-4,atol=8e-5)
        assert ((np.diff(hard)>=0)&(np.diff(hard)<=1)).all()
        np.testing.assert_array_equal([r['accepted_phase'] for r in out],np.where(expected['alarm'],0,hard))
        monitor.reset(episode_seed=seed)
        prefix=([monitor.step(s,v) for s,v in zip(values[0][:65],values[1][:65])] if raw else
                [monitor.step_normalized(v) for v in values[:65]])
        np.testing.assert_array_equal([r['risk_score'] for r in prefix],score[:65])
        if raw:
            from agent_closed_loop.action_units import PACKED_TO_ARMS
            monitor.reset(episode_seed=seed)
            adapted=[monitor.step(s[PACKED_TO_ARMS],v,state_layout='left7_right7') for s,v in zip(values[0][:65],values[1][:65])]
            np.testing.assert_array_equal([r['risk_score'] for r in adapted],score[:65])
        row=dict(record=name,frames=len(out),max_risk_difference=float(np.max(np.abs(score-expected['risk_score'].numpy()))),passed=True)
        rows.append(row);print(json.dumps(row),flush=True)
    # Legacy dispatch remains loadable.
    for version in ('v5','v6','v7','v8','v9','v10'):
        old=OnlineMonitor(f'models/{version}/fold_all.pt',a.device)
        for v in inputs[0][1][:10]:assert np.isfinite(old.step_normalized(v)['risk_score'])
    save_json(dict(passed=True,device=a.device,bundle_sha256=sha(bundle),frames=sum(r['frames'] for r in rows),records=rows,
        checks=['three signals only; dropout readout forbidden','full-trajectory cached vs streaming alarms',
                'raw external state/visual input','reset prefix equality','no stage backtrack or skip','legacy loading']),
        Path(f'models/{a.version}')/f'validation_{a.device.replace(":","_")}.json')


if __name__=='__main__':main()
