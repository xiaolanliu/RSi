"""Complete-trajectory parity and isolated-weight loading for migrated monitors."""
import argparse,builtins,json,shutil,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from agent_closed_loop.monitor import OnlineMonitor
from agent_closed_loop.fit_ood_v4 import load,section,episode_seed,sha,encode_episode
from agent_closed_loop.paths import artifact_path
from agent_closed_loop.evaluate_change_subtask import replay
from agent_closed_loop.change_subtask import ChangeConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['v5','v6'],required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--evaluation',type=Path,default=Path('runs/latent_change_v6_20260920/memory_evaluation'));a=p.parse_args();torch.set_num_threads(2)
    source=Path('runs/ood_v4_2000_20260920/dim128/ood');m=json.loads((source/'manifest.json').read_text())
    frames=np.load(artifact_path(m['data'])/'frames.npy',mmap_mode='r');results=[]
    if a.kind=='v6':
        ev=json.loads((a.evaluation/'evaluation.json').read_text());config=ChangeConfig(**ev['config'])
        signals=np.load(a.evaluation/'signals.npy',mmap_mode='r');recent=np.load(a.evaluation/'recent_latent.npy',mmap_mode='r')
    else:phase=np.load('runs/online_subtask_v5_20260920/ood/ordered_phase.npy',mmap_mode='r')
    for fold in range(3):
        riskfolder=source if a.kind=='v6' else Path('runs/online_subtask_v5_20260920/ood')
        predictions=load(riskfolder/f'fold_{fold}/predictions.pt');bundle=Path(f'models/{a.kind}/fold_{fold}.pt')
        # The bundle is copied outside both projects. Forbid every attempted
        # read from COMPILE during load, and check no offline module is imported.
        with tempfile.TemporaryDirectory() as tmp:
            isolated=Path(tmp)/'monitor.pt';shutil.copy2(bundle,isolated);original_open=builtins.open
            def checked_open(file,*args,**kwargs):
                if isinstance(file,(str,Path)) and str(file).startswith('/data/users/liuweiyuan/Code/COMPILE'):raise AssertionError('Old-project runtime dependency')
                return original_open(file,*args,**kwargs)
            with patch('builtins.open',checked_open):model=OnlineMonitor(isolated,a.device)
        import sys
        assert 'compile' not in sys.modules
        positive=next(r for r in m['records'] if r['split']=='ood' and r['episode_index']==fold)
        normal=min((r for r in m['records'] if r['split']=='validation'),key=lambda r:abs(float(predictions[r['file']]['risk_score'].max())-model.threshold))
        for r in (positive,normal):
            raw=np.array(frames[r['source_offset']:r['source_offset']+r['length'],:76]);expected_risk=predictions[r['file']]
            if a.kind=='v6':expected=replay(section(signals,r),section(recent,r),config,expected_risk['alarm'].numpy())
            else:expected=dict(phase_probs=section(phase,r)[:,:5],**{k:expected_risk[k].numpy() for k in ('confirmed_phase','accepted_phase')})
            model.reset(episode_seed=episode_seed(r));out=[model.step_normalized(x) for x in raw]
            actual_q=np.stack([o['phase_probs'] for o in out]);risk=np.array([o['risk_score'] for o in out])
            np.testing.assert_allclose(actual_q,expected['phase_probs'],atol=5e-5,rtol=5e-5)
            np.testing.assert_allclose(risk,expected_risk['risk_score'],atol=3e-5,rtol=3e-5)
            for key in ('confirmed_phase','accepted_phase'):np.testing.assert_array_equal([o[key] for o in out],expected[key])
            np.testing.assert_array_equal([o['alarm'] for o in out],expected_risk['alarm'])
            hard=np.array([o['confirmed_phase'] for o in out]);assert ((np.diff(hard)>=0)&(np.diff(hard)<=1)).all()
            np.testing.assert_allclose(actual_q.sum(-1),1,atol=1e-6)
            model.reset(episode_seed=episode_seed(r));prefix=[model.step_normalized(x) for x in raw[:100]]
            np.testing.assert_allclose(np.stack([o['phase_probs'] for o in prefix]),actual_q[:100],atol=1e-7)
            # Perturb every future input, encode with a different chunk size.
            changed=raw.copy();changed[100:]+=3
            with torch.inference_mode():
                z=encode_episode(model.encoder,changed,a.device,chunk=173).to(a.device)
                normalized=((z-model.zmean)/model.zscale).clamp(-15,15);s,_=model.stage(normalized[None])
                if a.kind=='v6':
                    np.testing.assert_allclose(s['boundary_event_probs'][0,:100].cpu().numpy(),section(signals,r)[:100,:4],atol=5e-5,rtol=5e-5)
                else:np.testing.assert_allclose(s['phase_probs'][0,:100].cpu().numpy(),actual_q[:100],atol=5e-5,rtol=5e-5)
            data=json.loads((artifact_path(m['data'])/'manifest.json').read_text());record=load(artifact_path(data['source'])/r['file'])
            model.reset(episode_seed=episode_seed(r))
            for t in range(100):
                o=model.step(record['states'][t].numpy(),record['features'][t,-48:].numpy())
                np.testing.assert_allclose(o['phase_probs'],actual_q[t],atol=5e-5,rtol=5e-5)
            row=dict(record=r['file'],fold=fold,frames=r['length'],split=r['split'],passed=True,bundle_sha256=sha(bundle),
                max_probability_difference=float(np.abs(actual_q-expected['phase_probs']).max()),max_risk_difference=float(np.abs(risk-expected_risk['risk_score'].numpy()).max()))
            results.append(row);print(json.dumps(row),flush=True)
    result=dict(passed=True,kind=a.kind,device=a.device,frames=sum(r['frames'] for r in results),records=results,
        checks=['isolated portable bundle; COMPILE reads forbidden','no offline compile module import','3 full held-out OOD + 3 threshold-nearest normal trajectories',
            'phase/risk numerical parity; exact alarm/confirmed/accepted decisions','reset and raw backward-delta API','future perturbation and different chunk size','ordered confirmation, probability mass'])
    output=Path(f'models/{a.kind}/validation_{a.device.replace(":","_")}.json');output.write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
