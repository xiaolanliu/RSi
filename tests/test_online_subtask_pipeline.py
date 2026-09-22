"""Final-weight complete trajectory parity, causality, ordering and reset."""
from agent_closed_loop.paths import artifact_path
import argparse,json
from pathlib import Path
import numpy as np
import torch
from agent_closed_loop.fit_ood_v4 import load,sha,section,episode_seed,save_json,encode_episode
from agent_closed_loop.online_subtask_inference import OnlineSubtaskMonitor


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(2);m=json.loads((a.folder/'manifest.json').read_text());src=artifact_path(m['source'])
    source=json.loads((src/'manifest.json').read_text());frames=np.load(artifact_path(source['data'])/'frames.npy',mmap_mode='r')
    phase=np.load(a.folder/'ordered_phase.npy',mmap_mode='r');results=[]
    for fold in range(3):
        ck=a.folder/f'fold_{fold}/checkpoint.pt';model=OnlineSubtaskMonitor(ck,a.device)
        predictions=load(a.folder/f'fold_{fold}/predictions.pt')
        positive=next(r for r in m['records'] if r['split']=='ood' and r['episode_index']==fold)
        normal=min((r for r in m['records'] if r['split']=='validation'),key=lambda r:abs(float(predictions[r['file']]['risk_score'].max())-model.threshold))
        for r in [positive,normal]:
            x=np.array(frames[r['source_offset']:r['source_offset']+r['length'],:76]);target=section(phase,r);saved=predictions[r['file']]
            model.reset(episode_seed=episode_seed(r));out=[model.step_normalized(v) for v in x]
            q=np.stack([o['phase_probs'] for o in out]);c=np.stack([o['boundary_cdf'] for o in out]);risk=np.array([o['risk_score'] for o in out])
            np.testing.assert_allclose(q,target[:,:5],atol=3e-5,rtol=3e-5);np.testing.assert_allclose(c,target[:,5:],atol=3e-5,rtol=3e-5)
            np.testing.assert_allclose(risk,saved['risk_score'].numpy(),atol=3e-5,rtol=3e-5)
            for key in ['alarm','confirmed_phase','accepted_phase']:
                np.testing.assert_array_equal([o[key] for o in out],saved[key].numpy())
            confirmed=np.array([o['confirmed_phase'] for o in out]);assert ((np.diff(confirmed)>=0)&(np.diff(confirmed)<=1)).all()
            assert (np.diff(c,axis=0)>=-1e-7).all();np.testing.assert_allclose(q.sum(1),1,atol=1e-6)
            model.reset(episode_seed=episode_seed(r));prefix=[model.step_normalized(v) for v in x[:100]]
            np.testing.assert_allclose(np.stack([o['phase_probs'] for o in prefix]),q[:100],atol=1e-7)
            # Future changes leave the full bulk encoder/head prefix unchanged.
            changed=x.copy();changed[100:]+=2
            with torch.inference_mode():
                z=encode_episode(model.encoder,changed,a.device,chunk=173).to(a.device)
                z=((z-model.zmean)/model.zscale).clamp(-15,15);pred,_=model.ordered(z[None])
                np.testing.assert_allclose(pred['phase_probs'][0,:100].cpu().numpy(),q[:100],atol=3e-5,rtol=3e-5)
            # Raw API including observed backward delta, not supplied teacher.
            original=load(Path(json.loads((artifact_path(source['data'])/'manifest.json').read_text())['source'])/r['file'])
            model.reset(episode_seed=episode_seed(r))
            for t in range(100):
                raw=model.step(original['states'][t].numpy(),original['features'][t,-48:].numpy())
                np.testing.assert_allclose(raw['phase_probs'],q[t],atol=3e-5,rtol=3e-5)
            row=dict(record=r['file'],split=r['split'],fold=fold,frames=r['length'],passed=True,
                checkpoint_sha256=sha(ck),max_phase_difference=float(np.abs(q-target[:,:5]).max()),
                max_risk_difference=float(np.abs(risk-saved['risk_score'].numpy()).max()),
                confirmed_changes=int((np.diff(confirmed)>0).sum()))
            results.append(row);print(json.dumps(row),flush=True)
    result=dict(passed=True,device=a.device,records=results,phase_sha256=m['signature']['phase_sha256'],
        checks=['complete 3 held-out OOD + 3 threshold-nearest normal trajectories','soft probabilities and boundary memory',
        'risk/alarm/confirmed/accepted parity','reset','future perturbation prefix invariance','raw input normalization and backward delta',
        'no confirmed regression or skipping'],frames=sum(r['frames'] for r in results))
    save_json(result,a.folder/f'inference_validation_{a.device.replace(":","_")}.json')


if __name__=='__main__':main()
