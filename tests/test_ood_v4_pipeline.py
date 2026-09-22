"""Meaningful checkpoint integration checks; use smoke artifacts, then finals."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from agent_closed_loop.fit_ood_v4 import load,episode_seed,encode_episode,section,save_json
from agent_closed_loop.ood_v4_inference import OODV4Stream
from agent_closed_loop.streaming_fusion import readout_disagreement
from agent_closed_loop.temporal_evidence import time_evidence
from agent_closed_loop.recovery import intervention_trace
from agent_closed_loop.prepare_ood_v4 import sha
from agent_closed_loop.paths import artifact_path


def check_record(folder,manifest,norm,data,r,fold,device='cpu',limit=None):
    n=r['length'] if limit is None else min(r['length'],limit)
    values=np.array(data[r['source_offset']:r['source_offset']+n,:76])
    model=OODV4Stream(folder/f'fold_{fold}/checkpoint.pt',device,episode_seed=episode_seed(r))
    with torch.inference_mode():
        z=encode_episode(model.encoder,values,device,chunk=173).to(device)
        energy,logits=model.line(z)
        u=readout_disagreement(z,model.encoder.phase,torch.arange(n,device=device),episode_seed(r))
        h=time_evidence(values[:,:14],z.cpu().numpy(),norm['z_scale'].numpy(),norm['visual_importance'].numpy())
        e=torch.cat((energy[:,None],u[:,None],torch.tensor(h,device=device)),-1)
        raw=model.head(((z-model.zmean)/model.zscale).clamp(-15,15)[None],((e-model.emean)/model.escale).clamp(-15,15)[None])[0].sigmoid().cpu().numpy()
        risk=intervention_trace(raw);risk[:15]=0
    stream=[model.step_normalized(v,debug=True) for v in values]
    tol=1e-4
    np.testing.assert_allclose(np.stack([x['debug']['latent'] for x in stream]),z.cpu().numpy(),atol=tol,rtol=tol)
    np.testing.assert_allclose(np.stack([x['debug']['evidence'] for x in stream]),e.cpu().numpy(),atol=tol,rtol=tol)
    np.testing.assert_allclose([x['frame_risk_score'] for x in stream],raw,atol=tol,rtol=tol)
    np.testing.assert_allclose([x['risk_score'] for x in stream],risk,atol=tol,rtol=tol)
    # Also compare against predictions actually consumed by the web report.
    saved=load(folder/f'fold_{fold}/predictions.pt')[r['file']]
    np.testing.assert_allclose(raw,saved['frame_risk_score'][:n].numpy(),atol=tol,rtol=tol)
    np.testing.assert_allclose(risk,saved['risk_score'][:n].numpy(),atol=tol,rtol=tol)
    np.testing.assert_array_equal(risk>model.threshold,saved['alarm'][:n].numpy())
    # Near-threshold decisions need separate accounting, not an allclose claim.
    decisions=np.array([x['alarm'] for x in stream]);offline=risk>model.threshold
    disagreements=np.flatnonzero(decisions!=offline)
    if len(disagreements):raise AssertionError(f'Alarm parity failed at {disagreements.tolist()}')
    model.reset(episode_seed=episode_seed(r));again=[model.step_normalized(v) for v in values[:33]]
    np.testing.assert_allclose([x['risk_score'] for x in again],[x['risk_score'] for x in stream[:33]],atol=1e-7)
    # Normalization matches original raw source, including episode-start delta.
    source=json.loads((artifact_path(manifest['data'])/'manifest.json').read_text())['source']
    arrays=load(Path(source)/r['file']);s=arrays['states'].numpy();v=arrays['features'][:,-48:].numpy()
    model.reset(episode_seed=episode_seed(r))
    for t in range(33):np.testing.assert_allclose(model.normalize(s[t],v[t]),values[t],atol=1e-6,rtol=1e-6)
    # A future alteration must leave the prefix unchanged under chunked encode.
    changed=values.copy();changed[200:]+=4
    with torch.inference_mode():new=encode_episode(model.encoder,changed,device,chunk=173)
    torch.testing.assert_close(new[:200],z[:200].cpu(),atol=tol,rtol=tol)
    result=dict(passed=True,frames=n,device=device,dimension=manifest['dimension'],smoke=manifest['signature']['smoke'],
        record=r['file'],fold=fold,split=r['split'],alarm_frames=int(offline.sum()),
        risk_checkpoint_sha256=sha(folder/f'fold_{fold}/checkpoint.pt'),
        checks=['chunked vs cached shared z','LINe/dropout/loop evidence parity','risk and alarm parity','episode reset',
                'raw input normalization and backward deltas','future perturbation prefix invariance','saved report prediction parity'],
        max_risk_difference=float(np.max(np.abs(np.array([x['risk_score'] for x in stream])-risk))))
    print(json.dumps(result),flush=True);return result


def check(folder,device='cpu',quick=False):
    manifest=json.loads((folder/'manifest.json').read_text());norm=load(folder/'normalization.pt')
    data=np.load(artifact_path(manifest['data'])/'frames.npy',mmap_mode='r')
    records=manifest['records'];by_file={r['file']:r for r in records};results=[]
    for fold in range(3):
        r=next(r for r in records if r['split']=='ood' and r['episode_index']==fold)
        results.append(check_record(folder,manifest,norm,data,r,fold,device,410 if quick else None))
        if not quick:
            predictions=load(folder/f'fold_{fold}/predictions.pt')
            threshold=load(folder/f'fold_{fold}/checkpoint.pt')['threshold']
            normal=[(k,p) for k,p in predictions.items() if p['split']=='id_test']
            # Hardest decision boundary in this fold, selected for numerical
            # validation only. It never feeds training or threshold selection.
            key,_=min(normal,key=lambda item:abs(float(item[1]['risk_score'].max())-threshold))
            results.append(check_record(folder,manifest,norm,data,by_file[key],fold,device))
    result=dict(passed=True,device=device,dimension=manifest['dimension'],smoke=manifest['signature']['smoke'],quick=quick,
                backbone_sha256=manifest['signature']['checkpoint_sha256'],normalization_sha256=sha(folder/'normalization.pt'),
                frames=sum(r['frames'] for r in results),records=results,max_risk_difference=max(r['max_risk_difference'] for r in results))
    if not quick:save_json(result,folder/f'inference_validation_{device.replace(":","_")}.json')
    print(json.dumps(dict(passed=True,frames=result['frames'],records=len(results),max_risk_difference=result['max_risk_difference'])),flush=True)



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--quick',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);check(a.folder,a.device,a.quick)
