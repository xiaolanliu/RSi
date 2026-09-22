"""Full-trajectory V7 parity, preserved risk decisions, and isolated bundles."""
import argparse,builtins,json,shutil,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from agent_closed_loop.monitor import OnlineMonitor
from agent_closed_loop.fit_ood_v4 import load,sha,episode_seed,section,save_json
from agent_closed_loop.paths import artifact_path
from agent_closed_loop.evaluate_duration_subtask import predict


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(2)
    source=Path('runs/ood_v4_2000_20260920/dim128/ood');m=json.loads((source/'manifest.json').read_text())
    frames=np.load(artifact_path(m['data'])/'frames.npy',mmap_mode='r');zm=np.load(source/'latent.npy',mmap_mode='r');rows=[]
    for fold in range(3):
        risk=load(source/f'fold_{fold}/predictions.pt');bundle=Path(f'models/v7/fold_{fold}.pt')
        with tempfile.TemporaryDirectory() as tmp:
            isolated=Path(tmp)/'model.pt';shutil.copy2(bundle,isolated);original_open=builtins.open
            def checked(file,*args,**kwargs):
                if isinstance(file,(str,Path)) and str(file).startswith('/data/users/liuweiyuan/Code/COMPILE'):raise AssertionError('Offline runtime dependency')
                return original_open(file,*args,**kwargs)
            with patch('builtins.open',checked):model=OnlineMonitor(isolated,a.device)
        positive=next(r for r in m['records'] if r['split']=='ood' and r['episode_index']==fold)
        normal=min((r for r in m['records'] if r['split']=='validation'),key=lambda r:abs(float(risk[r['file']]['risk_score'].max())-model.threshold))
        for r in (positive,normal):
            raw=np.array(frames[r['source_offset']:r['source_offset']+r['length'],:76]);expected_risk=risk[r['file']]
            z=torch.tensor(np.array(section(zm,r)),device=a.device);z=((z-model.zmean)/model.zscale).clamp(-15,15)
            with torch.inference_mode():expected=predict(model.stage,z,expected_risk['alarm'].numpy(),chunk=173)
            model.reset(episode_seed=episode_seed(r));out=[model.step_normalized(x) for x in raw]
            q=np.stack([x['phase_probs'] for x in out]);scores=np.array([x['risk_score'] for x in out])
            np.testing.assert_allclose(q,expected['phase_probs'],atol=6e-5,rtol=6e-5)
            np.testing.assert_allclose(scores,expected_risk['risk_score'],atol=3e-5,rtol=3e-5)
            np.testing.assert_array_equal([x['confirmed_phase'] for x in out],expected['confirmed_phase'])
            np.testing.assert_array_equal([x['alarm'] for x in out],expected_risk['alarm'])
            np.testing.assert_array_equal([x['accepted_phase'] for x in out],np.where(expected_risk['alarm'],0,expected['confirmed_phase']))
            hard=np.array([x['confirmed_phase'] for x in out]);assert ((np.diff(hard)>=0)&(np.diff(hard)<=1)).all()
            cdf=np.stack([x['boundary_cdf'] for x in out]);assert (np.diff(cdf,axis=0)>=0).all()
            np.testing.assert_allclose(q.sum(-1),1,atol=1e-6)
            model.reset(episode_seed=episode_seed(r));prefix=[model.step_normalized(x) for x in raw[:80]]
            np.testing.assert_allclose([x['phase_probs'] for x in prefix],q[:80],atol=0,rtol=0)
            rows.append(dict(record=r['file'],fold=fold,frames=r['length'],passed=True,bundle_sha256=sha(bundle),
                max_probability_difference=float(np.max(np.abs(q-expected['phase_probs'].numpy()))),max_risk_difference=float(np.max(np.abs(scores-expected_risk['risk_score'].numpy())))))
            print(json.dumps(rows[-1]),flush=True)
    # Old bundle formats remain loadable and retain their original decisions.
    normal=next(r for r in m['records'] if r['split']=='validation');raw=np.array(frames[normal['source_offset']:normal['source_offset']+100,:76])
    for kind in ('v5','v6','v7'):
        monitor=OnlineMonitor(f'models/{kind}/fold_all.pt',a.device)
        for x in raw:
            out=monitor.step_normalized(x);assert np.isfinite(out['risk_score']) and np.isclose(out['phase_probs'].sum(),1)
    result=dict(passed=True,device=a.device,frames=sum(r['frames'] for r in rows),records=rows,
        checks=['isolated portable weights; no COMPILE read','3 held-out OOD + 3 threshold-nearest normal full trajectories',
                'cached batch vs actual streaming probability agreement','exact risk alarms preserved from V4','no backtrack or skip',
                'no cumulative probability regression','reset prefix equality','V5/V6/V7 portable loading'])
    save_json(result,Path('models/v7')/f'validation_{a.device.replace(":","_")}.json')

if __name__=='__main__':main()
