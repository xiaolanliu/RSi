import numpy as np
import torch
from agent_closed_loop.uncertainty_ood import FeatureLINe,SingleRiskHead
from agent_closed_loop.temporal_evidence import time_evidence,HistoryEvidence


def test_line_fitting_and_serialization():
    torch.manual_seed(5);z=torch.rand(400,32);y=torch.rand(400,5);fc=torch.nn.Linear(32,5)
    m=FeatureLINe(32);m.fit(z,y,fc)
    e,p=m(z);torch.testing.assert_close(p,fc(z));assert torch.isfinite(e).all()
    copy=FeatureLINe(32);copy.load_state_dict(m.state_dict());torch.testing.assert_close(e,copy(z)[0])
    try:m.fit(z,y,fc,split='ood')
    except ValueError:pass
    else:raise AssertionError('OOD fitting accepted')


def test_additive_risk_prefix_and_cache():
    torch.manual_seed(6);m=SingleRiskHead(32).eval();z=torch.randn(1,120,32);e=torch.randn(1,120,7)
    with torch.no_grad():
        full=m(z,e);torch.testing.assert_close(full[:,:51],m(z[:,:51],e[:,:51]),atol=2e-5,rtol=2e-5)
        bigger=e.clone();bigger[:,:,:3]+=3
        assert torch.all(m(z,bigger)>full)
        cache=None;outputs=[]
        for t in range(120):
            v,cache=m.step(z[:,t],e[:,t],cache);outputs.append(v)
        torch.testing.assert_close(full,torch.stack(outputs,1),atol=2e-5,rtol=2e-5)


def test_line_reference_exceeds_torch_quantile_limit():
    z=torch.full((262145,64),.25);y=torch.ones(len(z),5)
    m=FeatureLINe(64);result=m.fit(z,y,torch.nn.Linear(64,5))
    assert z.numel()>2**24 and result['clip']==.25


def test_temporal_cache_and_normal_counterexamples():
    t=np.arange(490)/30.;s=np.repeat(np.sin(2*np.pi*t)[:,None],14,axis=1);static=np.zeros((len(t),32));moving=np.repeat(t[:,None],32,axis=1)
    scale=np.ones(32);importance=np.ones(32)
    full=time_evidence(s,static,scale,importance)
    assert full[150:,0].mean()>.8
    assert time_evidence(s,moving,scale,importance)[150:,0].max()<1e-4
    assert time_evidence(s*0,static,scale,importance)[:,0].max()==0
    cache=HistoryEvidence(scale,importance);stream=np.stack([cache.step(a,b) for a,b in zip(s,static)])
    np.testing.assert_allclose(full,stream,atol=1e-5)
    rng=np.random.default_rng(19)
    s=s.astype('float32');moving=(moving+.02*rng.normal(size=moving.shape)).astype('float32')
    full=time_evidence(s,moving,scale,importance);cache=HistoryEvidence(scale,importance)
    stream=np.stack([cache.step(a,b) for a,b in zip(s,moving)])
    np.testing.assert_allclose(full,stream,atol=1e-5)


if __name__=='__main__':
    torch.set_num_threads(2)
    for k,fn in list(globals().items()):
        if k.startswith('test_'):fn();print(k,'PASS')
