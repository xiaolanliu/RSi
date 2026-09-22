import numpy as np
import torch
from agent_closed_loop.online_subtask import OrderedConfig,OrderedSubtaskHead,ordered_memory,StageConfirmation,OnlineRiskHead


def test_memory_conservation_causality_no_skips():
    torch.manual_seed(7);e=torch.rand(3,250,4)
    c,q,last=ordered_memory(e)
    assert q.min()>=0;torch.testing.assert_close(q.sum(-1),torch.ones(3,250))
    assert (c[:,1:]-c[:,:-1]).min()>=0
    prev=torch.cat((torch.tensor([1.,0,0,0,0])[None,None].expand(3,1,5),q[:,:-1]),1)
    flow=c-torch.cat((torch.zeros(3,1,4),c[:,:-1]),1)
    assert (flow<=prev[:,:,:4]+1e-7).all()
    a,b,mem=ordered_memory(e[:,:73]);d,f,_=ordered_memory(e[:,73:],mem)
    torch.testing.assert_close(torch.cat((a,d),1),c);torch.testing.assert_close(torch.cat((b,f),1),q)
    changed=e.clone();changed[:,73:]=torch.rand_like(changed[:,73:]);torch.testing.assert_close(ordered_memory(changed)[1][:,:73],q[:,:73])


def test_constant_weak_evidence_does_not_accumulate():
    c,q,_=ordered_memory(torch.full((1,30000,4),.01))
    torch.testing.assert_close(c[:,5:],c[:,5:6].expand_as(c[:,5:]))
    assert q[0,-1,0]>.98
    control=StageConfirmation()
    for row in c[0]:assert control.step(row)==(1,1)
    strong=torch.zeros(1,200,4);strong[:,:,0]=.99
    c,_,_=ordered_memory(strong);control.reset()
    phases=[control.step(row)[0] for row in c[0]];assert max(phases)==2


def test_head_chunk_step_reset_and_future():
    torch.manual_seed(13);model=OrderedSubtaskHead(OrderedConfig(dimension=16,hidden=16)).eval();x=torch.randn(1,310,16)
    with torch.no_grad():
        whole,_=model(x);a,mem=model(x[:,:117]);b,_=model(x[:,117:],mem)
        for k in whole:torch.testing.assert_close(torch.cat((a[k],b[k]),1),whole[k],atol=2e-6,rtol=2e-5)
        cache=None;rows=[]
        for t in range(len(x[0])):out,cache=model.step(x[:,t],cache);rows.append(out['phase_probs'])
        torch.testing.assert_close(torch.stack(rows,1),whole['phase_probs'],atol=2e-6,rtol=2e-5)
        changed=x.clone();changed[:,117:]+=10;future,_=model(changed)
        torch.testing.assert_close(future['phase_probs'][:,:117],whole['phase_probs'][:,:117])
        reset,_=model(x);torch.testing.assert_close(reset['phase_probs'],whole['phase_probs'])


def test_confirmation_rejection_and_risk_cache():
    confirm=StageConfirmation(frames=3);history=[]
    for t in range(60):
        phase,accepted=confirm.step([.99]*4,alarm=5<=t<30);history.append(phase)
        if 5<=t<30:assert accepted==0 and phase==2
    assert all(0<=b-a<=1 for a,b in zip(history,history[1:]))
    assert max(history)==5
    torch.manual_seed(2);model=OnlineRiskHead(16).eval();z=torch.randn(1,160,16);e=torch.randn(1,160,7);p=torch.randn(1,160,9)
    with torch.no_grad():
        full=model(z,e,p);cache=None;rows=[]
        for t in range(160):out,cache=model.step(z[:,t],e[:,t],p[:,t],cache);rows.append(out)
        torch.testing.assert_close(torch.stack(rows,1),full,atol=2e-6,rtol=2e-5)


if __name__=='__main__':
    torch.set_num_threads(2)
    for name,fn in list(globals().items()):
        if name.startswith('test_'):fn();print(name,'PASS',flush=True)
