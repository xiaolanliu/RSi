"""Behavioral tests: causal equivalence, prior gradient, ordered decisions, OOD."""
import unittest
import torch
from agent_closed_loop.duration_subtask import DurationConfig,DurationOrderedHead,ordered_duration,duration_penalty,duration_objective


class DurationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(7)

    def test_causal_chunk_and_frame_equivalence(self):
        model=DurationOrderedHead(DurationConfig()).eval();z=torch.randn(2,243,128)
        with torch.no_grad():
            whole,_=model(z);cache=None;rows=[]
            for part in z.split(31,dim=1):
                out,cache=model(part,cache);rows.append(out)
            for k in whole:torch.testing.assert_close(whole[k],torch.cat([r[k] for r in rows],1),atol=2e-5,rtol=2e-5)
            cache=None;rows=[]
            for t in range(243):out,cache=model.step(z[:,t],cache);rows.append(out)
            for k in whole:torch.testing.assert_close(whole[k],torch.stack([r[k] for r in rows],1),atol=2e-5,rtol=2e-5)
            z[:,110:]+=4;changed,_=model(z)
            for k in whole:torch.testing.assert_close(whole[k][:,:110],changed[k][:,:110],atol=0,rtol=0)

    def test_length_prior_has_centered_gradient(self):
        lengths=torch.tensor([100.,180.,200.,220.,500.],requires_grad=True)
        loss=duration_penalty(lengths);loss.sum().backward()
        self.assertEqual(float(loss[2]),0.)
        self.assertTrue((lengths.grad[:2]<0).all());self.assertTrue((lengths.grad[3:]>0).all())
        self.assertTrue((loss[[0,1,3,4]]>0).all())

    def test_no_backtrack_or_skip_and_no_banked_future(self):
        cfg=DurationConfig();logits=torch.ones(1,1200,4)
        support=torch.ones(1,1200)
        out,_=ordered_duration(logits,support,cfg)
        d=out['confirmed_phase'].diff();self.assertTrue(((d>=0)&(d<=1)).all())
        self.assertTrue((out['boundary_cdf'].diff(dim=1)>=0).all())
        torch.testing.assert_close(out['phase_probs'].sum(-1),torch.ones(1,1200))
        # All future boundaries high from the beginning: each must accrue age.
        starts=[int(torch.where(out['confirmed_phase'][0]>=k)[0][0]) for k in range(2,6)]
        self.assertTrue(all(b-a>60 for a,b in zip([0]+starts[:-1],starts)))
        # Weak scores or an unchanged latent cannot progress merely with age.
        for x,s in [(logits,torch.zeros_like(support)),(-logits,support)]:
            p,_=ordered_duration(x,s,cfg);self.assertTrue((p['confirmed_phase']==1).all())

    def test_alarm_freezes_memory_clears_evidence_and_chunk_parity(self):
        cfg=DurationConfig();logits=torch.ones(1,500,4);support=torch.ones(1,500)
        pause=torch.zeros(1,500,dtype=torch.bool);pause[:,100:300]=True
        whole,_=ordered_duration(logits,support,cfg,pause=pause)
        self.assertTrue((whole['confirmed_phase'][:,100:300]==whole['confirmed_phase'][:,99:100]).all())
        cache=None;rows=[]
        for t in range(500):
            out,cache=ordered_duration(logits[:,t:t+1],support[:,t:t+1],cfg,cache,pause[:,t:t+1]);rows.append(out['phase_probs'])
        torch.testing.assert_close(whole['phase_probs'],torch.cat(rows,1),atol=1e-6,rtol=1e-6)

    def test_padding_and_right_censoring(self):
        cfg=DurationConfig();model=DurationOrderedHead(cfg);z=torch.randn(1,220,128)
        out,_=model(z);y=torch.zeros(1,220,5);y[:,:,0]=1;valid=torch.ones(1,220,dtype=torch.bool)
        loss,metrics=duration_objective(out,y,valid,cfg);self.assertEqual(metrics['weighted_duration'],0.)
        padded={k:torch.cat((v,v[:,:20]),1) for k,v in out.items()}
        yp=torch.cat((y,torch.zeros(1,20,5)),1);vp=torch.cat((valid,torch.zeros(1,20,dtype=torch.bool)),1)
        other,_=duration_objective(padded,yp,vp,cfg);torch.testing.assert_close(loss,other)
        loss.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

if __name__=='__main__':unittest.main()
