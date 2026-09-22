"""Mathematical behaviour, causal persistence, and strictly three inputs."""
import unittest
import numpy as np
import torch
from agent_closed_loop.three_signal_ood import ProjectionMemory, ThreeSignalRisk, sustained_trace,calibration_tail
from agent_closed_loop.temporal_evidence import time_evidence
from agent_closed_loop.uncertainty_ood import FeatureLINe


class ThreeSignalTests(unittest.TestCase):
    def test_weight_changes_priority_not_empirical_percentile(self):
        ref=torch.arange(100).float().repeat(3,1)
        plain=ThreeSignalRisk(reference=ref);weighted=ThreeSignalRisk(reference=ref,weights=[1,1.5,1])
        x=torch.tensor([[0.,95.,0.],[95.,0.,0.]])
        torch.testing.assert_close(plain.percentiles(x),weighted.percentiles(x))
        torch.testing.assert_close(weighted(x),plain(x)*torch.tensor([1.5,1.]))
        for weights in ([1,-1,1],[1,2],[1,float('nan'),1]):
            with self.assertRaises(ValueError):ThreeSignalRisk(reference=ref,weights=weights)

    def test_line_routing_stable_at_float_roundoff_tie(self):
        line=FeatureLINe(2,classes=2,route_tolerance=1e-5)
        line.fitted.fill_(True);line.clip.fill_(10);line.weight.copy_(torch.eye(2))
        line.activation_mask.copy_(torch.eye(2))
        line.masked_weight[0].copy_(torch.ones(2,2))
        line.masked_weight[1].copy_(torch.full((2,2),2.))
        x=torch.tensor([[1.,1.+5e-7],[1.+5e-7,1.]])
        energy,_=line(x)
        self.assertLess(float((energy[0]-energy[1]).abs()),2e-6)
        # A meaningful margin still routes to the stronger class.
        changed,_=line(torch.tensor([[1.,1.01]]))
        self.assertGreater(float((changed-energy[0]).abs()),.9)

    def test_reference_distance_and_episode_exclusion(self):
        memory=ProjectionMemory(torch.zeros(2),torch.eye(2),
            torch.tensor([[0.,0.],[3.,4.]]),torch.tensor([0,1]))
        query=torch.tensor([[0.,0.],[3.,4.],[9.,12.]])
        np.testing.assert_allclose(memory(query),[0.,0.,10/2**.5],atol=1e-6)
        np.testing.assert_allclose(memory(query[:1],torch.tensor([0])),[5/2**.5],atol=1e-6)
        # Single-frame inference must agree with batched inference.
        torch.testing.assert_close(memory(query),torch.cat([memory(q[None]) for q in query]))

    def test_exactly_three_and_no_learned_bypass(self):
        head=ThreeSignalRisk(reference=torch.arange(100).float().repeat(3,1)/20-2)
        self.assertEqual(list(head.parameters()),[])
        with self.assertRaises(ValueError):head(torch.zeros(1,7))
        rng=torch.Generator().manual_seed(1)
        base=torch.randn(100,3,generator=rng)
        for j in range(3):
            increased=base.clone();increased[:,j]+=2
            self.assertTrue((head(increased)>=head(base)).all())

    def test_empirical_tails_ties_and_finite_sample_calibration(self):
        head=ThreeSignalRisk(reference=torch.tensor([[0.,1.,2.]]).repeat(3,1))
        expected=-torch.tensor([[1.,.75,.25]]).log10()
        torch.testing.assert_close(head.components(torch.tensor([[-1.,1.,3.]])),expected)
        reference=np.arange(275,dtype='float32')
        self.assertGreater(calibration_tail(262,reference),.05)
        self.assertLess(calibration_tail(262.1,reference),.05)
        self.assertGreater(calibration_tail(1e9,reference),0)

    def test_persistence_prefix_and_large_scores(self):
        cfg=dict(smooth_frames=30,persist_frames=15,minimum_history=15)
        x=np.zeros(180,dtype='float32');x[60:]=4
        mean,score=sustained_trace(x,cfg)
        self.assertEqual(score[15],0);self.assertEqual(score[-1],4)
        np.testing.assert_array_equal(score[:95],sustained_trace(x[:95],cfg)[1])
        spike=np.zeros(180);spike[60]=4
        self.assertLess(sustained_trace(spike,cfg)[1].max(),.14)

    def test_loop_requires_motion_and_history(self):
        t=np.arange(400);s=np.tile(np.sin(2*np.pi*t/15)[:,None],(1,14))
        z=np.tile((1+np.cos(2*np.pi*t/15))[:,None],(1,8))
        loop=time_evidence(s,z,np.ones(8),np.ones(8))[:,0]
        self.assertTrue((loop[:45]==0).all());self.assertGreater(loop[-1],.9)
        stationary=time_evidence(np.zeros_like(s),z,np.ones(8),np.ones(8))[:,0]
        self.assertTrue((stationary==0).all())


if __name__=='__main__':unittest.main()
