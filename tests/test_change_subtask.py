import unittest
import numpy as np
import torch
from agent_closed_loop.change_subtask import ChangeConfig,LatentChangeHead,FreshStageMemory,duration_regularizer


class ChangeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_stream_chunks_future_and_stationary_contrasts(self):
        torch.manual_seed(7);model=LatentChangeHead(ChangeConfig(dimension=16,hidden=16)).eval();z=torch.randn(1,340,16)
        with torch.inference_mode():
            whole,_=model(z);first,c=model(z[:,:113]);second,_=model(z[:,113:],c)
            cache=None;steps=[]
            for t in range(z.shape[1]):o,cache=model.step(z[:,t],cache);steps.append(o['logits'])
            torch.testing.assert_close(torch.stack(steps,1),whole['logits'],atol=3e-6,rtol=3e-5)
            for k in whole:torch.testing.assert_close(torch.cat((first[k],second[k]),1),whole[k],atol=3e-6,rtol=3e-5)
            changed=z.clone();changed[:,113:]+=5;out,_=model(changed)
            for k in whole:torch.testing.assert_close(out[k][:,:113],whole[k][:,:113])
            stable,_=model(torch.full((1,1000,16),2.))
            assert stable['change_sizes'].max()<1e-5

    def test_no_clock_transition_and_no_reusing_high_future_scores(self):
        cfg=ChangeConfig(dimension=4,confirmation_frames=3,evidence_frames=2,rearm_frames=2)
        m=FreshStageMemory(cfg)
        for t in range(30000):
            out=m.step([.99]*4,[0,0,0],np.ones(4))
            self.assertEqual(out['confirmed_phase'],1)
        # Establish P1, then one genuine fresh boundary event.
        m.reset()
        for _ in range(5):m.step([.05,.99,.99,.99],[0,0,0],np.zeros(4))
        for t in range(30):out=m.step([.99]*4,[.5]*3,np.ones(4)*(1+t*.1))
        self.assertEqual(out['confirmed_phase'],2)  # persistent future highs cannot chain transitions
        for t in range(4):m.step([.99,.05,.99,.99],[.5]*3,np.ones(4)*(4+t*.1))
        for t in range(6):out=m.step([.99]*4,[.5]*3,np.ones(4)*(6+t*.1))
        self.assertEqual(out['confirmed_phase'],3)

    def test_soft_uncertainty_can_recede_and_alarm_clears_confirmation(self):
        cfg=ChangeConfig(dimension=4,evidence_frames=1,confirmation_frames=3,rearm_frames=1)
        m=FreshStageMemory(cfg);m.step([0]*4,[0]*3,np.zeros(4))
        a=m.step([.8]*4,[.5]*3,np.ones(4));b=m.step([.2]*4,[.5]*3,np.ones(4))
        self.assertLess(b['phase_probs'][1],a['phase_probs'][1]);self.assertEqual(b['confirmed_phase'],1)
        for _ in range(10):o=m.step([.99]*4,[.5]*3,np.ones(4),alarm=True);self.assertEqual(o['accepted_phase'],0)
        for _ in range(2):o=m.step([.99]*4,[.5]*3,np.ones(4));self.assertEqual(o['confirmed_phase'],1)
        o=m.step([.99]*4,[.5]*3,np.ones(4));self.assertEqual(o['confirmed_phase'],2)
        np.testing.assert_allclose(o['phase_probs'].sum(),1)

    def test_duration_band_not_equal_allocation_and_gradients(self):
        cfg=ChangeConfig();valid=torch.ones(1,1300,dtype=torch.bool)
        def events(centers):
            t=torch.arange(1300.)[None,:,None];p=torch.exp(-.5*((t-torch.tensor(centers)[None,None])/3)**2).clamp(1e-9,1-1e-5)
            return torch.logit(p)
        # Unequal segment lengths 120, 180, 280, 400 are all unpenalized.
        logits=events([120,300,580,980]).requires_grad_();loss,d=duration_regularizer(logits,valid,cfg)
        self.assertLess(float(loss),1e-6)
        short=events([15,30,45,60]).requires_grad_();penalty,_=duration_regularizer(short,valid,cfg)
        self.assertGreater(float(penalty),0);penalty.backward();self.assertTrue(torch.isfinite(short.grad).all());self.assertGreater(float(short.grad.abs().sum()),0)
        for frames in (100,200,400):
            config=ChangeConfig(nominal_frames=frames);m=FreshStageMemory(config)
            for t in range(600):out=m.step([.99]*4,[0]*3,np.ones(128))
            self.assertEqual(out['confirmed_phase'],1)


if __name__=='__main__':unittest.main()
