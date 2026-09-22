import unittest
import numpy as np
from agent_closed_loop.action_units import UnitConfig,unit_trace,packed_state,PACKED_TO_ARMS,resample_path,repetition_strength
from agent_closed_loop.command_dynamics import CommandDynamics


class PhysicalUnitTests(unittest.TestCase):
    def test_scalar_tail_ties_use_reference_precision(self):
        ref=dict(periodic=np.zeros(10,dtype='float32'),units=np.array([0.,.1,.1,.1,.2],dtype='float32'))
        x=float(np.float32(.1))+1e-10
        expected=repetition_strength(np.array([0.],dtype='float32'),np.array([x],dtype='float32'),ref)[0]
        self.assertEqual(float(repetition_strength(0.,x,ref)),float(expected))
        self.assertAlmostEqual(float(expected),-np.log10(5/6),places=6)

    def test_layout_and_arc_length(self):
        packed=np.arange(14,dtype='float32');arm=packed[PACKED_TO_ARMS]
        np.testing.assert_array_equal(packed_state(arm,'left7_right7'),packed)
        a=np.linspace(0,1,40);b=np.linspace(0,1,80)**2
        np.testing.assert_allclose(resample_path(np.c_[a,2*a],24),resample_path(np.c_[b,2*b],24),atol=1e-10)

    def test_variable_tempo_one_arm_cycles_and_no_future(self):
        # Non-grid periods and varying execution speed; the other arm is idle.
        phase=np.concatenate([np.arange(n)/n for n in [24,27,23,26,25,28,24,23,27,25,26,24]])
        state=np.zeros((len(phase),14));state[:,0]=np.sin(2*np.pi*phase);state[:,12]=.5+.5*np.cos(2*np.pi*phase)
        visual=np.zeros((len(phase),48));cfg=UnitConfig(close=(.25,.25),open=(.65,.65))
        score,events=unit_trace(state,visual,cfg)
        self.assertGreater(score.max(),7);self.assertTrue(all(e['arm']==0 for e in events))
        prefix=unit_trace(state[:173],visual[:173],cfg)[0];np.testing.assert_array_equal(score[:173],prefix)
        still=np.repeat(state[-1:],200,axis=0);extended=np.r_[state,still]
        stopped=unit_trace(extended,np.zeros((len(extended),48)),cfg)[0]
        self.assertLess(stopped[-1],score[-1]*.02)
        no_motion=unit_trace(np.zeros_like(state),visual,cfg)[0];self.assertTrue((no_motion==0).all())

    def test_task_effect_suppresses_repeated_body(self):
        t=np.arange(450);state=np.zeros((len(t),14));state[:,0]=np.sin(2*np.pi*t/25);state[:,12]=.5+.5*np.cos(2*np.pi*t/25)
        config=UnitConfig(close=(.25,.25),open=(.65,.65))
        unchanged=unit_trace(state,np.zeros((len(t),48)),config)[0]
        progressing=unit_trace(state,np.tile(t[:,None]/25,(1,48)),config)[0]
        self.assertLess(progressing.max(),unchanged.max()/3)

    def test_dense_retries_vs_same_number_of_spaced_actions(self):
        cfg=UnitConfig(close=(.25,.25),open=(.65,.65),recency_frames=90)
        def cycles(period):
            t=np.arange(period*10);s=np.zeros((len(t),14))
            s[:,0]=np.sin(2*np.pi*t/period);s[:,12]=.5+.5*np.cos(2*np.pi*t/period)
            return s
        fast=cycles(24);slow=cycles(120)
        fast_score,fast_events=unit_trace(fast,np.zeros((len(fast),48)),cfg)
        slow_score,slow_events=unit_trace(slow,np.zeros((len(slow),48)),cfg)
        self.assertEqual(len(fast_events),len(slow_events))
        self.assertGreater(fast_events[-1]['score'],slow_events[-1]['score']*3)
        # Future frames cannot affect an already emitted score.
        np.testing.assert_array_equal(fast_score[:173],unit_trace(fast[:173],np.zeros((173,48)),cfg)[0])
        still=np.repeat(fast[-1:],270,axis=0)
        stopped=unit_trace(np.r_[fast,still],np.zeros((len(fast)+270,48)),cfg)[0]
        self.assertLess(stopped[-1],fast_score[-1]*.06)
        # Successful visible change still suppresses dense repeated motions.
        changing=np.tile(np.arange(len(fast))[:,None]/24,(1,48))
        progressing=unit_trace(fast,changing,cfg)[0]
        self.assertLess(progressing.max(),fast_score.max()/2)

    def test_dynamics_rejects_duplicate_commands_and_detects_tracking_failure(self):
        rng=np.random.default_rng(13);n=1500;command=rng.normal(0,.4,(n,14)).astype('float32');state=np.zeros_like(command)
        for t in range(n-1):state[t+1]=state[t]+.15*(command[t]-state[t])
        stamps=np.arange(n)/30;ids=np.zeros(n)
        with self.assertRaises(ValueError):CommandDynamics().fit(state,state,stamps,ids,layout='joints12_grippers2')
        model=CommandDynamics().fit(state[:1000],command[:1000],stamps[:1000],ids[:1000],layout='joints12_grippers2')
        t=1200;velocity=(state[t,:12]-state[t-1,:12])*30
        good=model.residual(state[t],velocity,command[t],state[t+1],1/30,layout='joints12_grippers2')
        bad=model.residual(state[t],velocity,command[t],state[t],1/30,layout='joints12_grippers2')
        self.assertLess(good,.05);self.assertGreater(bad,100)


if __name__=='__main__':unittest.main()
