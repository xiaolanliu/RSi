import unittest
import numpy as np
from agent_closed_loop.command_logs import CausalCommandEvidence,compare_future_plans


class CommandLogTests(unittest.TestCase):
    def test_uses_previous_sent_command_and_resets_transition(self):
        h=CausalCommandEvidence(.1)
        a=np.ones(14);zero=np.zeros(14)
        self.assertIsNone(h.step(0,a,a,zero,'provider')['lagged_tracking_rms_rad'])
        h.step(.05,a,a,a,'interpolation')
        current=a*3
        out=h.step(.11,current,a*2,a,'provider')
        self.assertEqual(out['lagged_tracking_rms_rad'],0)
        self.assertAlmostEqual(out['clip_ratio'],10)
        self.assertFalse(h.step(.12,a,a,a,'transition')['valid'])
        self.assertIsNone(h.step(.13,a,a,a,'provider')['lagged_tracking_rms_rad'])
        with self.assertRaises(ValueError):h.step(.12,a,a,a,'provider')

    def test_causal_prefix_independent_of_future_commands(self):
        rng=np.random.default_rng(12);cmd=rng.normal(size=(100,14));sent=cmd*.8;measured=cmd*.7
        def run(n):
            h=CausalCommandEvidence(.1)
            return [h.step(i/60,cmd[i],sent[i],measured[i],'provider') for i in range(n)]
        self.assertEqual(run(100)[:45],run(45))

    def test_future_chunk_overlap_discards_stale_prefix(self):
        a=np.tile(np.arange(50)[:,None],(1,14)).astype(float)
        old=dict(absolute_targets=a,observed_sent_action_count=0,apply_sent_action_count=0,received_at_monotonic_ns=1)
        b=np.tile(np.arange(15,65)[:,None],(1,14)).astype(float);b[:12]=999
        new=dict(absolute_targets=b,observed_sent_action_count=15,apply_sent_action_count=27,received_at_monotonic_ns=2)
        out=compare_future_plans(new,old)
        self.assertEqual(out['overlap_steps'],23);self.assertEqual(out['rms_rad'],0)
        b[12:]+=1
        self.assertEqual(compare_future_plans(new,old)['rms_rad'],1)
        with self.assertRaises(ValueError):compare_future_plans(old,new)


if __name__=='__main__':unittest.main()
