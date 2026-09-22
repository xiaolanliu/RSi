"""Regression checks for the final sustained-evidence training objective."""
from dataclasses import replace
import unittest
import torch
from agent_closed_loop.change_subtask import ChangeConfig
from agent_closed_loop.train_change_memory import memory_objective


class MemoryTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def example(self, centers, length=1300):
        t = torch.arange(length, dtype=torch.float32)
        cdf = ((t[:, None] - torch.tensor(centers)[None]) / 3).sigmoid()
        teacher = torch.cat((1-cdf[:, :1], cdf[:, :-1]-cdf[:, 1:], cdf[:, -1:]), -1)[None]
        logits = torch.logit(cdf.clamp(1e-6, 1-1e-6))[None].requires_grad_()
        valid = torch.ones(1, length, dtype=torch.bool)
        return logits, teacher, valid

    def test_unequal_completed_lengths_and_weak_gradient(self):
        config = ChangeConfig()
        logits, teacher, valid = self.example([120, 300, 580, 980])
        loss, metrics = memory_objective({'logits': logits}, teacher, valid, config)
        self.assertEqual(metrics['duration_smooth_l1'], 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

        logits, teacher, valid = self.example([20, 40, 60, 80])
        regularized, metrics = memory_objective({'logits': logits}, teacher, valid, config)
        baseline, _ = memory_objective({'logits': logits}, teacher, valid, replace(config, duration_weight=0))
        self.assertGreater(metrics['weighted_duration'], 0.)
        self.assertAlmostEqual(float((regularized-baseline).detach()), metrics['weighted_duration'], places=6)
        gradient = torch.autograd.grad(regularized-baseline, logits)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_padding_and_unobserved_terminal_stages(self):
        config = ChangeConfig()
        logits, teacher, valid = self.example([20, 40, 60, 80])
        loss, metrics = memory_objective({'logits': logits}, teacher, valid, config)
        # Invalid batch padding must not become duration or boundary evidence.
        padded_logits = torch.cat((logits, torch.full((1, 400, 4), 20.)), 1)
        padded_teacher = torch.cat((teacher, torch.zeros(1, 400, 5)), 1)
        padded_valid = torch.cat((valid, torch.zeros(1, 400, dtype=torch.bool)), 1)
        padded_loss, padded_metrics = memory_objective({'logits': padded_logits}, padded_teacher, padded_valid, config)
        torch.testing.assert_close(loss, padded_loss)
        self.assertEqual(metrics['duration_smooth_l1'], padded_metrics['duration_smooth_l1'])
        # An unfinished recording in P1 has no observed completed duration.
        unfinished = torch.zeros_like(teacher)
        unfinished[..., 0] = 1
        _, censored = memory_objective({'logits': logits}, unfinished, valid, config)
        self.assertEqual(censored['duration_smooth_l1'], 0.)


if __name__ == '__main__':
    unittest.main()
