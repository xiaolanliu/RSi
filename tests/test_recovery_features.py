"""Mechanism tests for causal behavior evidence, not recoverability accuracy."""

import numpy as np
import torch

from agent_closed_loop.recovery_features import behavior_feature_names, causal_behavior_features


def features(s, a=None, v=None, **kwargs):
    if a is None:
        a = s
    if v is None:
        v = s
    return causal_behavior_features(s, a, v, state_scale=np.ones(s.shape[1]),
                                    action_scale=np.ones(a.shape[1]),
                                    visual_scale=np.ones(v.shape[1]), **kwargs)


def test_prefix_invariance_and_future_changes():
    rng = np.random.default_rng(112)
    s, a, v = (rng.normal(size=(350, dim)) for dim in (14, 14, 48))
    full, names = features(s, a, v)
    assert names == behavior_feature_names() and len(names) == len(set(names)) == 48
    for stop in (1, 16, 31, 60, 61, 119, 241, 349):
        prefix, prefix_names = features(s[:stop], a[:stop], v[:stop])
        np.testing.assert_array_equal(prefix, full[:stop])
        assert prefix_names == names
    changed = [x.copy() for x in (s, a, v)]
    for x in changed:
        x[111:] += 1000 * rng.normal(size=x[111:].shape)
    output, _ = features(*changed)
    np.testing.assert_array_equal(output[:111], full[:111])
    assert full.dtype == np.float32 and np.all((full >= 0) & (full <= 1))


def test_loop_hold_and_progress_counterexamples():
    time = np.arange(361) / 30
    loop = np.stack((np.sin(2 * np.pi * time), np.cos(2 * np.pi * time)), axis=1)
    repeated, names = features(loop)
    index = names.index("repeat_without_progress_1s")
    assert np.mean(repeated[90:, index]) > .9
    held, _ = features(np.full_like(loop, 17.0))
    for j, name in enumerate(names):
        if "repeat" in name:
            assert np.all(held[:, j] == 0), name
    # Equal velocity segments repeat perfectly, but they continually advance.
    moving = np.stack((time, 2 * time), axis=1)
    progressing, _ = features(moving)
    assert progressing[-1, names.index("action_repeat_similarity_1s")] > .9
    assert progressing[-1, index] < 1e-6
    # A periodic arm trajectory can still make visual progress (e.g. sweeping).
    visual_progress, _ = features(loop, v=moving)
    assert np.mean(visual_progress[90:, index]) < 1e-6
    # No full adjacent blocks exist before two seconds at 30 FPS.
    assert np.all(repeated[:60, names.index("action_repeat_similarity_1s")] == 0)
    assert repeated[59, names.index("history_fraction_1s")] < 1
    assert repeated[60, names.index("history_fraction_1s")] == 1


def test_scaling_tracking_empty_and_tensor_inputs():
    rng = np.random.default_rng(90)
    s = rng.normal(size=(270, 3))
    a = s + .2
    v = rng.normal(size=(270, 4))
    base, names = features(s, a, v)
    scaled, _ = causal_behavior_features(torch.tensor(2 * s), torch.tensor(2 * a),
                                          torch.tensor(3 * v), state_scale=np.ones(3) * 2,
                                          action_scale=np.ones(3) * 2, visual_scale=np.ones(4) * 3)
    np.testing.assert_allclose(base, scaled, atol=1e-7)
    np.testing.assert_allclose(base[:, names.index("tracking_pose_error")], .2 / 1.2)
    empty, _ = features(np.empty((0, 3)))
    assert empty.shape == (0, 48)
    one, _ = features(np.ones((1, 3)))
    assert np.isfinite(one).all()
    for bad in (np.full((2, 3), np.nan), np.ones((2,))):
        try:
            causal_behavior_features(bad, a[:2], v[:2], state_scale=np.ones(3),
                                     action_scale=np.ones(3), visual_scale=np.ones(4))
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was accepted")


if __name__ == "__main__":
    for test in (test_prefix_invariance_and_future_changes,
                 test_loop_hold_and_progress_counterexamples,
                 test_scaling_tracking_empty_and_tensor_inputs):
        test()
        print(test.__name__ + ": PASS", flush=True)
