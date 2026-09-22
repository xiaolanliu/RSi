"""Small exact LINe mechanism checks, not claims about robot recovery."""

import io

import torch

from agent_closed_loop.line_probe import PhaseProbe


def raises(fn, error=ValueError):
    try:
        fn()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def toy():
    probe = PhaseProbe(3, 3, 2)
    with torch.no_grad():
        probe.hidden.weight.copy_(torch.eye(3))
        probe.hidden.bias.zero_()
        probe.classifier.weight.copy_(torch.tensor([[1., 2., -1.], [-2., 1., 3.]]))
        probe.classifier.bias.zero_()
    return probe


def test_exact_importance_activation_and_signed_weight_pruning():
    probe = toy()
    x = torch.tensor([[1., 2., 0.], [3., 0., 2.], [0., 4., 2.], [2., 2., 4.]])
    labels = torch.tensor([0, 0, 1, 1])
    raw_before = probe(x).detach().clone()
    meta = probe.fit_line(x, labels, correct_only=False, clip_quantile=1., neuron_keep=2/3, weight_keep=.5)
    expected_c = torch.tensor([[2., 2., 1.], [2., 3., 9.]])
    torch.testing.assert_close(probe.line_importance, expected_c, rtol=0, atol=0)
    # Equal importance of units 0/1 is resolved by their stable index order.
    assert probe.line_activation_mask.tolist() == [[True, True, False], [False, True, True]]
    expected_w0 = torch.tensor([[True, True, False], [False, False, True]])
    expected_w1 = torch.tensor([[False, True, False], [False, True, True]])
    assert torch.equal(probe.line_weight_mask[0], expected_w0)
    assert torch.equal(probe.line_weight_mask[1], expected_w1)
    # Magnitude ranking would wrongly retain negative weights; signed LINe
    # ranking excludes them here.
    assert not probe.line_weight_mask[:, 0, 2].any()
    assert not probe.line_weight_mask[:, 1, 0].any()
    assert meta["class_counts"] == [2, 2]
    scored = probe.line_forward(x)
    expected_logits = []
    for xi, route in zip(x, probe(x).argmax(-1)):
        h = xi * probe.line_activation_mask[route]
        expected_logits.append((probe.line_masked_weight[route] * h).sum(-1))
    torch.testing.assert_close(scored["line_logits"], torch.stack(expected_logits), rtol=0, atol=0)
    torch.testing.assert_close(scored["energy"], -torch.logsumexp(torch.stack(expected_logits), -1))
    torch.testing.assert_close(probe(x), raw_before, rtol=0, atol=0)


def test_clipped_routing_and_strict_serialization():
    probe = PhaseProbe(2, 2, 2)
    with torch.no_grad():
        probe.hidden.weight.copy_(torch.eye(2)); probe.hidden.bias.zero_()
        probe.classifier.weight.copy_(torch.tensor([[2., 0.], [0., 3.]])); probe.classifier.bias.zero_()
    probe.fit_line(torch.tensor([[1., 1.], [1., 1.]]), torch.tensor([0, 1]), correct_only=False,
                   clip_quantile=1., neuron_keep=1., weight_keep=1.)
    x = torch.tensor([[[10., 1.], [1., 1.]], [[1., 10.], [0., 0.]]])
    result = probe.line_forward(x)
    assert result["predicted_class"][0, 0] == 0
    assert result["routing_class"][0, 0] == 1
    assert result["energy"].shape == (2, 2)
    stream = io.BytesIO()
    torch.save(probe.state_dict(), stream); stream.seek(0)
    restored = PhaseProbe(2, 2, 2)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    for key, value in result.items():
        torch.testing.assert_close(restored.line_forward(x)[key], value, rtol=0, atol=0)
    # A pointwise probe is causal under prefix/extension and batched evaluation.
    torch.testing.assert_close(probe.line_forward(x[0, :1])["energy"], result["energy"][0, :1])


def test_correct_only_empty_classes_and_deterministic_zero_importance():
    probe = toy()
    x = torch.tensor([[1., 0., 0.], [2., 0., 0.], [0., 0., 3.]])
    labels = torch.tensor([0, 0, 0])  # The last sample is deliberately wrong.
    meta = probe.fit_line(x, labels, clip_quantile=1., neuron_keep=1/3, weight_keep=.5)
    assert meta["class_counts"] == [2, 0] and meta["fallback_classes"] == [1]
    torch.testing.assert_close(probe.line_importance, torch.tensor([[1.5, 0., 0.], [3., 0., 0.]]))
    zero = toy()
    zeros = torch.zeros(3, 3)
    zero.fit_line(zeros, torch.zeros(3, dtype=torch.long), neuron_keep=1/3, weight_keep=1/3)
    assert zero.line_clip == 0
    assert zero.line_activation_mask.tolist() == [[True, False, False], [True, False, False]]
    assert torch.isfinite(zero.line_forward(zeros)["energy"]).all()


def test_training_only_fitting_and_inference_has_no_fit_side_effects():
    probe = toy()
    x = torch.eye(3)
    labels = torch.tensor([0, 0, 1])
    raises(lambda: probe.line_forward(x), RuntimeError)
    for split in ("validation", "calibration", "ood_train", "test"):
        raises(lambda: probe.fit_line(x, labels, fit_split=split))
    raises(lambda: probe.fit_line(x, labels.float()))
    raises(lambda: probe.fit_line(x, labels, neuron_keep=0))
    raises(lambda: probe.fit_line(x * float("nan"), labels))
    probe.fit_line(x, labels)
    before = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.line_forward(torch.full((7, 3), 1000.))
    for key, value in before.items():
        torch.testing.assert_close(probe.state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    for test in (
        test_exact_importance_activation_and_signed_weight_pruning,
        test_clipped_routing_and_strict_serialization,
        test_correct_only_empty_classes_and_deterministic_zero_importance,
        test_training_only_fitting_and_inference_has_no_fit_side_effects,
    ):
        test()
        print(test.__name__ + ": PASS", flush=True)
