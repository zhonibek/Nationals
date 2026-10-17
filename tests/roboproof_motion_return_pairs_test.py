import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from roboproof.motion_learning.return_pairs import MeasuredReturnPairs
from roboproof.motion_learning.td3 import TwinQ


def row(index=0, difference=0.5):
    return {"modelInput": [float(index)] + [0.0] * 144, "baselineAction": [0.0] * 4,
            "alternativeAction": [0.5, 0.0, 0.0, 0.0], "returnDifference": difference}


class LinearCritics(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scales", torch.ones(145))
        self.first_weight = nn.Parameter(torch.tensor([0.5, 0.0, 0.0, 0.0]))
        self.second_weight = nn.Parameter(torch.tensor([-0.5, 0.0, 0.0, 0.0]))
        self.bias = nn.Parameter(torch.tensor([4.0, -8.0]))
        self.feature_weight = nn.Parameter(torch.tensor([0.25, 0.5]))
        self.seen = None

    def forward(self, features, actions):
        self.seen = features.detach().clone()
        return (actions @ self.first_weight + self.bias[0] + features[:, 0] * self.feature_weight[0],
                actions @ self.second_weight + self.bias[1] + features[:, 0] * self.feature_weight[1])


class MeasuredReturnPairsTests(unittest.TestCase):
    def test_exact_absolute_weighted_pair_loss_and_both_action_gradients(self):
        values = [row(0, 0.5), row(1, -0.5), row(2, 0.25)]
        weights = torch.tensor([0.5, 2.0, 1.0])
        pairs = MeasuredReturnPairs(values, row_weights=weights)
        critics = LinearCritics()
        targets = torch.tensor([0.5, -0.5, 0.25])
        expected = (((0.25 - targets).square() + (-0.25 - targets).square()) * weights).mean()
        with patch("torch.randperm", return_value=torch.arange(3)):
            loss = pairs.loss(critics)
        self.assertTrue(torch.allclose(loss, expected, rtol=1e-6, atol=1e-7))
        loss.backward()
        self.assertTrue(torch.allclose(critics.first_weight.grad[0], (2 * (0.25 - targets) * 0.5 * weights).mean()))
        self.assertTrue(torch.allclose(critics.second_weight.grad[0], (2 * (-0.25 - targets) * 0.5 * weights).mean()))
        self.assertTrue(torch.equal(critics.first_weight.grad[1:], torch.zeros(3)))
        self.assertTrue(torch.equal(critics.bias.grad, torch.zeros(2)))
        self.assertTrue(torch.equal(critics.feature_weight.grad, torch.zeros(2)))
        self.assertNotEqual(float(loss.detach()), float(expected / weights.mean()))

    def test_common_value_bias_cancels_and_inputs_are_not_mutated(self):
        values = [row(0), row(1)]
        original = copy.deepcopy(values)
        pairs = MeasuredReturnPairs(values)
        critics = LinearCritics()
        first = pairs.loss(critics)
        with torch.no_grad():
            critics.bias.add_(torch.tensor([1000.0, -1000.0]))
        self.assertTrue(torch.equal(first, pairs.loss(critics)))
        self.assertEqual(values, original)

    def test_uniform_bounded_draw_uses_later_rows_and_replays_rng_exactly(self):
        pairs = MeasuredReturnPairs([row(index) for index in range(80)])
        critics = LinearCritics()
        with patch("torch.randperm", return_value=torch.arange(80).flip(0)) as sample:
            pairs.loss(critics)
        sample.assert_called_once_with(80, device="cpu")
        self.assertEqual(critics.seen[:, 0].tolist(), list(range(79, 47, -1)))
        torch.manual_seed(890121)
        before = torch.get_rng_state()
        first = pairs.loss(critics)
        features = critics.seen.clone()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        self.assertTrue(torch.equal(first, pairs.loss(critics)))
        self.assertTrue(torch.equal(features, critics.seen))
        self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_invalid_rows_weights_and_float32_overflow_reject_before_sampling(self):
        malformed = [[], [row()] * 4097, [{**row(), "modelInput": [0.0] * 34}],
            [{**row(), "extra": True}], [{**row(), "returnDifference": True}],
            [{**row(), "returnDifference": float("nan")}], [{**row(), "returnDifference": 1000001}],
            [{**row(), "baselineAction": [0.0, 0.0, 0.0, 2.0]}],
            [{**row(), "alternativeAction": [0.5, 0.5, 0.0, 0.0]}],
            [{**row(), "alternativeAction": [0.0] * 4}],
            [{**row(), "modelInput": [1e100] + [0.0] * 144}]]
        with patch("torch.randperm") as sample:
            for values in malformed:
                with self.subTest(values=values[:1]), self.assertRaises(ValueError):
                    MeasuredReturnPairs(values)
            for weights in ([], [0.0], [-1.0], [True], [float("inf")], [4097.0], [1e-100], torch.tensor([True])):
                with self.subTest(weights=weights), self.assertRaises(ValueError):
                    MeasuredReturnPairs([row()], row_weights=weights)
            with self.assertRaises(ValueError):
                MeasuredReturnPairs([row()], feature_size=True)
        sample.assert_not_called()

    def test_zero_effect_measurement_is_allowed_only_with_zero_target(self):
        values = [{**row(), "alternativeAction": [0.0] * 4, "returnDifference": 0.0}]
        loss = MeasuredReturnPairs(values).loss(LinearCritics())
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()

    def test_actual_twin_critic_receives_gradients_and_rejects_bad_prediction_shapes(self):
        pairs = MeasuredReturnPairs([row()])
        critics = TwinQ(torch.ones(145), 32)
        loss = pairs.loss(critics)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in critics.first.parameters()), 0)
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in critics.second.parameters()), 0)
        invalid = LinearCritics()
        for output in ((torch.tensor([float("nan")]), torch.zeros(1)), (torch.zeros((1, 1)), torch.zeros(1)), (torch.zeros(1),)):
            with patch.object(invalid, "forward", return_value=output), self.assertRaises(ValueError):
                pairs.loss(invalid)
        invalid.scales.zero_()
        with patch("torch.randperm") as sample, self.assertRaises(ValueError):
            pairs.loss(invalid)
        sample.assert_not_called()


if __name__ == "__main__":
    unittest.main()
