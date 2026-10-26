import contextlib
import io
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from roboproof.tactical_learning import train
from roboproof.tactical_learning.model import FEATURE_COUNT, TacticalActorCritic
from roboproof_tactical_shutdown_test import FakeBridgeProcess, FakeModel, fake_episode


class FixedModel:
    """Returns fixed logits and values, so the PPO loss can be checked by hand."""

    def __init__(self, logits, values):
        self.logits, self.values = logits, values

    def __call__(self, features, mask):
        return self.logits, self.values


class CountingOptimizer:
    """Counts steps but never changes parameters: these tests check the update loop, not training."""

    def __init__(self):
        self.steps = 0

    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        self.steps += 1


def transitions(model, count, log_probability_shift=0.0):
    torch.manual_seed(3)  # freshly initialized model inputs only
    rows = []
    for index in range(count):
        features = torch.randn((3, FEATURE_COUNT)).tolist()
        with torch.no_grad():
            logits, values = model(torch.tensor([features]), torch.ones((1, 3), dtype=torch.bool))
        choice = index % 3
        rows.append({"features": features, "choice": choice, "reward": float(index % 4), "durationSeconds": 2.0,
                     "logProbability": float(torch.log_softmax(logits, dim=-1)[0, choice]) + log_probability_shift,
                     "value": float(values[0])})
    return rows


class TacticalPpoTests(unittest.TestCase):
    def test_gae_uses_the_time_based_discount_and_ends_each_episode(self):
        trajectory = [{"features": "a", "choice": 0, "reward": 2.0, "durationSeconds": 1.0, "logProbability": -0.1, "value": 1.0},
                      {"features": "b", "choice": 1, "reward": 3.0, "durationSeconds": 2.0, "logProbability": -0.2, "value": 0.5}]
        last, first = train.gae_rows([trajectory])
        self.assertAlmostEqual(last["advantage"], 2.5)
        self.assertAlmostEqual(last["return"], 3.0)
        expected = (2 + 0.995 * 0.5 - 1) + 0.995 * 0.95 * 2.5
        self.assertAlmostEqual(first["advantage"], expected)
        self.assertAlmostEqual(first["return"], expected + 1.0)
        self.assertEqual((first["oldLogProbability"], first["choice"]), (-0.1, 0))
        # With lambda = 1 the advantage is the actor-critic discounted return minus the value estimate.
        returns = train.discounted_rows([trajectory])
        for row, plain, transition in zip(train.gae_rows([trajectory], gae_lambda=1.0), returns, reversed(trajectory)):
            self.assertAlmostEqual(row["advantage"], plain["return"] - transition["value"])
        self.assertEqual(len(train.gae_rows([trajectory, trajectory])), 4)

    def test_the_surrogate_is_clipped_both_ways(self):
        logits = torch.zeros((2, 2))
        values = torch.tensor([1.0, 2.0])
        chosen = -math.log(2)
        old = torch.tensor([chosen - math.log(1.5), chosen - math.log(0.5)])  # ratios 1.5 and 0.5
        advantages = torch.tensor([1.0, -1.0])
        loss, parts = train.ppo_losses(FixedModel(logits, values), None, None, torch.tensor([0, 1]), old, advantages, values)
        # min(1.5 * 1, 1.2 * 1) = 1.2 and min(0.5 * -1, 0.8 * -1) = -0.8, so the actor loss is -(1.2 - 0.8) / 2.
        self.assertAlmostEqual(float(parts["actorLoss"]), -0.2, places=6)
        self.assertAlmostEqual(float(parts["criticLoss"]), 0.0, places=6)
        self.assertAlmostEqual(float(parts["entropy"]), math.log(2), places=6)
        self.assertAlmostEqual(float(parts["clipFraction"]), 1.0)
        self.assertAlmostEqual(float(loss), -0.2 - train.PPO_ENTROPY_WEIGHT * math.log(2), places=6)

    def test_an_update_runs_all_epochs_over_shuffled_minibatches(self):
        model = TacticalActorCritic(width=16)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer = CountingOptimizer()
        result = train.ppo_update(model, optimizer, [transitions(model, 70)], torch.device("cpu"), torch.Generator().manual_seed(1))
        self.assertEqual((result["samples"], result["epochs"], result["stoppedEarlyForKl"]), (70, train.PPO_EPOCHS, False))
        self.assertEqual(optimizer.steps, train.PPO_EPOCHS * 2)  # 70 rows -> minibatches of 64 and 6
        self.assertAlmostEqual(result["approximateKl"], 0.0, places=5)
        self.assertAlmostEqual(result["clipFraction"], 0.0)
        self.assertEqual(result["algorithm"], "ppo-clip-v1")
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.state_dict().items()))
        empty = train.ppo_update(model, optimizer, [], torch.device("cpu"), torch.Generator())
        self.assertEqual((empty["samples"], empty["epochs"]), (0, 0))

    def test_an_update_stops_early_when_the_policy_moves_too_far(self):
        model = TacticalActorCritic(width=16)
        optimizer = CountingOptimizer()
        result = train.ppo_update(model, optimizer, [transitions(model, 10, log_probability_shift=1.0)], torch.device("cpu"),
                                  torch.Generator().manual_seed(1))
        self.assertEqual((result["epochs"], result["stoppedEarlyForKl"], optimizer.steps), (1, True, 1))
        self.assertAlmostEqual(result["approximateKl"], 1.0, places=5)

    def test_ppo_is_opt_in_and_recorded_in_the_run_checkpoint_and_evaluation(self):
        argv = ["train", "--approved-free-t4", "--seed", "12345", "--episodes", "2", "--max-seconds", "60",
                "--evaluation-episodes", "2"]
        with mock.patch.object(train.sys, "argv", argv):
            self.assertEqual(train.parse_args().algorithm, "actor-critic")
        with mock.patch.object(train.sys, "argv", argv + ["--algorithm", "ppo"]):
            self.assertEqual(train.parse_args().algorithm, "ppo")
        with mock.patch.object(train.sys, "argv", argv + ["--algorithm", "trpo"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train.parse_args()

        fake_torch = types.SimpleNamespace(device=lambda name: name, manual_seed=lambda seed: None, Generator=torch.Generator,
                                           cuda=types.SimpleNamespace(manual_seed_all=lambda seed: None),
                                           optim=types.SimpleNamespace(Adam=lambda parameters, lr: types.SimpleNamespace(state_dict=dict)))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with mock.patch.object(train.sys, "argv", argv + ["--algorithm", "ppo", "--output", str(output)]), \
                    mock.patch.object(train, "validate_runtime", return_value={"device": "cuda:0", "gpu": "fake"}), \
                    mock.patch.object(train, "torch", fake_torch), \
                    mock.patch.object(train, "TacticalActorCritic", FakeModel), \
                    mock.patch.object(train, "run_episode", side_effect=fake_episode), \
                    mock.patch.object(train, "update") as plain_update, \
                    mock.patch.object(train, "ppo_update", return_value={"loss": 0.0, "samples": 2, "algorithm": "ppo-clip-v1"}) as ppo_update, \
                    mock.patch.object(train, "atomic_torch_save") as save, \
                    mock.patch.object(train.shutil, "which", return_value="node"), \
                    mock.patch.object(train.subprocess, "Popen", return_value=FakeBridgeProcess()), \
                    contextlib.redirect_stdout(io.StringIO()):
                train.main()
            run = json.loads((output / "run.json").read_text("utf-8"))
            evaluation = json.loads((output / "evaluation.json").read_text("utf-8"))
        self.assertEqual(plain_update.call_count, 0)
        self.assertEqual(ppo_update.call_count, 1)
        self.assertEqual(run["algorithm"], train.ppo_plan())
        self.assertEqual((run["status"], run["algorithm"]["clipRange"], run["algorithm"]["epochs"]), ("completed", 0.2, 4))
        self.assertEqual(evaluation["algorithm"], "ppo-clip-v1")
        self.assertEqual(save.call_args.args[0]["algorithm"], "ppo-clip-v1")
        self.assertIs(run["learnedImprovementVerified"], False)


if __name__ == "__main__":
    unittest.main()
