import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import MotionActorCritic
from roboproof.motion_learning.td3 import ReplayBuffer, TD3Learner, clipped_target, mean_action
from roboproof.motion_learning.td3_train import configuration, train_seed, source_identity


def populated_replay():
    replay = ReplayBuffer(34, 64)
    replay.append([{"modelInput": [index / 100] * 34, "nextModelInput": [(index + 1) / 100] * 34,
                   "action": [0.1, -0.1, 0.05, -0.05], "reward": index / 20 - 1.5,
                   "terminal": index % 16 == 0, "requiredPace": 0.9} for index in range(64)])
    return replay


class TD3SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_mixture_probability_and_inverse_weight_preserve_uniform_objective_and_gradient(self):
        replay = populated_replay()
        probabilities = replay.sampling_probabilities(0.25)
        expected = torch.full((64,), 0.75 / 64, dtype=torch.float64)
        expected[torch.arange(64) % 16 == 0] += 0.25 / 4
        torch.testing.assert_close(probabilities, expected, rtol=0, atol=1e-15)
        parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        losses = (parameter * torch.arange(64, dtype=torch.float64) - 2).square()
        weighted = (probabilities * (1 / (64 * probabilities)) * losses).sum()
        uniform = losses.mean()
        torch.testing.assert_close(weighted, uniform, rtol=0, atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(weighted, parameter, retain_graph=True)[0],
                                   torch.autograd.grad(uniform, parameter)[0], rtol=0, atol=1e-12)
        sampled = replay.sample(64, 0.25)
        sampled_indices = (sampled["observations"][:, 0] * 100).round().long()
        torch.testing.assert_close(sampled["sampling_probabilities"], probabilities[sampled_indices])
        torch.testing.assert_close(sampled["importance_weights"], (1 / (64 * probabilities[sampled_indices])).float())
        self.assertLessEqual(float(sampled["importance_weights"].max()), 2)
        for terminal in (False, True):
            replay.terminals.fill_(terminal)
            torch.testing.assert_close(replay.sampling_probabilities(0.5), torch.full((64,), 1 / 64, dtype=torch.float64))
        for invalid in (True, -0.1, 0.51, math.nan):
            with self.assertRaises(ValueError):
                replay.sample(32, invalid)
        with self.assertRaises(ValueError):
            ReplayBuffer(34, 64).sampling_probabilities(0.25)

    def test_default_sample_preserves_exact_uniform_rng_and_tensors(self):
        replay = populated_replay()
        torch.manual_seed(106001)
        indices = torch.randint(64, (32,))
        expected_rng = torch.get_rng_state()
        torch.manual_seed(106001)
        actual = replay.sample(32)
        self.assertNotIn("importance_weights", actual)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for name, tensor in actual.items():
            torch.testing.assert_close(tensor, getattr(replay, name)[indices], rtol=0, atol=0)

    def test_critic_weighted_loss_actor_uniform_and_invalid_weight_guards(self):
        torch.manual_seed(106002)
        actor = MotionActorCritic("broad-pace-v1", value_critic=False)
        learner = TD3Learner(actor, policy_delay=1, target_noise_std=0)
        replay = populated_replay()
        critic_batch = replay.sample(64, 0.25)
        actor_batch = replay.sample(64)
        actor_batch["observations"] = actor_batch["observations"] + 0.2
        targets, _ = clipped_target(learner.target_actor, learner.target_critics, critic_batch, 1, 0, 0.1)
        first, second = learner.critics(critic_batch["observations"], critic_batch["actions"])
        expected_critic_loss = (critic_batch["importance_weights"] * ((first - targets).square() + (second - targets).square())).mean()
        with patch.object(learner.actor_optimizer, "step", return_value=None):
            result = learner.update(critic_batch, actor_batch=actor_batch)
        expected_actor_loss = -learner.critics(actor_batch["observations"], mean_action(actor, actor_batch["observations"]))[0].mean()
        self.assertAlmostEqual(result["criticLoss"], float(expected_critic_loss.detach()), places=6)
        self.assertAlmostEqual(result["actorLoss"], float(expected_actor_loss.detach()), places=6)
        self.assertTrue(torch.all(actor.actor.weight.grad.abs().sum(-1) > 0))
        count = learner.gradient_updates
        rng = torch.get_rng_state().clone()
        for replacement in (torch.full((64,), math.nan), torch.zeros(64), torch.full((64,), 2.1), torch.ones(63)):
            with self.assertRaises(ValueError):
                learner.update({**critic_batch, "importance_weights": replacement}, actor_batch=actor_batch)
        with self.assertRaises(ValueError):
            learner.update(critic_batch)
        with self.assertRaises(ValueError):
            learner.update(critic_batch, actor_batch=critic_batch)
        self.assertEqual(learner.gradient_updates, count)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_original_simulator_exact_resume_and_sampling_configuration_binding(self):
        config = configuration(106003, 1, 4, 3000, 16, 1, "deadline-context-v1",
                               "finite-total-return-settle-margin-v4", "mixed-full-reach-v5", 4, 1024, 0.05,
                               1, "public-deadline-residual-v1", 4, 8, 0.25)
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 106003, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 106003, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 106003, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            first = torch.load(directory / "complete/checkpoint.pt", weights_only=True)
            second = torch.load(directory / "resumed/checkpoint.pt", weights_only=True)
            def assert_state_equal(actual, expected):
                if isinstance(actual, torch.Tensor):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                elif isinstance(actual, dict):
                    self.assertEqual(actual.keys(), expected.keys())
                    for key in actual:
                        assert_state_equal(actual[key], expected[key])
                elif isinstance(actual, (list, tuple)):
                    self.assertIs(type(actual), type(expected))
                    self.assertEqual(len(actual), len(expected))
                    for item, matching in zip(actual, expected):
                        assert_state_equal(item, matching)
                else:
                    self.assertEqual(actual, expected)
            for name in ("model", "learner", "worldRng"):
                assert_state_equal(first[name], second[name])
            self.assertEqual(complete["collectionSteps"], resumed["collectionSteps"])
            torch.testing.assert_close(first["torchRng"], second["torchRng"], rtol=0, atol=0)
            for name in first["replay"]:
                if isinstance(first["replay"][name], torch.Tensor):
                    torch.testing.assert_close(first["replay"][name], second["replay"][name], rtol=0, atol=0)
            changed = copy.deepcopy(config)
            changed["criticTerminalFraction"] = 0
            with self.assertRaises(ValueError):
                train_seed(environment, directory / "resumed", 106003, changed, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["updates"], 4)
            self.assertEqual(first["config"]["td3Definition"]["criticSampling"]["method"], "uniform-terminal-mixture-v1")
            actor = json.loads((directory / "complete/policy-0004.json").read_text())
            self.assertEqual(actor["algorithm"], "td3-beta-mean-reference-v1")
            self.assertEqual(len(actor["layers"][2]["weight"]), 8)


if __name__ == "__main__":
    unittest.main()
