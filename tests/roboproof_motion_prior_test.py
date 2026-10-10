import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest

import torch
from torch.distributions import Beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import MotionActorCritic, control_prior_loss, public_deadline_floor, update
from roboproof.motion_learning.train import compatible_config, source_identity, train_seed


class MotionPriorTests(unittest.TestCase):
    def test_public_deadline_floor_uses_only_declared_timing(self):
        observation = [0.0] * 34
        observation[18] = 10
        observation[25] = 1
        self.assertEqual(public_deadline_floor(5, observation), 0.5)
        observation[18] = 3
        self.assertEqual(public_deadline_floor(5, observation), 1.0)
        observation[25] = 5
        self.assertEqual(public_deadline_floor(5, observation), 0.25)
        observation[18] = 0
        self.assertEqual(public_deadline_floor(5, observation), 0.25)
        self.assertEqual(public_deadline_floor(0, observation), 0.25)
        with self.assertRaises(ValueError):
            public_deadline_floor(math.nan, observation)
        with self.assertRaises(ValueError):
            public_deadline_floor(5, observation[:33])

    def test_prior_has_zero_residual_cost_at_center_and_no_pace_penalty_above_floor(self):
        alpha = torch.tensor([[4.5, 60, 60, 60]], requires_grad=True)
        beta = torch.tensor([[1.5, 60, 60, 60]], requires_grad=True)
        pace, residual = control_prior_loss(Beta(alpha, beta), torch.tensor([0.5]))
        self.assertEqual(float(pace.detach()), 0)
        self.assertEqual(float(residual.detach()), 0)
        pace, residual = control_prior_loss(Beta(alpha, beta), torch.tensor([1.0]))
        pace.backward()
        self.assertLess(float(alpha.grad[0, 0]), 0)
        self.assertGreater(float(beta.grad[0, 0]), 0)
        self.assertTrue(torch.all(alpha.grad[0, 1:] == 0))
        self.assertTrue(torch.all(beta.grad[0, 1:] == 0))

    def test_residual_prior_gradient_recenters_without_constraining_pace(self):
        alpha = torch.tensor([[4.5, 70, 60, 60]], requires_grad=True)
        beta = torch.tensor([[1.5, 50, 60, 60]], requires_grad=True)
        _, residual = control_prior_loss(Beta(alpha, beta), torch.tensor([0.5]))
        residual.backward()
        self.assertGreater(float(residual.detach()), 0)
        self.assertEqual(float(alpha.grad[0, 0]), 0)
        self.assertEqual(float(beta.grad[0, 0]), 0)
        self.assertGreater(float(alpha.grad[0, 1]), 0)
        self.assertLess(float(beta.grad[0, 1]), 0)

    def test_invalid_prior_targets_fail_before_an_optimizer_step(self):
        model = MotionActorCritic("broad-pace-v1")
        distribution, _ = model(torch.zeros((2, 34)))
        for target in [torch.tensor([math.nan, 1]), torch.tensor([0.24, 1]),
                       torch.tensor([1.01, 1]), torch.tensor([[1, 1]])]:
            with self.assertRaises(ValueError):
                control_prior_loss(distribution, target)
        with self.assertRaises(ValueError):
            update(model, torch.optim.Adam(model.parameters()), [], policy_regularization="unknown")

    def test_regularized_real_gradients_resume_exactly_without_changing_actor_input(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 2, "maxSteps": 1000, "rolloutSteps": 16, "seed": 302, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "objective": "finite-deadline-pose-effort-v2", "policyRegularization": "public-deadline-residual-v1"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            observation, info = environment.reset(seed=302)
            self.assertEqual(len(observation), 34)
            self.assertGreater(info["referenceDurationSeconds"], 0)
            complete = train_seed(environment, directory / "complete", 302, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 1:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 302, config, sources, time.monotonic() + 90,
                           on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 302, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(complete["actorWeightsChanged"])
            for row in complete["history"]:
                self.assertTrue(math.isfinite(row["meanDeadlinePaceLoss"]))
                self.assertTrue(math.isfinite(row["meanResidualPriorKL"]))
            actor = json.loads((directory / "complete" / complete["policyFile"]).read_text())
            self.assertEqual(actor["observationSize"], 34)
            self.assertEqual(len(actor["layers"][0]["weight"][0]), 34)
            self.assertNotIn("requiredPace", actor)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 302, {**config, "policyRegularization": "none"},
                           sources, time.monotonic() + 90, resume=True)
        self.assertFalse(compatible_config(config, {**config, "policyRegularization": "none"}))
        self.assertFalse(compatible_config(config, {**config, "regularizationDefinition": {"paceLossWeight": 2.0}}))


if __name__ == "__main__":
    unittest.main()
