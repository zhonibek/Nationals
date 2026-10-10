import math
from pathlib import Path
import sys
import tempfile
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.imitation import MeasuredAnchor, POSE_BUDGET_SCALES
from roboproof.motion_learning.model import MotionActorCritic, DEADLINE_CONTEXT, update
from roboproof.motion_learning.train import compatible_config, source_identity, train_seed, run


class MotionPrecisionAnchorTests(unittest.TestCase):
    def test_precision_units_weight_residuals_without_downweighting_pace_or_freezing_heads(self):
        torch.set_num_threads(1)
        torch.manual_seed(3901)
        model = MotionActorCritic("broad-pace-v1")
        anchor = MeasuredAnchor(34)
        anchor.append([{"modelInput": [0] * 34, "teacherAction": [0.5, 0, 0, 0]}] * 32)
        original_bias = model.actor.bias.detach().clone()
        with torch.no_grad():
            model.actor.bias[1] += 2
        uniform = anchor.loss(model)
        precision = anchor.loss(model, "pose-budget-v2")
        self.assertGreater(float(uniform.detach()), 0)
        self.assertAlmostEqual(float((precision / uniform).detach()), POSE_BUDGET_SCALES[1] ** 2, places=3)
        precision.backward()
        self.assertGreater(float(model.actor.bias.grad[1]), 0)
        self.assertLess(float(model.actor.bias.grad[5]), 0)
        self.assertIsNone(model.critic.weight.grad)
        with torch.no_grad():
            model.actor.bias.copy_(original_bias)
            model.actor.bias[0] += 1
        self.assertAlmostEqual(float(anchor.loss(model).detach()), float(anchor.loss(model, "pose-budget-v2").detach()), places=7)
        model.zero_grad()
        with torch.no_grad():
            model.actor.bias[:4] += 1
        anchor.loss(model, "pose-budget-v2").backward()
        self.assertTrue(torch.all(model.actor.bias.grad[:4] > 0))
        self.assertTrue(torch.all(model.actor.bias.grad[4:] < 0))
        self.assertTrue(all(math.isfinite(scale) and scale > 0 for scale in POSE_BUDGET_SCALES))
        self.assertEqual(POSE_BUDGET_SCALES[0], 1)

    def test_unknown_or_inactive_precision_profiles_fail_before_optimizer_and_collection(self):
        model = MotionActorCritic("broad-pace-v1")
        anchor = MeasuredAnchor(34)
        with self.assertRaises(ValueError):
            anchor.loss(model, "unknown")
        with self.assertRaises(ValueError):
            update(model, torch.optim.Adam(model.parameters()), [], anchor_loss_profile="pose-budget-v2")
        with self.assertRaises(ValueError):
            run(demonstration_anchor_loss_profile="pose-budget-v2")
        with self.assertRaises(ValueError):
            run(demonstration_anchor_loss_profile="unknown", demonstration_anchor_weight=10, warm_start_updates=1)

    def test_real_precision_retention_gradients_resume_and_metadata_bind_the_profile(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 3, "maxSteps": 2000, "rolloutSteps": 16, "seed": 3902, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "deadline-context-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
                  "policyRegularization": "public-deadline-residual-v1", "deadlinePaceLossWeight": 100,
                  "warmStartUpdates": 1, "demonstrationAnchorWeight": 10,
                  "demonstrationAnchorLossProfile": "pose-budget-v2", "teacherSelection": "pose-deadline-margin-v2",
                  "objective": "finite-total-return-pose-effort-v3"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 3902, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 3902, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 3902, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(complete["actorWeightsChanged"])
            self.assertTrue(all(history["demonstrationAnchorLossProfile"] == "pose-budget-v2" for history in complete["history"][1:]))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 3902, {**config, "demonstrationAnchorLossProfile": "uniform-action-v1"},
                           sources, time.monotonic() + 90, resume=True)
        self.assertFalse(compatible_config(config, {**config, "demonstrationAnchorLossProfile": "uniform-action-v1"}))
        with tempfile.TemporaryDirectory() as temporary:
            declared = run(directory=temporary, seed=3903, seed_count=1, updates=2, max_steps=1000, rollout_steps=16,
                           max_seconds=90, warm_start_updates=1, demonstration_anchor_weight=10,
                           demonstration_anchor_loss_profile="pose-budget-v2")
        self.assertEqual(declared["status"], "completed")
        self.assertEqual(declared["config"]["demonstrationAnchorDefinition"]["axisScales"], list(POSE_BUDGET_SCALES))
        self.assertEqual(declared["config"]["demonstrationAnchorDefinition"]["precisionDefinition"]["fraction"], 0.2)


if __name__ == "__main__":
    unittest.main()
