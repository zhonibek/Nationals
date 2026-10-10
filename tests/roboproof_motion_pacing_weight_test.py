import copy
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
from roboproof.motion_learning.model import MotionActorCritic, DEADLINE_CONTEXT, update
from roboproof.motion_learning.train import compatible_config, source_identity, train_seed, run


class MotionPacingWeightTests(unittest.TestCase):
    def test_weight_scales_only_training_shortfall_and_preserves_default_loss(self):
        torch.set_num_threads(1)
        torch.manual_seed(2801)
        model = MotionActorCritic("broad-pace-v1")
        observation = [0.0] * 34
        action, unit_action, log_prob, value = model.act(observation, deterministic=True)
        rows = [{"observation": observation, "action": unit_action, "logProb": float(log_prob),
                 "value": float(value), "nextValue": 0, "reward": 0, "requiredPace": 1,
                 "terminated": True, "ended": True} for _ in range(64)]
        results = []
        for weight in (1, 100):
            candidate = copy.deepcopy(model)
            optimizer = torch.optim.Adam(candidate.parameters(), lr=0.001)
            torch.manual_seed(2802)
            result = update(candidate, optimizer, rows, epochs=1, policy_regularization="public-deadline-residual-v1",
                            deadline_pace_loss_weight=weight)
            self.assertFalse(torch.equal(candidate.actor.weight, model.actor.weight))
            self.assertEqual(result["deadlinePaceLossWeight"], weight)
            self.assertAlmostEqual(result["meanWeightedDeadlinePaceLoss"], weight * result["meanDeadlinePaceLoss"])
            results.append(result)
        self.assertAlmostEqual(results[0]["meanDeadlinePaceLoss"], results[1]["meanDeadlinePaceLoss"])
        self.assertAlmostEqual(results[1]["meanLoss"] - results[0]["meanLoss"],
                               99 * results[0]["meanDeadlinePaceLoss"], places=5)
        self.assertGreater(results[1]["maximumActorGradientNorm"], results[0]["maximumActorGradientNorm"] * 50)
        default = copy.deepcopy(model)
        explicit = copy.deepcopy(model)
        torch.manual_seed(2802)
        omitted = update(default, torch.optim.Adam(default.parameters()), rows, epochs=1, policy_regularization="public-deadline-residual-v1")
        torch.manual_seed(2802)
        declared = update(explicit, torch.optim.Adam(explicit.parameters()), rows, epochs=1,
                          policy_regularization="public-deadline-residual-v1", deadline_pace_loss_weight=1)
        self.assertEqual(omitted, declared)
        self.assertTrue(all(torch.equal(default.state_dict()[key], tensor) for key, tensor in explicit.state_dict().items()))

    def test_invalid_or_inactive_weights_fail_before_optimization_or_collection(self):
        model = MotionActorCritic("broad-pace-v1")
        optimizer = torch.optim.Adam(model.parameters())
        for invalid in (True, -1, 101, float("nan"), float("inf"), "100"):
            with self.assertRaises(ValueError):
                update(model, optimizer, [], policy_regularization="public-deadline-residual-v1", deadline_pace_loss_weight=invalid)
            with self.assertRaises(ValueError):
                run(policy_regularization="public-deadline-residual-v1", deadline_pace_loss_weight=invalid)
        with self.assertRaises(ValueError):
            update(model, optimizer, [], deadline_pace_loss_weight=100)
        with self.assertRaises(ValueError):
            run(deadline_pace_loss_weight=100)

    def test_weighted_real_ppo_and_imitation_resume_exactly_and_bind_weight(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 3, "maxSteps": 2000, "rolloutSteps": 16, "seed": 2803, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "deadline-context-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
                  "policyRegularization": "public-deadline-residual-v1", "deadlinePaceLossWeight": 100,
                  "warmStartUpdates": 1, "demonstrationAnchorWeight": 10, "teacherSelection": "pose-deadline-margin-v2",
                  "objective": "finite-total-return-pose-effort-v3"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 2803, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 2803, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 2803, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(complete["actorWeightsChanged"])
            for history in complete["history"][1:]:
                self.assertEqual(history["deadlinePaceLossWeight"], 100)
                self.assertTrue(math.isfinite(history["meanWeightedDeadlinePaceLoss"]))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 2803, {**config, "deadlinePaceLossWeight": 1},
                           sources, time.monotonic() + 90, resume=True)
        self.assertFalse(compatible_config(config, {**config, "deadlinePaceLossWeight": 1}))
        with tempfile.TemporaryDirectory() as temporary:
            declared = run(directory=temporary, seed=2804, seed_count=1, updates=1, max_steps=500, rollout_steps=16,
                           max_seconds=90, policy_regularization="public-deadline-residual-v1", deadline_pace_loss_weight=100)
        self.assertEqual(declared["status"], "completed")
        self.assertEqual(declared["config"]["regularizationDefinition"]["paceLossWeight"], 100)


if __name__ == "__main__":
    unittest.main()
