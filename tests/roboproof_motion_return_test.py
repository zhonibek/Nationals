from pathlib import Path
import sys
import tempfile
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import advantages
from roboproof.motion_learning.objective import LearningObjective, OBJECTIVE, TOTAL_RETURN_OBJECTIVE, discount_parameters
from roboproof.motion_learning.train import compatible_config, source_identity, train_seed


class MotionReturnTests(unittest.TestCase):
    def test_complete_episode_returns_reach_early_actions_and_never_cross_resets(self):
        gamma, decay = discount_parameters(TOTAL_RETURN_OBJECTIVE)
        self.assertEqual((gamma, decay), (1.0, 1.0))
        values = [1, 0.9, 0.8, 0.7, 0.6]
        advantage, returns = advantages([0, 0, 2, 0, -2], values, [0.9, 0.8, 0, 0.6, 0],
                                       [False, False, True, False, True], [False, False, True, False, True],
                                       gamma=gamma, decay=decay)
        torch.testing.assert_close(returns, torch.tensor([2, 2, 2, -2, -2], dtype=torch.float32))
        torch.testing.assert_close(advantage, returns - torch.tensor(values))
        long_advantage, _ = advantages([0] * 119 + [2], [0] * 120, [0] * 120,
                                      [False] * 119 + [True], [False] * 119 + [True], gamma=gamma, decay=decay)
        self.assertEqual(float(long_advantage[0]), 2)
        self.assertEqual(float(long_advantage[-1]), 2)

    def test_total_return_changes_credit_assignment_not_reward_or_terminal_rules(self):
        task = {"start": {"xIn": 0, "yIn": 0, "headingDeg": 0},
                "goal": {"xIn": 0, "yIn": 24, "headingDeg": 90}}
        previous = LearningObjective(task, OBJECTIVE)
        current = LearningObjective(task, TOTAL_RETURN_OBJECTIVE)
        components = {"progress": 0.1, "time": -0.001, "effort": -0.0005,
                      "contact": 0, "success": 0, "fault": 0}
        for heading, reason in ((1.2, "running"), (0.8, "running"), (0.5, "time_limit")):
            info = {"rewardComponents": components, "headingErrorRadians": heading, "reason": reason}
            self.assertEqual(current.reward(0.0985, info), previous.reward(0.0985, info))
        self.assertTrue(current.terminal(False, True, {"reason": "time_limit"}))
        self.assertFalse(current.terminal(False, False, {"reason": "running"}))
        self.assertEqual(discount_parameters(OBJECTIVE), (0.995, 0.98))
        self.assertEqual(discount_parameters("canonical-v1"), (0.99, 0.95))
        with self.assertRaises(ValueError):
            discount_parameters("unknown")

    def test_real_total_return_training_resumes_exactly_and_rejects_estimator_changes(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 2, "maxSteps": 1000, "rolloutSteps": 16, "seed": 607, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "objective": TOTAL_RETURN_OBJECTIVE,
                  "initialization": "broad-pace-v1", "featureTransform": "reference-frame-v1",
                  "policyRegularization": "public-deadline-residual-v1",
                  "returnEstimator": {"discountFactor": 1.0, "gaeLambda": 1.0,
                                      "completeEpisodesOnly": True, "intrinsicDeadlineTerminal": True}}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 607, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 1:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 607, config, sources, time.monotonic() + 90,
                           on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 607, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(complete["actorWeightsChanged"])
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 607,
                           {**config, "returnEstimator": {**config["returnEstimator"], "gaeLambda": 0.98}},
                           sources, time.monotonic() + 90, resume=True)
        self.assertFalse(compatible_config(config, {**config, "objective": OBJECTIVE}))


if __name__ == "__main__":
    unittest.main()
