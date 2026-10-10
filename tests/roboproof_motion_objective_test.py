import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from roboproof.motion_learning.model import MotionActorCritic, advantages
from roboproof.motion_learning.objective import LearningObjective


TASK = {"start": {"xIn": 0, "yIn": 0}, "goal": {"xIn": 0, "yIn": 24}}


class LearningObjectiveTests(unittest.TestCase):
    def test_intrinsic_deadline_suppresses_bootstrap_but_canonical_cutoff_does_not(self):
        objective = LearningObjective(TASK)
        self.assertTrue(objective.terminal(False, True, {"reason": "time_limit"}))
        self.assertFalse(LearningObjective(TASK, "canonical-v1").terminal(False, True, {"reason": "time_limit"}))
        advantage, _ = advantages([-2], [0], [10], [objective.terminal(False, True, {"reason": "time_limit"})], [True])
        self.assertEqual(float(advantage[0]), -2)

    def test_learning_reward_uses_measured_components_and_is_separate_from_canonical_reward(self):
        components = {"progress": 0.01, "time": -0.0005, "effort": -0.001, "contact": 0, "success": 0, "fault": 0}
        info = {"reason": "running", "rewardComponents": components, "headingErrorRadians": 0}
        objective = LearningObjective(TASK)
        expected = 0.01 / (24 * 0.0254) - 0.0005 - 0.1
        self.assertAlmostEqual(objective.reward(sum(components.values()), info), expected)
        self.assertEqual(LearningObjective(TASK, "canonical-v1").reward(0.4, {}), 0.4)
        with self.assertRaisesRegex(ValueError, "components"):
            objective.reward(0.1, {})

    def test_heading_progress_is_signed_and_telescopes_without_actor_input_leakage(self):
        task = {"start": {"xIn": 0, "yIn": 0, "headingDeg": 0},
                "goal": {"xIn": 0, "yIn": 24, "headingDeg": 90}}
        objective = LearningObjective(task)
        components = {"progress": 0, "time": 0, "effort": 0, "contact": 0, "success": 0, "fault": 0}
        toward = objective.reward(0, {"rewardComponents": components, "headingErrorRadians": 0.5})
        away = objective.reward(0, {"rewardComponents": components, "headingErrorRadians": 1.5})
        final = objective.reward(0, {"rewardComponents": components, "headingErrorRadians": 0})
        self.assertLess(away, 0)
        self.assertAlmostEqual(toward + away + final, 0.5)

    def test_broad_pace_initialization_is_explicit_and_preserves_bounded_mean_actions(self):
        torch.manual_seed(2)
        model = MotionActorCritic("broad-pace-v1")
        distribution, _ = model(torch.zeros(34))
        self.assertAlmostEqual(float(distribution.mean[0].detach()), 0.75, places=6)
        self.assertGreater(float(distribution.variance[0].detach()), 0.02)
        self.assertTrue(all(-1 <= value <= 1 for value in model.act([0] * 34, deterministic=True)[0]))
        with self.assertRaisesRegex(ValueError, "initialization"):
            MotionActorCritic("unknown")


if __name__ == "__main__":
    unittest.main()
