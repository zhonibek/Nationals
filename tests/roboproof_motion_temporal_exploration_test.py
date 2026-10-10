import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_learning.td3 import TrainingExploration
from roboproof.motion_learning.td3_train import configuration, compatible_config


class TemporalExplorationTests(unittest.TestCase):
    def test_default_iid_exact_rng_and_values_and_reset_no_rng_change(self):
        torch.manual_seed(108001)
        expected = torch.randn(4) * 0.05
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(108001)
        noise = TrainingExploration()
        noise.reset()
        torch.testing.assert_close(noise.sample(), expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        noise.reset()
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_correlated_pose_matches_independent_recurrence_and_episode_reset(self):
        torch.manual_seed(108002)
        state = torch.zeros(3)
        decay = math.exp(-0.05)
        expected = []
        for step in range(6):
            draw = torch.randn(4)
            state = decay * state + 0.2 * math.sqrt(1 - decay ** 2) * draw[1:]
            expected.append(torch.cat((draw[:1] * 0.05, state)))
        torch.manual_seed(108002)
        noise = TrainingExploration(0.05, "ou-pose-v1", 0.2, 1)
        for wanted in expected:
            torch.testing.assert_close(noise.sample(), wanted, rtol=0, atol=0)
        before = torch.get_rng_state().clone()
        noise.reset()
        torch.testing.assert_close(noise.pose_state, torch.zeros(3), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_profile_parameters_fail_closed_and_configuration_resume_is_bound(self):
        for parameters in ((0.05, "iid-v1", 0.2, 0), (0.05, "iid-v1", 0.05, 1),
                           (0.05, "ou-pose-v1", 0.2, 0), (0.05, "ou-pose-v1", 0.21, 1),
                           (0.05, "ou-pose-v1", 0.2, math.nan), (True, "iid-v1", None, 0)):
            with self.assertRaises(ValueError):
                TrainingExploration(*parameters)
        parameters = (201, 1, 64, 50000, 256, 8, "deadline-context-v1", "finite-total-return-settle-margin-v4",
                      "mixed-full-reach-v5", 32, 1024, 0.05, 1, "public-deadline-residual-v1", 4, 8)
        default = configuration(*parameters)
        selected = configuration(*parameters, exploration_profile="ou-pose-v1", pose_exploration_std=0.2, exploration_correlation_seconds=1)
        self.assertEqual(default["explorationProfile"], "iid-v1")
        self.assertEqual(default["poseExplorationStd"], 0.05)
        self.assertEqual(selected["td3Definition"]["explorationDefinition"]["poseStd"], 0.2)
        self.assertFalse(compatible_config(default, selected))
        self.assertFalse(compatible_config(selected, default))


if __name__ == "__main__":
    unittest.main()
