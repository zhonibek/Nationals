import copy
from pathlib import Path
import sys
import tempfile
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import MotionActorCritic
from roboproof.motion_learning.td3 import ReplayBuffer, clipped_target, complete_episode_returns
from roboproof.motion_learning.td3_train import configuration, source_identity, train_seed


def measured_rows():
    return [{"modelInput": [index] * 34, "nextModelInput": [index + 1] * 34, "action": [0.1, 0, 0, 0],
             "reward": reward, "requiredPace": 0.85, "terminal": index in (2, 4)}
            for index, reward in enumerate((1, 2, 3, 10, 20))]


class TD3ReturnTests(unittest.TestCase):
    def test_multi_step_rewards_keep_start_action_end_state_and_never_cross_resets(self):
        rows = measured_rows()
        untouched = copy.deepcopy(rows)
        self.assertIs(complete_episode_returns(rows), rows)
        packed = complete_episode_returns(rows, 3)
        self.assertEqual([row["reward"] for row in packed], [6, 5, 3, 30, 20])
        self.assertEqual([row["horizon"] for row in packed], [3, 2, 1, 2, 1])
        self.assertEqual([row["nextModelInput"][0] for row in packed], [3, 3, 3, 5, 5])
        self.assertTrue(all(row["terminal"] for row in packed))
        short = complete_episode_returns(rows, 2)
        self.assertEqual([row["reward"] for row in short], [3, 5, 3, 30, 20])
        self.assertFalse(short[0]["terminal"])
        self.assertEqual(short[0]["nextModelInput"], rows[1]["nextModelInput"])
        self.assertEqual([row["action"] for row in packed], [row["action"] for row in rows])
        self.assertEqual(rows, untouched)
        discounted = complete_episode_returns(rows, 3, gamma=0.5)
        self.assertEqual([row["reward"] for row in discounted], [2.75, 3.5, 3, 20, 20])
        for invalid in (0, 33, True, 1.5, "8"):
            with self.assertRaises(ValueError):
                complete_episode_returns(rows, invalid)
        for invalid in (0, 1.1, True, float("nan")):
            with self.assertRaises(ValueError):
                complete_episode_returns(rows, 3, invalid)
        with self.assertRaisesRegex(ValueError, "Complete"):
            complete_episode_returns(rows[:-1], 2)

    def test_replay_horizons_and_twin_target_discount_match_independent_formula(self):
        replay = ReplayBuffer(34, 64)
        packed = complete_episode_returns(measured_rows(), 2)
        replay.append(packed)
        self.assertEqual(replay.horizons[:5].tolist(), [2, 2, 1, 2, 1])
        saved = replay.state()
        restored = ReplayBuffer(34, 64)
        restored.restore(saved)
        torch.testing.assert_close(replay.horizons, restored.horizons, rtol=0, atol=0)
        actor = MotionActorCritic("broad-pace-v1", value_critic=False)
        batch = {name: getattr(replay, name)[:5] for name in
                 ("observations", "next_observations", "actions", "rewards", "terminals", "required_paces", "horizons")}
        class Critics(torch.nn.Module):
            def forward(self, observations, actions):
                return torch.full((len(observations),), 4.0), torch.full((len(observations),), 2.0)
        targets, _ = clipped_target(actor, Critics(), batch, gamma=0.5, noise_std=0.05, noise_clip=0.1)
        expected = torch.tensor([3 + 0.5**2 * 2, 5, 3, 30, 20])
        torch.testing.assert_close(targets, expected)
        malformed = copy.deepcopy(saved)
        malformed["horizons"][0] = 0
        with self.assertRaises(ValueError):
            restored.restore(malformed)
        with self.assertRaises(ValueError):
            replay.append([{**packed[0], "horizon": 33}])

    def test_real_n_step_training_resumes_exactly_with_actual_decisions_and_all_heads_updated(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = configuration(103001, 1, 4, 3000, 16, 1, "deadline-context-v1",
                               "finite-total-return-settle-margin-v4", "mixed-full-reach-v5", 4, 1024, 0.05,
                               10, "public-deadline-residual-v1", 40, n_steps=8)
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 103001, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 103001, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 103001, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            checkpoint = torch.load(directory / "complete/checkpoint.pt", weights_only=True)
            replay = checkpoint["replay"]
            self.assertTrue(torch.any(replay["horizons"][:replay["count"]] == 8))
            self.assertTrue(torch.any(replay["horizons"][:replay["count"]] == 1))
            self.assertTrue(complete["actorWeightsChanged"])
            self.assertGreater(complete["collectionSteps"], complete["steps"])
            self.assertTrue(all(row["nStepReturns"] == 8 for row in complete["history"][1:]))
            altered = {**config, "nStepReturns": 1}
            with self.assertRaisesRegex(ValueError, "definition mismatch"):
                train_seed(environment, directory / "resumed", 103001, altered, sources, time.monotonic() + 90, resume=True)


if __name__ == "__main__":
    unittest.main()
