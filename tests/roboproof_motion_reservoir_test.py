import copy
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_learning.td3 import ReplayBuffer, ReservoirReplayBuffer
from roboproof.motion_learning.td3_train import configuration, compatible_config


def rows(start, stop):
    return [{"modelInput": [index / 100] * 34, "nextModelInput": [(index + 1) / 100] * 34,
             "action": [0.1, -0.1, 0.05, -0.05], "reward": index / 20,
             "requiredPace": 0.9, "terminal": index % 16 == 0, "horizon": index % 8 + 1}
            for index in range(start, stop)]


class ReservoirTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_replay_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for name, value in first.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, second[name], rtol=0, atol=0)
            else:
                self.assertEqual(value, second[name])

    def test_replacement_matches_independent_uniform_reservoir_and_exact_rng(self):
        expected = list(range(64))
        torch.manual_seed(107001)
        for seen in range(65, 257):
            selected = int(torch.randint(seen, (1,)))
            if selected < 64:
                expected[selected] = seen - 1
        expected_rng = torch.get_rng_state()
        torch.manual_seed(107001)
        replay = ReservoirReplayBuffer(34, 64)
        replay.append(rows(0, 256))
        self.assertEqual((replay.count, replay.cursor, replay.seen_rows), (64, 0, 256))
        torch.testing.assert_close(replay.observations[:, 0], torch.tensor([index / 100 for index in expected]), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        self.assertTrue(any(index < 64 for index in expected))
        self.assertTrue(any(index >= 192 for index in expected))

    def test_exact_checkpoint_continuation_and_uniform_sample_after_retention(self):
        torch.manual_seed(107002)
        replay = ReservoirReplayBuffer(34, 64)
        replay.append(rows(0, 130))
        saved, rng = replay.state(), torch.get_rng_state().clone()
        replay.append(rows(130, 256))
        expected, expected_rng = replay.state(), torch.get_rng_state().clone()
        restored = ReservoirReplayBuffer(34, 64)
        restored.restore(saved)
        torch.set_rng_state(rng)
        restored.append(rows(130, 256))
        self.assert_replay_equal(restored.state(), expected)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        torch.manual_seed(107003)
        first = restored.sample(32, 0.25)
        torch.manual_seed(107003)
        second = replay.sample(32, 0.25)
        self.assert_replay_equal(first, second)

    def test_invalid_rows_and_metadata_fail_before_state_or_rng_mutation(self):
        replay = ReservoirReplayBuffer(34, 64)
        replay.append(rows(0, 64))
        saved, rng = replay.state(), torch.get_rng_state().clone()
        invalid = {**rows(65, 66)[0], "action": [2, 0, 0, 0]}
        with self.assertRaises(ValueError):
            replay.append([rows(64, 65)[0], invalid])
        self.assert_replay_equal(replay.state(), saved)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for change in ({"seenRows": True}, {"seenRows": 50001}, {"cursor": 1}, {"count": 63}, {"strategy": "recent-ring-v1"}):
            with self.assertRaises(ValueError):
                replay.restore({**saved, **change})
        malformed = copy.deepcopy(saved)
        malformed["horizons"].fill_(33)
        with self.assertRaises(ValueError):
            replay.restore(malformed)
        self.assert_replay_equal(replay.state(), saved)
        with self.assertRaises(ValueError):
            ReplayBuffer(34, 64).restore(saved)
        with self.assertRaises(ValueError):
            replay.restore(ReplayBuffer(34, 64).state())

    def test_default_ring_rng_unchanged_and_strategy_configuration_bound(self):
        torch.manual_seed(107004)
        rng = torch.get_rng_state().clone()
        replay = ReplayBuffer(34, 64)
        replay.append(rows(0, 128))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        torch.testing.assert_close(replay.observations[:, 0], torch.tensor([index / 100 for index in range(64, 128)]), rtol=0, atol=0)
        parameters = (201, 1, 64, 50000, 256, 8, "deadline-context-v1", "finite-total-return-settle-margin-v4",
                      "mixed-full-reach-v5", 32, 1024, 0.05, 1, "public-deadline-residual-v1", 4, 8)
        default = configuration(*parameters)
        selected = configuration(*parameters, replay_strategy="uniform-reservoir-v1")
        self.assertEqual(default["replayStrategy"], "recent-ring-v1")
        self.assertEqual(selected["td3Definition"]["replayRetention"]["method"], "uniform-reservoir-v1")
        self.assertFalse(compatible_config(default, selected))
        self.assertFalse(compatible_config(selected, default))
        with self.assertRaises(ValueError):
            configuration(*parameters, replay_strategy="outcome-selected")


if __name__ == "__main__":
    unittest.main()
