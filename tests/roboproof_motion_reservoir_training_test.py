import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.td3_train import configuration, source_identity, train_seed


class ReservoirTrainingTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for name in actual:
                self.assert_state_equal(actual[name], expected[name])
        elif isinstance(actual, (list, tuple)):
            self.assertIs(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for item, matching in zip(actual, expected):
                self.assert_state_equal(item, matching)
        else:
            self.assertEqual(actual, expected)

    def test_original_simulator_reservoir_replacement_exact_resume_all_state_and_work(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = configuration(107005, 1, 4, 3000, 16, 1, "deadline-context-v1",
                               "finite-total-return-settle-margin-v4", "mixed-full-reach-v5", 4, 64, 0.05,
                               1, "public-deadline-residual-v1", 4, 8, 0, "uniform-reservoir-v1")
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 107005, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 107005, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 107005, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertEqual(complete["collectionSteps"], resumed["collectionSteps"])
            first = torch.load(directory / "complete/checkpoint.pt", weights_only=True)
            second = torch.load(directory / "resumed/checkpoint.pt", weights_only=True)
            for name in ("model", "learner", "replay", "torchRng", "worldRng", "config"):
                self.assert_state_equal(first[name], second[name])
            self.assertEqual(first["replay"]["schemaVersion"], 3)
            self.assertEqual(first["replay"]["count"], 64)
            self.assertGreater(first["replay"]["seenRows"], 64)
            self.assertEqual(first["replay"]["seenRows"], sum(row["acceptedRows"] for row in complete["history"] if row["phase"] == "td3"))
            self.assertLess((directory / "complete/checkpoint.pt").stat().st_size, 4 * 1024 * 1024)
            initial = json.loads((directory / "complete/policy-0001.json").read_text())
            learned = json.loads((directory / "complete/policy-0004.json").read_text())
            self.assertTrue(all(before != after for before, after in zip(initial["layers"][2]["weight"], learned["layers"][2]["weight"])))


if __name__ == "__main__":
    unittest.main()
