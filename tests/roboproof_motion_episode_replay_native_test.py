import copy
import json
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
from roboproof.motion_learning.episode_replay import STRATEGY, definition
from roboproof.motion_learning.td3_train import run, source_identity, train_seed


def arguments(updates=4):
    return dict(seed=1360053, seed_count=1, updates=updates, max_steps=4000, rollout_steps=64,
        max_seconds=90, warm_start_updates=1, gradient_steps=4, replay_capacity=64, n_steps=32,
        replay_strategy=STRATEGY)


def assert_identical(test, first, second):
    if isinstance(first, torch.Tensor):
        test.assertIsInstance(second, torch.Tensor)
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for name, value in first.items():
            assert_identical(test, value, second[name])
    elif isinstance(first, (tuple, list)):
        test.assertIs(type(first), type(second))
        test.assertEqual(len(first), len(second))
        for value, matching in zip(first, second):
            assert_identical(test, value, matching)
    else:
        test.assertEqual(first, second)


class EpisodeReplayNativeTests(unittest.TestCase):
    def test_original_engine_packed_episode_lengths_resume_all_states_exactly(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            complete = run(directory=directory / "complete", **arguments())
            prefix = run(directory=directory / "resumed", **arguments(2))
            resumed = run(directory=directory / "resumed", resume=prefix["runId"], **arguments())
            first_folder = directory / "complete" / complete["runId"] / "seed-1360053"
            second_folder = directory / "resumed" / prefix["runId"] / "seed-1360053"
            first = torch.load(first_folder / "checkpoint.pt", weights_only=True, map_location="cpu")
            second = torch.load(second_folder / "checkpoint.pt", weights_only=True, map_location="cpu")
            assert_identical(self, first, second)
            self.assertEqual(complete["status"], "completed")
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(complete["models"][0]["policySha256"], resumed["models"][0]["policySha256"])
            replay = first["replay"]
            self.assertEqual(replay["schemaVersion"], 4)
            self.assertEqual(replay["strategy"], STRATEGY)
            self.assertTrue(torch.all(replay["episodeLengths"][:replay["count"]] >= replay["horizons"][:replay["count"]]))
            self.assertTrue(torch.any(replay["episodeLengths"][:replay["count"]] > 32))
            self.assertEqual(first["config"]["episodeReplayDefinition"], definition())
            self.assertEqual(first["config"]["td3Definition"]["criticSampling"], definition())
            warm = json.loads((first_folder / "policy-0001.json").read_text())
            final = json.loads((first_folder / "policy-0004.json").read_text())
            self.assertTrue(all(before != after for before, after in zip(warm["layers"][-1]["weight"], final["layers"][-1]["weight"])))
            with self.assertRaisesRegex(ValueError, "source/configuration mismatch"):
                run(directory=directory / "resumed", resume=prefix["runId"], **{**arguments(), "replay_strategy": "uniform-reservoir-v1"})
            altered = copy.deepcopy(first["config"])
            altered["episodeReplayDefinition"]["probability"] = "unrecorded sampling"
            with MotionEnvironment() as environment, self.assertRaisesRegex(ValueError, "definition mismatch"):
                train_seed(environment, directory / "invalid", 1360053, altered, source_identity(), time.monotonic() + 30)
            self.assertFalse((directory / "invalid").exists())

    def test_terminal_mixture_is_rejected_before_artifacts_or_engine_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "invalid"
            with patch("roboproof.motion_learning.td3_train.MotionEnvironment") as environment:
                with self.assertRaisesRegex(ValueError, "cannot combine terminal mixture"):
                    run(directory=directory, **{**arguments(), "critic_terminal_fraction": 0.25})
                environment.assert_not_called()
            self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()
