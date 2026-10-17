import copy
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
from roboproof.motion_learning.td3_train import configuration, run, source_identity, train_seed


def arguments(updates=4):
    return dict(seed=1250054, seed_count=1, updates=updates, max_steps=3000, rollout_steps=16,
        max_seconds=90, warm_start_updates=1, gradient_steps=4, critic_warmup_steps=8)


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
        for original, restored in zip(first, second):
            assert_identical(test, original, restored)
    else:
        test.assertEqual(first, second)


class CriticWarmupNativeTests(unittest.TestCase):
    def test_configuration_is_explicit_optional_and_bounded(self):
        required = (1250054, 1, 4, 3000, 16, 1, "deadline-context-v1", "finite-total-return-settle-margin-v4",
                    "mixed-full-reach-v5", 4, 1024, 0.05, 10, "public-deadline-residual-v1", 40)
        default = configuration(*required)
        self.assertEqual(default, configuration(*required, critic_warmup_steps=0))
        self.assertNotIn("criticWarmupSteps", default)
        self.assertNotIn("criticWarmupDefinition", default)
        actual = configuration(*required, critic_warmup_steps=256)
        self.assertEqual(set(actual) - set(default), {"criticWarmupSteps", "criticWarmupDefinition"})
        self.assertEqual(actual["criticWarmupDefinition"]["gradientSteps"], 256)
        self.assertEqual(actual["criticWarmupDefinition"]["method"], "initial-native-critic-only-v1")
        for value in (True, False, -1, 513, 1.0, "256", None):
            with self.assertRaises(ValueError):
                configuration(*required, critic_warmup_steps=value)

    def test_original_engine_prefix_resumes_exactly_across_first_actor_update(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            complete = run(directory=directory / "complete", **arguments())
            prefix = run(directory=directory / "resumed", **arguments(2))
            prefix_folder = directory / "resumed" / prefix["runId"] / "seed-1250054"
            saved_prefix = torch.load(prefix_folder / "checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertEqual(saved_prefix["learner"]["gradientUpdates"], 4)
            self.assertEqual(saved_prefix["learner"]["actorUpdates"], 0)
            self.assertEqual(saved_prefix["learner"]["schemaVersion"], 2)
            self.assertEqual(saved_prefix["learner"]["criticWarmupSteps"], 8)
            prefix_policy = json.loads((prefix_folder / "policy-0002.json").read_text())
            warm_policy = json.loads((prefix_folder / "policy-0001.json").read_text())
            self.assertEqual(prefix_policy["layers"], warm_policy["layers"])
            resumed = run(directory=directory / "resumed", resume=prefix["runId"], **arguments())
            complete_folder = directory / "complete" / complete["runId"] / "seed-1250054"
            saved = torch.load(complete_folder / "checkpoint.pt", weights_only=True, map_location="cpu")
            actual = torch.load(prefix_folder / "checkpoint.pt", weights_only=True, map_location="cpu")
            assert_identical(self, saved, actual)
            self.assertEqual(complete["status"], "completed")
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(saved["learner"]["gradientUpdates"], 12)
            self.assertEqual(saved["learner"]["actorUpdates"], 2)
            self.assertEqual([row["actorStepsThisCollection"] for row in saved["history"][1:]], [0, 0, 2])
            final_policy = json.loads((complete_folder / "policy-0004.json").read_text())
            self.assertTrue(all(before != after for before, after in zip(warm_policy["layers"][-1]["weight"], final_policy["layers"][-1]["weight"])))
            with self.assertRaisesRegex(ValueError, "source/configuration mismatch"):
                run(directory=directory / "resumed", resume=prefix["runId"], **{**arguments(), "critic_warmup_steps": 9})
            malformed = copy.deepcopy(complete["config"])
            malformed["criticWarmupDefinition"]["method"] = "unrecorded-method"
            with MotionEnvironment() as environment, self.assertRaisesRegex(ValueError, "definition mismatch"):
                train_seed(environment, directory / "invalid", 1250054, malformed, source_identity(), time.monotonic() + 30)
            self.assertFalse((directory / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
