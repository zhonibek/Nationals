import json
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
from roboproof.motion_learning.model import DEADLINE_CONTEXT
from roboproof.motion_learning.train import compatible_config, learning_rate_definition, run, source_identity, train_seed, validate_ppo_actor_learning_rate


def configuration(rate):
    return {"updates": 4, "maxSteps": 3000, "rolloutSteps": 16, "seed": 94001, "seedCount": 1,
            "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
            "featureTransform": "deadline-context-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
            "explorationConcentration": 8, "objective": "finite-total-return-pose-effort-v3",
            "policyRegularization": "public-deadline-residual-v1", "deadlinePaceLossWeight": 40,
            "warmStartUpdates": 1, "demonstrationAnchorWeight": 10, "demonstrationAnchorLossProfile": "pose-budget-v2",
            "teacherSelection": "pose-deadline-margin-v2", "ppoActorLearningRate": rate,
            "learningRateDefinition": learning_rate_definition(rate)}


class ActorRateTests(unittest.TestCase):
    def test_actual_phase_switch_keeps_identical_imitation_and_resumes_optimizer_exactly(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = configuration(0.0003)
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 94001, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 94001, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            saved = torch.load(directory / "resumed/checkpoint.pt", weights_only=True)
            self.assertEqual(saved["optimizer"]["param_groups"][0]["lr"], 0.0003)
            self.assertEqual(saved["optimizer"]["param_groups"][1]["lr"], 0.0003)
            resumed = train_seed(environment, directory / "resumed", 94001, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(complete["actorWeightsChanged"])
            self.assertEqual(complete["history"][0]["actorLearningRate"], 0.001)
            self.assertTrue(all(row["actorLearningRate"] == 0.0003 for row in complete["history"][1:]))
            self.assertTrue(all(row["criticLearningRate"] == 0.0003 for row in complete["history"]))
            control = train_seed(environment, directory / "control", 94001, configuration(0.001), sources, time.monotonic() + 90)
            warm = json.loads((directory / "complete/policy-0001.json").read_text())
            warm_control = json.loads((directory / "control/policy-0001.json").read_text())
            self.assertEqual(warm, warm_control)
            self.assertNotEqual(complete["policySha256"], control["policySha256"])
            self.assertTrue(all(row["maximumActorGradientNorm"] > 0 for row in complete["history"][1:]))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 94001, configuration(0.001), sources, time.monotonic() + 90, resume=True)

    def test_default_explicit_rate_preserves_actor_and_positive_range_guards(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        explicit = configuration(0.001)
        legacy = {key: value for key, value in explicit.items() if key not in ("ppoActorLearningRate", "learningRateDefinition")}
        declared = {**legacy, "ppoActorLearningRate": 0.001}
        self.assertTrue(compatible_config(legacy, declared))
        self.assertFalse(compatible_config(legacy, {**declared, "ppoActorLearningRate": 0.0003}))
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            omitted = train_seed(environment, directory / "omitted", 94001, legacy, sources, time.monotonic() + 90)
            specified = train_seed(environment, directory / "explicit", 94001, declared, sources, time.monotonic() + 90)
            self.assertEqual(omitted["policySha256"], specified["policySha256"])
            self.assertEqual(omitted["history"], specified["history"])
            with self.assertRaisesRegex(ValueError, "definition"):
                train_seed(environment, directory / "bad", 94001, {**explicit, "learningRateDefinition": {}}, sources, time.monotonic() + 90)
        for invalid in (True, 0, -0.001, 0.00001, 0.002, math.nan, math.inf, "0.0003"):
            with self.assertRaises(ValueError):
                validate_ppo_actor_learning_rate(invalid)
            with self.assertRaises(ValueError):
                run(ppo_actor_learning_rate=invalid)
        self.assertEqual(validate_ppo_actor_learning_rate(0.00005), 0.00005)

    def test_official_run_binds_phase_specific_learning_rate_without_verification_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = run(directory=temporary, seed=94002, seed_count=1, updates=2, max_steps=2000,
                        rollout_steps=16, max_seconds=90, warm_start_updates=1, ppo_actor_learning_rate=0.0003)
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["config"]["actorLearningRate"], 0.001)
        self.assertEqual(saved["config"]["ppoActorLearningRate"], 0.0003)
        self.assertEqual(saved["config"]["learningRateDefinition"], learning_rate_definition(0.0003))
        self.assertFalse(saved["learnedImprovementVerified"])


if __name__ == "__main__":
    unittest.main()
