import copy
import math
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import DEADLINE_CONTEXT
from roboproof.motion_learning.train import curriculum_definition, compatible_config, run, source_identity, train_seed, training_world


class FullCurriculumTests(unittest.TestCase):
    def test_seeded_public_family_coverage_and_actual_original_engine_acceptance(self):
        first_generator = random.Random(91001)
        repeated_generator = random.Random(91001)
        worlds = [training_world(first_generator, index, "mixed-full-reach-v5") for index in range(512)]
        repeated = [training_world(repeated_generator, index, "mixed-full-reach-v5") for index in range(512)]
        self.assertEqual(worlds, repeated)
        legacy = random.Random(91001)
        self.assertEqual(worlds[:4], [training_world(legacy, index, "field-reach-v4") for index in range(4)])
        self.assertEqual([world["stage"] for world in worlds[4:12]], [1, 2, 0, 1, 1, 2, 0, 1])
        self.assertEqual({world["task"]["deadlineSeconds"] for world in worlds[4:]}, {6, 10})
        fraction = sum(world["task"]["deadlineSeconds"] == 10 for world in worlds[4:]) / 508
        self.assertGreater(fraction, 0.65)
        self.assertLess(fraction, 0.85)
        full = [world for world in worlds[4:] if world["stage"] == 2]
        self.assertTrue(any(abs(world["task"]["start"][axis]) > 40 for world in full for axis in ("xIn", "yIn")))
        self.assertTrue(any(abs(world["task"]["goal"][axis]) == 60 for world in full for axis in ("xIn", "yIn")))
        ranges = {"massKg": (3.4, 13.6), "moiKgM2": (0.0725, 0.58), "muLong": (0.3, 1.1), "muLat": (0.02, 0.15),
                  "kSlip": (325, 975), "batteryInternalR": (0, 0.12)}
        representatives = {}
        for world in worlds[4:]:
            task = world["task"]
            self.assertTrue(all(math.isfinite(value) for pose in (task["start"], task["goal"]) for value in pose.values()))
            self.assertTrue(all(abs(task["start"][axis]) <= 50 and abs(task["goal"][axis]) <= 60 for axis in ("xIn", "yIn")))
            for name, value in world["configuration"].items():
                self.assertGreaterEqual(value, ranges[name][0])
                self.assertLessEqual(value, ranges[name][1])
            representatives.setdefault((world["stage"], tuple(sorted(world["configuration"]))), world)
        self.assertEqual(len(representatives), 12)
        with MotionEnvironment() as environment:
            for world in representatives.values():
                observation, _ = environment.reset(seed=world["seed"], task=world["task"], configuration=world["configuration"])
                self.assertEqual(len(observation), 34)
                next_observation, reward, _, _, _ = environment.step([1, 0, 0, 0])
                self.assertTrue(all(math.isfinite(value) for value in next_observation))
                self.assertTrue(math.isfinite(reward))

    def test_full_curriculum_real_gradients_and_exact_resume_bind_distribution(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 7, "maxSteps": 5000, "rolloutSteps": 16, "seed": 91002, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "curriculum": "mixed-full-reach-v5", "curriculumDefinition": curriculum_definition("mixed-full-reach-v5"),
                  "featureTransform": "deadline-context-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
                  "explorationConcentration": 8,
                  "objective": "finite-total-return-pose-effort-v3"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 91002, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 3:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 91002, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 91002, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["updates"], 7)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(complete["actorWeightsChanged"])
            self.assertTrue(any(world["task"]["deadlineSeconds"] == 10 for row in complete["history"] for world in row["worlds"]))
            self.assertTrue(any(world["stage"] == 2 for row in complete["history"] for world in row["worlds"]))
            changed = copy.deepcopy(config)
            changed["curriculumDefinition"]["deadlineChoicesSeconds"] = [6]
            self.assertFalse(compatible_config(config, changed))
            with self.assertRaisesRegex(ValueError, "definition"):
                train_seed(environment, directory / "resumed", 91002, changed, sources, time.monotonic() + 90, resume=True)

    def test_official_run_records_full_distribution_and_legacy_profiles_keep_default(self):
        self.assertIsNone(curriculum_definition("field-reach-v4"))
        self.assertIsNone(curriculum_definition("mixed-reach-v2"))
        with self.assertRaises(ValueError):
            curriculum_definition("unknown")
        definition = curriculum_definition("mixed-full-reach-v5")
        definition["familyCycle"].clear()
        self.assertEqual(len(curriculum_definition("mixed-full-reach-v5")["familyCycle"]), 4)
        with tempfile.TemporaryDirectory() as temporary:
            saved = run(directory=temporary, seed=91003, seed_count=1, updates=1, max_steps=500,
                        rollout_steps=16, max_seconds=90, curriculum="mixed-full-reach-v5", exploration_concentration=8)
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["config"]["curriculumDefinition"], curriculum_definition("mixed-full-reach-v5"))
        self.assertFalse(saved["learnedImprovementVerified"])


if __name__ == "__main__":
    unittest.main()
