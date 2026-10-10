import copy
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import DEADLINE_CONTEXT
from roboproof.motion_learning.train import compatible_config, curriculum_definition, online_anchor_definition, run, source_identity, train_seed, validate_online_anchor_interval


def configuration(seed, updates=7, interval=2):
    return {"updates": updates, "maxSteps": 5000, "rolloutSteps": 16, "seed": seed, "seedCount": 1,
            "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
            "curriculum": "mixed-full-reach-v5", "curriculumDefinition": curriculum_definition("mixed-full-reach-v5"),
            "featureTransform": "deadline-context-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
            "explorationConcentration": 8, "objective": "finite-total-return-pose-effort-v3",
            "policyRegularization": "public-deadline-residual-v1", "deadlinePaceLossWeight": 40,
            "warmStartUpdates": 1, "demonstrationAnchorWeight": 10, "demonstrationAnchorLossProfile": "pose-budget-v2",
            "teacherSelection": "pose-deadline-margin-v2", "onlineAnchorInterval": interval,
            "onlineAnchorDefinition": online_anchor_definition(interval)}


class OnlineAnchorTests(unittest.TestCase):
    def test_real_periodic_retention_counts_every_teacher_decision_and_resumes_exactly(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = configuration(92001)
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 92001, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 3:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 92001, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 92001, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertEqual(complete["collectionSteps"], resumed["collectionSteps"])
            teacher_decisions = imitation_rows = 0
            for row in complete["history"]:
                refreshed = row["phase"] == "ppo" and (row["update"] - 2) % 2 == 0
                self.assertEqual(row["onlineTeacherQueries"], 3 if refreshed else 0)
                self.assertEqual(row["onlineAnchorInterval"], 2)
                self.assertAlmostEqual(row["meanTrainingReward"], sum(world["reward"] for world in row["worlds"]) / len(row["worlds"]))
                imitation_rows += row.get("imitationRows", 0)
                for world in row["worlds"]:
                    for attempt in [*world.get("teacherAttempts", []), *world.get("onlineTeacherAttempts", [])]:
                        teacher_decisions += math.ceil(attempt["metrics"]["elapsedSeconds"] / 0.05 - 1e-7)
            self.assertEqual(complete["collectionSteps"], complete["steps"] - imitation_rows + teacher_decisions)
            self.assertTrue(complete["actorWeightsChanged"])
            self.assertTrue(all(row["maximumActorGradientNorm"] > 0 for row in complete["history"] if row["phase"] == "ppo"))
            changed = {**config, "onlineAnchorInterval": 1, "onlineAnchorDefinition": online_anchor_definition(1)}
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 92001, changed, sources, time.monotonic() + 90, resume=True)

    def test_completed_online_teacher_is_not_committed_when_ppo_collection_interrupts(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = configuration(92002, updates=3, interval=1)
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            state = {"enabled": False, "expired": False, "phase": None, "spent": 0, "ppoSteps": 0, "teachers": 0}
            warm = {}
            deadline = time.monotonic() + 90
            original_reset, original_step = environment.reset, environment.step
            def tracked_reset(*args, **kwargs):
                state["phase"] = kwargs.get("options", {}).get("policyIdentity", {}).get("id")
                if state["enabled"] and state["phase"] == "training-teacher":
                    state["teachers"] += 1
                return original_reset(*args, **kwargs)
            def tracked_step(action):
                result = original_step(action)
                if state["enabled"]:
                    state["spent"] += 1
                    if state["phase"] == "ppo-training-rollout":
                        state["ppoSteps"] += 1
                        state["expired"] = state["ppoSteps"] >= 2
                return result
            def checkpoint(row):
                if row["updates"] == 1 and not state["enabled"]:
                    warm["record"] = copy.deepcopy(row)
                    warm["checkpoint"] = torch.load(directory / "resumed/checkpoint.pt", weights_only=True)
                    state["enabled"] = True
            clock = SimpleNamespace(monotonic=lambda: deadline + 1 if state["expired"] else time.monotonic())
            with mock.patch.object(environment, "reset", tracked_reset), mock.patch.object(environment, "step", tracked_step), \
                 mock.patch("roboproof.motion_learning.train.time", clock), mock.patch("roboproof.motion_learning.imitation.time", clock):
                partial = train_seed(environment, directory / "resumed", 92002, config, sources, deadline, on_checkpoint=checkpoint)
            self.assertEqual(partial["updates"], 1)
            self.assertEqual(state["teachers"], 3)
            self.assertEqual(state["ppoSteps"], 2)
            self.assertEqual(partial["discardedUncommittedSteps"], state["spent"])
            saved = torch.load(directory / "resumed/checkpoint.pt", weights_only=True)
            for name in ("features", "targets"):
                self.assertTrue(torch.equal(saved["measuredAnchor"][name], warm["checkpoint"]["measuredAnchor"][name]))
            resumed = train_seed(environment, directory / "resumed", 92002, config, sources, time.monotonic() + 90, resume=True)
            complete = train_seed(environment, directory / "complete", 92002, config, sources, time.monotonic() + 90)
            self.assertEqual(resumed["policySha256"], complete["policySha256"])
            self.assertEqual(resumed["history"], complete["history"])
            self.assertEqual(resumed["collectionSteps"], complete["collectionSteps"] + state["spent"])
            self.assertEqual(resumed["discardedCollectionStepsTotal"], state["spent"])

    def test_guards_official_definition_and_default_disabled_reproduction(self):
        for invalid in (True, -1, 17, 2.5, math.nan, "4"):
            with self.assertRaises(ValueError):
                validate_online_anchor_interval(invalid, 1, 10)
            with self.assertRaises(ValueError):
                run(online_anchor_interval=invalid)
        for warm_updates, weight in ((0, 10), (1, 0), (True, 10), (1, math.inf)):
            with self.assertRaises(ValueError):
                validate_online_anchor_interval(4, warm_updates, weight)
        config = configuration(92003, updates=3, interval=0)
        legacy = {key: value for key, value in config.items() if key not in ("onlineAnchorInterval", "onlineAnchorDefinition")}
        self.assertTrue(compatible_config(legacy, config))
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            omitted = train_seed(environment, directory / "omitted", 92003, legacy, sources, time.monotonic() + 90)
            explicit = train_seed(environment, directory / "explicit", 92003, config, sources, time.monotonic() + 90)
            self.assertEqual(omitted["policySha256"], explicit["policySha256"])
            self.assertEqual(omitted["history"], explicit["history"])
            self.assertTrue(all(row["onlineTeacherQueries"] == 0 for row in explicit["history"]))
            with self.assertRaisesRegex(ValueError, "definition"):
                train_seed(environment, directory / "bad", 92003, {**config, "onlineAnchorDefinition": {}}, sources, time.monotonic() + 90)
        with tempfile.TemporaryDirectory() as temporary:
            saved = run(directory=temporary, seed=92004, seed_count=1, updates=2, max_steps=2000, rollout_steps=16,
                        max_seconds=90, warm_start_updates=1, demonstration_anchor_weight=10, online_anchor_interval=4)
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["config"]["onlineAnchorDefinition"], online_anchor_definition(4))
        self.assertFalse(saved["learnedImprovementVerified"])


if __name__ == "__main__":
    unittest.main()
