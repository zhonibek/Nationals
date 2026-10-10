import copy
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
from roboproof.motion_learning.objective import (
    LearningObjective, TOTAL_RETURN_OBJECTIVE, SETTLING_MARGIN_OBJECTIVE, SETTLING_TOLERANCES,
    objective_definition, terminal_settling_margin, discount_parameters,
)
from roboproof.motion_learning.train import compatible_config, run, source_identity, train_seed


TASK = {"start": {"xIn": 0, "yIn": 0, "headingDeg": 0},
        "goal": {"xIn": 0, "yIn": 24, "headingDeg": 90}, "deadlineSeconds": 6}
COMPONENTS = {"progress": 0.01, "time": -0.0005, "effort": -0.001,
              "contact": 0, "success": 0, "fault": 0}


class SettlingMarginTests(unittest.TestCase):
    def test_four_criteria_are_monotonic_bounded_and_use_existing_success_units(self):
        self.assertEqual(SETTLING_TOLERANCES, {"positionErrorMeters": 0.02032, "headingErrorRadians": 0.035,
                                            "speedMetersPerSecond": 0.0254, "yawRateRadiansPerSecond": 0.0873})
        perfect = {"reason": "success", **{name: 0 for name in SETTLING_TOLERANCES}}
        self.assertEqual(terminal_settling_margin(perfect), 1)
        for name, tolerance in SETTLING_TOLERANCES.items():
            scores = [terminal_settling_margin({**perfect, name: tolerance * fraction})
                      for fraction in (0, 0.25, 0.8, 1, 2, 10)]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertAlmostEqual(scores[3], 0.5)
            self.assertTrue(all(0 <= score <= 1 for score in scores))
            self.assertGreater(terminal_settling_margin({**perfect, name: 1e308}), 0)
        combined = {"reason": "time_limit", **{name: 0.8 * tolerance for name, tolerance in SETTLING_TOLERANCES.items()}}
        self.assertAlmostEqual(terminal_settling_margin(combined), 1 / 1.8)

    def test_terminal_only_feedback_leaves_running_canonical_and_fault_rewards_unchanged(self):
        baseline = LearningObjective(TASK, TOTAL_RETURN_OBJECTIVE)
        candidate = LearningObjective(TASK, SETTLING_MARGIN_OBJECTIVE)
        components = copy.deepcopy(COMPONENTS)
        for reason, heading in (("running", 1.2), ("running", 0.8), ("controller_or_sensor_fault", 0.5)):
            info = {"reason": reason, "rewardComponents": components, "headingErrorRadians": heading}
            self.assertEqual(candidate.reward(0.1, info), baseline.reward(0.1, info))
            self.assertIsNone(candidate.terminal_margin_score)
        terminal = {"reason": "time_limit", "rewardComponents": components,
                    **{name: 0.5 * tolerance for name, tolerance in SETTLING_TOLERANCES.items()}}
        self.assertAlmostEqual(candidate.reward(0.1, terminal) - baseline.reward(0.1, terminal), 2 / 1.5)
        self.assertEqual(components, COMPONENTS)
        self.assertEqual(discount_parameters(SETTLING_MARGIN_OBJECTIVE), (1, 1))
        self.assertTrue(candidate.terminal(False, True, terminal))
        self.assertFalse(candidate.terminal(False, False, {"reason": "running"}))

    def test_invalid_measured_metrics_fail_closed_and_definitions_are_not_mutable_aliases(self):
        info = {"reason": "success", **{name: 0 for name in SETTLING_TOLERANCES}}
        for name in SETTLING_TOLERANCES:
            for invalid in (None, True, -1, math.nan, math.inf, "0.1"):
                with self.assertRaisesRegex(ValueError, "measured"):
                    terminal_settling_margin({**info, name: invalid})
            missing = dict(info)
            del missing[name]
            with self.assertRaises(ValueError):
                terminal_settling_margin(missing)
        definition = objective_definition(SETTLING_MARGIN_OBJECTIVE)
        definition["successTolerances"]["positionErrorMeters"] = 1
        self.assertEqual(SETTLING_TOLERANCES["positionErrorMeters"], 0.02032)
        self.assertIsNone(objective_definition(TOTAL_RETURN_OBJECTIVE))
        with self.assertRaises(ValueError):
            objective_definition("unknown")

    def test_original_simulator_terminal_metrics_and_return_accounting_match_independent_formula(self):
        with MotionEnvironment() as environment:
            observation, _ = environment.reset(seed=97001, task=TASK)
            candidate = LearningObjective(TASK, SETTLING_MARGIN_OBJECTIVE)
            baseline = LearningObjective(TASK, TOTAL_RETURN_OBJECTIVE)
            difference = 0
            while True:
                observation, reward, terminated, truncated, info = environment.step([1, 0, 0, 0])
                difference += candidate.reward(reward, info) - baseline.reward(reward, info)
                if terminated or truncated:
                    break
                self.assertIsNone(candidate.terminal_margin_score)
            report = environment.report()
            self.assertIn(report["reason"], ("success", "time_limit"))
            ratios = [report["metrics"][name] / tolerance for name, tolerance in SETTLING_TOLERANCES.items()]
            self.assertAlmostEqual(difference, 2 / (1 + max(ratios)), places=12)
            self.assertAlmostEqual(candidate.terminal_margin_score, difference / 2)
            self.assertEqual(len(observation), 34)
            self.assertEqual(environment.contract["actionSize"], 4)

    def test_real_training_changes_actor_after_imitation_resumes_exactly_and_binds_metadata(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 3, "maxSteps": 3000, "rolloutSteps": 16, "seed": 97002, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "deadline-context-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
                  "explorationConcentration": 8,
                  "objective": SETTLING_MARGIN_OBJECTIVE, "objectiveDefinition": objective_definition(SETTLING_MARGIN_OBJECTIVE),
                  "warmStartUpdates": 1, "demonstrationAnchorWeight": 10, "demonstrationAnchorLossProfile": "pose-budget-v2",
                  "policyRegularization": "public-deadline-residual-v1", "deadlinePaceLossWeight": 40}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 97002, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 97002, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 97002, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            warm = json.loads((directory / "complete/policy-0001.json").read_text())["layers"]
            learned = json.loads((directory / "complete/policy-0003.json").read_text())["layers"]
            self.assertNotEqual(warm, learned)
            for row in complete["history"][1:]:
                for world in row["worlds"]:
                    self.assertTrue(0 <= world["terminalSettlingMarginScore"] <= 1)
            with self.assertRaisesRegex(ValueError, "objective definition mismatch"):
                train_seed(environment, directory / "resumed", 97002,
                           {**config, "objectiveDefinition": {"method": "changed"}}, sources, time.monotonic() + 90, resume=True)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 97002,
                           {**config, "objective": TOTAL_RETURN_OBJECTIVE, "objectiveDefinition": None},
                           sources, time.monotonic() + 90, resume=True)
        self.assertFalse(compatible_config(config, {**config, "objectiveDefinition": None}))
        with tempfile.TemporaryDirectory() as temporary:
            saved = run(directory=temporary, seed=97003, seed_count=1, updates=1, max_steps=1000,
                        rollout_steps=16, max_seconds=90, objective=SETTLING_MARGIN_OBJECTIVE)
        self.assertEqual(saved["config"]["objectiveDefinition"], objective_definition(SETTLING_MARGIN_OBJECTIVE))
        self.assertFalse(saved["learnedImprovementVerified"])


if __name__ == "__main__":
    unittest.main()
