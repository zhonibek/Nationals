import copy
import inspect
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.td3_train import configuration, online_anchor_definition, run, source_identity, train_seed


def arguments(seed=790001, updates=4, interval=2):
    defaults = {name: parameter.default for name, parameter in inspect.signature(run).parameters.items()}
    result = {name: defaults[name] for name in inspect.signature(configuration).parameters}
    result.update(seed=seed, updates=updates, rollout_steps=16, max_steps=5000, warm_start_updates=1,
                  gradient_steps=2, replay_capacity=64, online_anchor_interval=interval)
    return result


def identical(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(identical(value, second[name]) for name, value in first.items())
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(identical(left, right) for left, right in zip(first, second))
    return first == second


class Td3OnlineAnchorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_disabled_configuration_is_exact_and_enabled_definition_is_required(self):
        baseline_arguments = arguments(interval=0)
        omitted = {name: value for name, value in baseline_arguments.items() if name != "online_anchor_interval"}
        baseline = configuration(**omitted)
        self.assertEqual(baseline, configuration(**baseline_arguments))
        self.assertNotIn("onlineAnchorInterval", baseline)
        self.assertNotIn("onlineAnchorDefinition", baseline)
        enabled = configuration(**arguments())
        self.assertEqual(enabled["onlineAnchorDefinition"], online_anchor_definition(2))
        for value in (True, -1, 17, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                configuration(**{**baseline_arguments, "online_anchor_interval": value})
        with self.assertRaises(ValueError):
            configuration(**{**arguments(), "demonstration_anchor_weight": 0})
        with self.assertRaises(ValueError):
            configuration(**{**arguments(), "warm_start_updates": 0, "demonstration_anchor_weight": 0})
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            with self.assertRaisesRegex(ValueError, "definition mismatch"):
                train_seed(environment, temporary, 790001, {**enabled, "onlineAnchorDefinition": {}},
                           source_identity(), time.monotonic() + 90)

    def test_actual_refresh_queries_and_all_learning_state_resume_exactly(self):
        config = configuration(**arguments())
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", config["seed"], config, sources, time.monotonic() + 90)

            class Interrupted(Exception):
                pass

            def interrupt(row):
                if row["updates"] == 3:
                    raise Interrupted()

            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", config["seed"], config, sources, time.monotonic() + 90,
                           on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", config["seed"], config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertTrue(identical(torch.load(directory / "complete/checkpoint.pt", weights_only=True),
                                      torch.load(directory / "resumed/checkpoint.pt", weights_only=True)))
            measured_queries = 0
            rollout_rows = 0
            for index, row in enumerate(complete["history"]):
                refreshed = index >= 1 and (index - 1) % 2 == 0
                self.assertEqual(row["onlineTeacherQueries"], 3 if refreshed else 0)
                self.assertEqual(row["onlineAnchorRowsAdded"] > 0, row["phase"] != "td3" or refreshed)
                if row["phase"] == "td3":
                    rollout_rows += row["acceptedRows"]
                for world in row["worlds"]:
                    for attempt in [*world.get("teacherAttempts", []), *world.get("onlineTeacherAttempts", [])]:
                        measured_queries += math.ceil(attempt["metrics"]["elapsedSeconds"] / 0.05 - 1e-7)
                self.assertEqual(row["onlineTeacherDecisions"], sum(world.get("onlineTeacherObservedSteps", 0) for world in row["worlds"]))
            self.assertEqual(complete["collectionSteps"], measured_queries + rollout_rows)
            exposure = json.loads((directory / "complete/exposure.json").read_text())["cases"]
            self.assertEqual(sum(world["kind"] == "td3-online-teacher" for world in exposure), 2)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", config["seed"], configuration(**arguments(interval=1)),
                           sources, time.monotonic() + 90, resume=True)

    def test_incomplete_teacher_or_rollout_never_commits_pending_labels(self):
        for phase in ("training-teacher", "td3-training-exploration"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
                config = configuration(**arguments(seed=790002, updates=3, interval=1))
                directory = Path(temporary)
                state = {"enabled": False, "expired": False, "phase": None, "spent": 0, "phaseSteps": 0}
                warm = {}
                deadline = time.monotonic() + 90
                original_reset, original_step = environment.reset, environment.step

                def tracked_reset(*args, **kwargs):
                    state["phase"] = kwargs.get("options", {}).get("policyIdentity", {}).get("id")
                    return original_reset(*args, **kwargs)

                def tracked_step(action):
                    result = original_step(action)
                    if state["enabled"]:
                        state["spent"] += 1
                        if state["phase"] == phase:
                            state["phaseSteps"] += 1
                            state["expired"] = state["phaseSteps"] == 2
                    return result

                def checkpoint(row):
                    if row["updates"] == 1 and not state["enabled"]:
                        warm.update(copy.deepcopy(torch.load(directory / "checkpoint.pt", weights_only=True)))
                        state["enabled"] = True

                clock = SimpleNamespace(monotonic=lambda: deadline + 1 if state["expired"] else time.monotonic())
                with mock.patch.object(environment, "reset", tracked_reset), mock.patch.object(environment, "step", tracked_step), \
                     mock.patch("roboproof.motion_learning.td3_train.time", clock), mock.patch("roboproof.motion_learning.imitation.time", clock):
                    partial = train_seed(environment, directory, config["seed"], config, source_identity(), deadline, on_checkpoint=checkpoint)
                saved = torch.load(directory / "checkpoint.pt", weights_only=True)
                self.assertEqual(partial["updates"], 1)
                self.assertEqual(partial["discardedUncommittedSteps"], state["spent"])
                self.assertEqual(partial["collectionSteps"], warm["collectionSteps"] + state["spent"])
                for name in ("measuredAnchor", "model", "learner", "replay", "torchRng", "worldRng", "history", "steps", "episodes"):
                    self.assertTrue(identical(saved[name], warm[name]), name)


if __name__ == "__main__":
    unittest.main()
