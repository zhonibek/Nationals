import math
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
from roboproof.motion_learning.imitation import collect_demonstration, imitation_update, MeasuredAnchor, demonstration_samples, teacher_margins, select_demonstration
from roboproof.motion_learning.model import MotionActorCritic, update
from roboproof.motion_learning.train import compatible_config, source_identity, train_seed, run


class MotionImitationTests(unittest.TestCase):
    def test_teacher_margins_prioritize_robust_success_without_relaxing_pass_rules(self):
        def candidate(pace, position, heading, elapsed, effort, contact=0, reason="success"):
            metrics = {"positionErrorMeters": position, "headingErrorRadians": heading, "elapsedSeconds": elapsed,
                       "effortProxyVAs": effort, "contactSeconds": contact}
            return {"pace": pace, "reason": reason, "metrics": metrics, **teacher_margins(metrics, {"deadlineSeconds": 6})}
        fragile = candidate(0.85, 0.019, 0.034, 5.9, 1)
        robust = candidate(0.95, 0.01, 0.02, 4.5, 4)
        robust_more_effort = candidate(1, 0.008, 0.018, 4, 5)
        choices = [fragile, robust, robust_more_effort]
        self.assertLess(fragile["marginScore"], 1)
        self.assertGreater(robust["marginScore"], 1)
        self.assertIs(select_demonstration(choices), fragile)
        self.assertIs(select_demonstration(choices, "pose-deadline-margin-v2"), robust)
        better_fragile = candidate(0.95, 0.018, 0.031, 5.5, 4)
        self.assertLess(better_fragile["marginScore"], 1)
        self.assertIs(select_demonstration([fragile, better_fragile], "pose-deadline-margin-v2"), better_fragile)
        touching = candidate(1, 0.004, 0.004, 3, 0.5, contact=0.1)
        self.assertIs(select_demonstration([fragile, touching], "pose-deadline-margin-v2"), fragile)
        failed = candidate(1, 0.004, 0.004, 3, 0.5, reason="time_limit")
        self.assertIsNone(select_demonstration([failed], "pose-deadline-margin-v2"))
        self.assertIs(select_demonstration([fragile, failed], "pose-deadline-margin-v2"), fragile)
        for metrics, task in (({}, {"deadlineSeconds": 6}), (fragile["metrics"], {}),
                              ({**fragile["metrics"], "positionErrorMeters": True}, {"deadlineSeconds": 6}),
                              ({**fragile["metrics"], "headingErrorRadians": float("nan")}, {"deadlineSeconds": 6})):
            with self.assertRaises(ValueError):
                teacher_margins(metrics, task)
        with self.assertRaises(ValueError):
            select_demonstration(choices, "weaker-pass-thresholds")
        with self.assertRaises(ValueError):
            collect_demonstration(None, None, {}, time.monotonic() + 1, True)
        with self.assertRaises(ValueError):
            run(teacher_selection="pose-deadline-margin-v2")

    def test_anchor_samples_endpoints_stays_bounded_and_rejects_bad_restoration(self):
        rows = [{"modelInput": [index / 1000] * 34, "teacherAction": [0.9, 0, 0, 0]} for index in range(1000)]
        selected = demonstration_samples(rows)
        self.assertEqual(len(selected), 16)
        self.assertEqual(selected[0], rows[0])
        self.assertEqual(selected[-1], rows[-1])
        self.assertEqual(demonstration_samples(rows[:1]), rows[:1])
        self.assertEqual(demonstration_samples([]), [])
        anchor = MeasuredAnchor(34)
        anchor.append(rows[:400])
        anchor.append(rows[400:])
        self.assertEqual(len(anchor.features), 512)
        self.assertAlmostEqual(float(anchor.features[0, 0]), 0.488, places=6)
        restored = MeasuredAnchor(34)
        restored.restore(anchor.state())
        self.assertTrue(torch.equal(anchor.features, restored.features))
        anchor.features[0, 0] = 1
        self.assertNotEqual(float(anchor.features[0, 0]), float(restored.features[0, 0]))
        for state in (None, {**restored.state(), "schemaVersion": 2},
                      {**restored.state(), "features": torch.zeros(513, 34)},
                      {**restored.state(), "targets": torch.full((512, 4), float("nan"))},
                      {**restored.state(), "targets": torch.ones(512, 4) * 2}):
            with self.assertRaises(ValueError):
                restored.restore(state)
        with self.assertRaises(ValueError):
            MeasuredAnchor(35).restore(anchor.state())

    def test_anchor_has_real_actor_gradients_and_is_explicitly_bounded(self):
        torch.set_num_threads(1)
        model = MotionActorCritic("broad-pace-v1")
        anchor = MeasuredAnchor(34)
        with self.assertRaises(ValueError):
            anchor.loss(model)
        anchor.append([{"modelInput": [0] * 34, "teacherAction": [0.95, 0, 0, 0]}] * 48)
        before = model.actor.weight.detach().clone()
        optimizer = torch.optim.Adam([*model.body.parameters(), *model.actor.parameters()], lr=0.003)
        initial_loss = float(anchor.loss(model).detach())
        for _ in range(8):
            optimizer.zero_grad()
            anchor.loss(model).backward()
            optimizer.step()
        self.assertFalse(torch.equal(before, model.actor.weight))
        self.assertLess(float(anchor.loss(model).detach()), initial_loss)
        self.assertIsNone(model.critic.weight.grad)
        for weight in (-1, float("nan"), True, 101):
            with self.assertRaises(ValueError):
                update(model, optimizer, [], anchor_weight=weight, measured_anchor=anchor)
        with self.assertRaises(ValueError):
            update(model, optimizer, [], anchor_weight=1)
        with self.assertRaises(ValueError):
            run(demonstration_anchor_weight=1)
        with self.assertRaises(ValueError):
            run(demonstration_anchor_weight=float("inf"), warm_start_updates=1)

    def test_demonstration_uses_original_simulator_and_keeps_every_attempt(self):
        torch.set_num_threads(1)
        model = MotionActorCritic("broad-pace-v1", "history-context-v1")
        world = {"seed": 1123, "task": {"start": {"xIn": 0, "yIn": 0, "headingDeg": 0},
                 "goal": {"xIn": 0, "yIn": 12, "headingDeg": 0}, "deadlineSeconds": 4}, "configuration": {}}
        with MotionEnvironment() as environment:
            result = collect_demonstration(environment, model, world, time.monotonic() + 60, 1000)
        self.assertFalse(result["interrupted"])
        self.assertEqual(len(result["attempts"]), 3)
        self.assertTrue(all(attempt["reason"] == "success" for attempt in result["attempts"]))
        self.assertGreater(result["observedSteps"], len(result["rows"]))
        self.assertEqual(result["reason"], "success")
        self.assertTrue(all(len(row["modelInput"]) == 144 and len(row["teacherAction"]) == 4 for row in result["rows"]))
        best = min(result["attempts"], key=lambda attempt: (attempt["metrics"]["contactSeconds"],
                   attempt["metrics"]["effortProxyVAs"], attempt["metrics"]["elapsedSeconds"]))
        self.assertEqual(result["selectedPace"], best["pace"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        before = model.actor.weight.detach().clone()
        diagnostic = imitation_update(model, optimizer, result["rows"], epochs=2)
        self.assertEqual(diagnostic["phase"], "measured-controller-imitation")
        self.assertTrue(math.isfinite(diagnostic["meanLoss"]))
        self.assertFalse(torch.equal(before, model.actor.weight))

    def test_failed_teacher_worlds_are_retained_but_not_imitated(self):
        model = MotionActorCritic("broad-pace-v1")
        world = {"seed": 1124, "task": {"start": {"xIn": 0, "yIn": 0, "headingDeg": 0},
                 "goal": {"xIn": 0, "yIn": 24, "headingDeg": 0}, "deadlineSeconds": 0.01}, "configuration": {}}
        with MotionEnvironment() as environment:
            result = collect_demonstration(environment, model, world, time.monotonic() + 60, 100)
        self.assertEqual(len(result["attempts"]), 3)
        self.assertEqual(result["rows"], [])
        self.assertIsNone(result["selectedPace"])
        self.assertTrue(all(attempt["reason"] == "time_limit" for attempt in result["attempts"]))

    def test_imitation_and_ppo_resume_exactly_with_all_query_steps_in_the_budget(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 2, "maxSteps": 2000, "rolloutSteps": 16, "seed": 1125, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "history-context-v1", "warmStartUpdates": 1,
                  "objective": "finite-total-return-pose-effort-v3"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 1125, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 1:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 1125, config, sources, time.monotonic() + 90,
                           on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 1125, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertEqual(complete["collectionSteps"], resumed["collectionSteps"])
            self.assertGreater(complete["collectionSteps"], complete["steps"])
            self.assertEqual([row["phase"] for row in complete["history"]], ["measured-controller-imitation", "ppo"])
            self.assertEqual(complete["history"][0]["teacherQueries"], 3)
            self.assertTrue(complete["actorWeightsChanged"])
            self.assertTrue((directory / "complete/exposure.json").is_file())
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 1125, {**config, "warmStartUpdates": 0}, sources,
                           time.monotonic() + 90, resume=True)
        self.assertFalse(compatible_config(config, {**config, "warmStartUpdates": 0}))

    def test_anchor_survives_exact_ppo_resume_and_configuration_changes_fail_closed(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 3, "maxSteps": 2000, "rolloutSteps": 16, "seed": 1126, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "history-context-v1", "warmStartUpdates": 1,
                  "demonstrationAnchorWeight": 10, "objective": "finite-total-return-pose-effort-v3",
                  "teacherSelection": "pose-deadline-margin-v2"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 1126, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 1126, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 1126, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(all(row["demonstrationAnchorRows"] == 16 for row in complete["history"]))
            self.assertGreater(complete["history"][-1]["meanMeasuredAnchorLoss"], 0)
            teacher_world = complete["history"][0]["worlds"][0]
            self.assertEqual(teacher_world["teacherSelection"], "pose-deadline-margin-v2")
            self.assertTrue(math.isfinite(teacher_world["teacherSelectedMarginScore"]))
            checkpoint = torch.load(directory / "complete/checkpoint.pt", weights_only=True)
            anchor = MeasuredAnchor(144)
            anchor.restore(checkpoint["measuredAnchor"])
            self.assertEqual(len(anchor.features), 16)
            self.assertLess((directory / "complete/checkpoint.pt").stat().st_size, 4 * 1024 * 1024)
            for changed in ({"demonstrationAnchorWeight": 1}, {"demonstrationAnchorDefinition": {"capacity": 256}},
                            {"teacherSelection": "minimum-effort-v1"}):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    train_seed(environment, directory / "resumed", 1126, {**config, **changed}, sources, time.monotonic() + 90, resume=True)

    def test_interrupted_teacher_and_ppo_work_is_journaled_and_never_refunds_the_budget(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for warm_updates in (0, 1):
            with self.subTest(warm_updates=warm_updates), tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
                directory = Path(temporary)
                config = {"updates": 2, "maxSteps": 2, "rolloutSteps": 16, "seed": 1130, "seedCount": 1,
                          "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                          "warmStartUpdates": warm_updates, "objective": "finite-total-return-pose-effort-v3"}
                sources = source_identity()
                interrupted = train_seed(environment, directory / "resumed", 1130, config, sources, time.monotonic() + 90)
                self.assertEqual(interrupted["updates"], 0)
                self.assertEqual(interrupted["collectionSteps"], 2)
                self.assertEqual(interrupted["discardedCollectionStepsTotal"], 2)
                self.assertEqual(interrupted["discardedUncommittedSteps"], 2)
                exposure = json.loads((directory / "resumed/exposure.json").read_text())
                self.assertEqual(len(exposure["cases"]), 1)
                self.assertIn("teacher and PPO", exposure["scope"])
                unchanged = train_seed(environment, directory / "resumed", 1130, config, sources, time.monotonic() + 90, resume=True)
                self.assertEqual(unchanged["collectionSteps"], 2)
                self.assertEqual(unchanged["updates"], 0)
                self.assertEqual(unchanged["discardedCollectionStepsTotal"], 2)
                extended = {**config, "maxSteps": 2000}
                resumed = train_seed(environment, directory / "resumed", 1130, extended, sources, time.monotonic() + 90, resume=True)
                complete = train_seed(environment, directory / "complete", 1130, extended, sources, time.monotonic() + 90)
                self.assertEqual(resumed["policySha256"], complete["policySha256"])
                self.assertEqual(resumed["history"], complete["history"])
                self.assertEqual(resumed["collectionSteps"], complete["collectionSteps"] + 2)
                self.assertEqual(resumed["discardedCollectionStepsTotal"], 2)


if __name__ == "__main__":
    unittest.main()
