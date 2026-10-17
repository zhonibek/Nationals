import copy
import time
import unittest

import torch

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.imitation import collect_demonstration
from roboproof.motion_learning.model import MotionActorCritic
from roboproof.motion_learning.td3_train import compatible_config, configuration
from roboproof.motion_learning.teacher_profiles import (DEFAULT_TEACHER_PROFILE, MEASURED_LATE_POSE_PROFILE,
                                                       FAILURE_GATED_LATE_POSE_PROFILE, FINE_POSE_HEADING_PROFILE,
                                                       ADAPTIVE_FINE_POSE_HEADING_PROFILE)


WORLD = {"seed": 1123, "task": {"start": {"xIn": 0, "yIn": 0, "headingDeg": 0},
         "goal": {"xIn": 0, "yIn": 12, "headingDeg": 0}, "deadlineSeconds": 4}, "configuration": {}}


class PublicModel:
    def __init__(self):
        self.seen = []

    def reset_policy_state(self, duration):
        self.duration = duration

    def encode_observation(self, observation):
        self.seen.append(list(observation))
        return torch.tensor(observation, dtype=torch.float32)


class CountedEnvironment:
    def __init__(self, environment):
        self.environment = environment
        self.calls = 0
        self.resets = 0

    def reset(self, **kwargs):
        self.resets += 1
        return self.environment.reset(**kwargs)

    def step(self, action):
        self.calls += 1
        return self.environment.step(action)

    def report(self):
        return self.environment.report()


class MeasuredFixtureEnvironment:
    def __init__(self):
        self.calls = 0
        self.episodes = []
        self.current = None

    def reset(self, **kwargs):
        self.current = []
        self.episodes.append(self.current)
        return [0.0] * 34, {"referenceDurationSeconds": 1.0}

    def step(self, action):
        self.calls += 1
        self.current.append(list(action))
        observation = [0.0] * 34
        observation[25] = min(1.0, len(self.current) * 0.5)
        return observation, 0.125, len(self.current) == 3, False, {}

    def report(self):
        if len(self.current) != 3:
            raise AssertionError("Terminal measurements only")
        action = self.current[-1]
        offsets = tuple(action[1:3])
        if offsets == (0.0, 0.0):
            pace = 0.25 + (action[0] + 1) * 0.75 / 2
            position = 0.04 if abs(pace - 0.95) < 0.000001 else 0.05
            effort = 3
        else:
            position = {(-0.5, 0.0): 0.08, (0.5, 0.0): 0.01,
                        (0.0, -0.5): 0.012, (0.0, 0.5): 0.06, (0.5, -0.5): 0.004}[offsets]
            effort = 1 if offsets == (0.5, -0.5) else 2
        return {"reason": "success" if position < 0.02032 else "time_limit", "metrics": {
            "positionErrorMeters": position, "headingErrorRadians": 0.003, "elapsedSeconds": 0.5,
            "effortProxyVAs": effort, "contactSeconds": 0}}


class SuccessfulBaseFixtureEnvironment(MeasuredFixtureEnvironment):
    def __init__(self, contact=0.0):
        super().__init__()
        self.contact = contact

    def report(self):
        result = super().report()
        if self.current[-1][1:3] == [0.0, 0.0]:
            result["reason"] = "success"
            result["metrics"]["positionErrorMeters"] = 0.005
            result["metrics"]["contactSeconds"] = self.contact
        return result


class FineMeasuredFixtureEnvironment(MeasuredFixtureEnvironment):
    def report(self):
        if len(self.current) != 3:
            raise AssertionError("Only actual completed episodes may provide metrics")
        action = self.current[-1]
        offsets = tuple(action[1:])
        errors = {(0.0, 0.0, 0.0): (0.032, 0.045), (-0.125, 0.0, 0.0): (0.024, 0.045),
                  (0.125, 0.0, 0.0): (0.04, 0.06), (0.0, -0.125, 0.0): (0.045, 0.045),
                  (0.0, 0.125, 0.0): (0.03, 0.07), (0.0, 0.0, -0.1): (0.032, 0.07),
                  (0.0, 0.0, 0.1): (0.032, 0.02), (-0.125, 0.0, 0.1): (0.009, 0.005)}
        position, heading = errors[offsets]
        return {"reason": "success" if position < 0.02032 and heading < 0.035 else "time_limit", "metrics": {
            "positionErrorMeters": position, "headingErrorRadians": heading, "elapsedSeconds": 0.5,
            "effortProxyVAs": 1 if offsets == (-0.125, 0.0, 0.1) else 3, "contactSeconds": 0}}


class MotionTeacherCollectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_adaptive_stops_after_first_robust_base_without_inventing_remaining_attempts(self):
        environment = SuccessfulBaseFixtureEnvironment()
        model = PublicModel()
        result = collect_demonstration(environment, model, WORLD, time.monotonic() + 10, 1000,
            "pose-deadline-margin-v2", ADAPTIVE_FINE_POSE_HEADING_PROFILE)
        self.assertTrue(result["searchStoppedOnRobustSuccess"])
        self.assertFalse(result["interrupted"])
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(result["observedSteps"], 3)
        self.assertEqual(environment.calls, 3)
        self.assertEqual(result["selectedPlan"]["kind"], "base")
        self.assertEqual(len(model.seen), 3)

    def test_adaptive_nonrobust_contact_and_failed_search_keep_actual_fallback(self):
        class ContactBaseEnvironment(FineMeasuredFixtureEnvironment):
            def report(self):
                measured = super().report()
                if self.current[-1][1:] == [0.0, 0.0, 0.0]:
                    measured["reason"] = "success"
                    measured["metrics"].update(positionErrorMeters=0.005, headingErrorRadians=0.005,
                                               contactSeconds=1e-12)
                return measured

        for environment in (ContactBaseEnvironment(), FineMeasuredFixtureEnvironment()):
            result = collect_demonstration(environment, PublicModel(), WORLD, time.monotonic() + 10, 1000,
                "pose-deadline-margin-v2", ADAPTIVE_FINE_POSE_HEADING_PROFILE)
            self.assertFalse(result["searchStoppedOnRobustSuccess"])
            self.assertEqual(result["observedSteps"], environment.calls)
            self.assertEqual(len(result["attempts"]), 9 if isinstance(environment, ContactBaseEnvironment) else 10)

    def test_adaptive_native_queries_use_original_engine_and_public_features(self):
        outputs = []
        for profile in (FINE_POSE_HEADING_PROFILE, ADAPTIVE_FINE_POSE_HEADING_PROFILE):
            model = MotionActorCritic("broad-pace-v1", "deadline-context-v1", network_width=256)
            with MotionEnvironment() as underlying:
                environment = CountedEnvironment(underlying)
                result = collect_demonstration(environment, model, WORLD, time.monotonic() + 30, 2000,
                    "pose-deadline-margin-v2", profile)
                self.assertFalse(result["interrupted"])
                self.assertEqual(result["observedSteps"], environment.calls)
                self.assertEqual(len(result["attempts"]), environment.resets)
                self.assertEqual(result["reason"], "success")
                self.assertTrue(all(len(row["modelInput"]) == len(model.scales) for row in result["rows"]))
                outputs.append(result)
        self.assertLessEqual(outputs[1]["observedSteps"], outputs[0]["observedSteps"])

    def test_adaptive_stops_on_measured_probe_and_charges_partial_attempts(self):
        class RobustProbeEnvironment(FineMeasuredFixtureEnvironment):
            def report(self):
                measured = super().report()
                if self.current[-1][1:] == [-0.125, 0.0, 0.0]:
                    measured["reason"] = "success"
                    measured["metrics"].update(positionErrorMeters=0.008, headingErrorRadians=0.008)
                return measured

        environment = RobustProbeEnvironment()
        result = collect_demonstration(environment, PublicModel(), WORLD, time.monotonic() + 10, 1000,
            "pose-deadline-margin-v2", ADAPTIVE_FINE_POSE_HEADING_PROFILE)
        self.assertTrue(result["searchStoppedOnRobustSuccess"])
        self.assertEqual(result["selectedPlan"]["kind"], "fine-right-negative")
        self.assertEqual(len(result["attempts"]), 4)
        self.assertEqual(result["observedSteps"], environment.calls)
        self.assertEqual(environment.calls, 12)
        for budget in (0, 1, 8, 9, 10, 11):
            environment = RobustProbeEnvironment()
            result = collect_demonstration(environment, PublicModel(), WORLD, time.monotonic() + 10, budget,
                "pose-deadline-margin-v2", ADAPTIVE_FINE_POSE_HEADING_PROFILE)
            self.assertTrue(result["interrupted"])
            self.assertEqual(result["observedSteps"], environment.calls)
            self.assertEqual(environment.calls, budget)
            self.assertNotIn("rows", result)

    def test_measured_combination_has_its_own_episode_and_only_public_actor_inputs(self):
        environment = MeasuredFixtureEnvironment()
        model = PublicModel()
        result = collect_demonstration(environment, model, WORLD, time.monotonic() + 10, 1000,
            "pose-deadline-margin-v2", MEASURED_LATE_POSE_PROFILE)
        self.assertFalse(result["interrupted"])
        self.assertEqual(len(result["attempts"]), 8)
        self.assertEqual(result["observedSteps"], environment.calls)
        self.assertEqual(environment.calls, 24)
        self.assertEqual(result["selectedPlan"]["kind"], "combined")
        self.assertEqual((result["selectedPlan"]["right"], result["selectedPlan"]["forward"]), (0.5, -0.5))
        self.assertEqual(result["reason"], "success")
        self.assertEqual(result["canonicalReward"], 0.375)
        for episode in environment.episodes[3:]:
            self.assertEqual(episode[0][1:], [0.0, 0.0, 0.0])
            self.assertEqual(episode[1][1:], [0.0, 0.0, 0.0])
            self.assertTrue(any(value != 0 for value in episode[2][1:3]))
        self.assertEqual(len(model.seen), environment.calls)
        self.assertTrue(all(len(observation) == 34 for observation in model.seen))
        self.assertTrue(all(set(row) == {"modelInput", "teacherAction"} for row in result["rows"]))
        self.assertEqual(result["rows"][-1]["teacherAction"][1:], [0.5, -0.5, 0.0])

    def test_partial_probe_work_is_charged_and_cannot_publish_selected_rows(self):
        for budget in (0, 1, 5, 9, 11, 23):
            with self.subTest(budget=budget):
                environment = MeasuredFixtureEnvironment()
                result = collect_demonstration(environment, PublicModel(), WORLD, time.monotonic() + 10, budget,
                    "pose-deadline-margin-v2", MEASURED_LATE_POSE_PROFILE)
                self.assertTrue(result["interrupted"])
                self.assertEqual(result["observedSteps"], environment.calls)
                self.assertEqual(environment.calls, budget)
                self.assertNotIn("rows", result)
                self.assertNotIn("selectedPlan", result)

    def test_failure_gated_success_retains_all_base_measurements_without_extra_queries(self):
        environment = SuccessfulBaseFixtureEnvironment()
        model = PublicModel()
        result = collect_demonstration(environment, model, WORLD, time.monotonic() + 10, 9,
            "pose-deadline-margin-v2", FAILURE_GATED_LATE_POSE_PROFILE)
        self.assertFalse(result["interrupted"])
        self.assertEqual(len(result["attempts"]), 3)
        self.assertEqual(result["observedSteps"], 9)
        self.assertEqual(environment.calls, 9)
        self.assertEqual(len(model.seen), 9)
        self.assertEqual(result["selectedPlan"]["kind"], "base")
        self.assertEqual(result["teacherProfile"], FAILURE_GATED_LATE_POSE_PROFILE)
        self.assertTrue(all(row["teacherAction"][1:] == [0.0, 0.0, 0.0] for row in result["rows"]))

    def test_failure_gated_unsuccessful_bases_keep_every_probe_and_measured_combined_episode(self):
        for contact in (None, 1e-12):
            environment = MeasuredFixtureEnvironment() if contact is None else SuccessfulBaseFixtureEnvironment(contact)
            result = collect_demonstration(environment, PublicModel(), WORLD, time.monotonic() + 10, 1000,
                "pose-deadline-margin-v2", FAILURE_GATED_LATE_POSE_PROFILE)
            self.assertFalse(result["interrupted"])
            self.assertEqual(len(result["attempts"]), 8 if contact is None else 7)
            self.assertEqual(result["observedSteps"], environment.calls)
            self.assertEqual(environment.calls, 24 if contact is None else 21)
            self.assertTrue(all(len(episode) == 3 for episode in environment.episodes))

    def test_failure_gated_partial_base_or_probe_work_is_still_charged(self):
        for budget in (0, 1, 5, 9, 11, 23):
            environment = MeasuredFixtureEnvironment()
            result = collect_demonstration(environment, PublicModel(), WORLD, time.monotonic() + 10, budget,
                "pose-deadline-margin-v2", FAILURE_GATED_LATE_POSE_PROFILE)
            self.assertTrue(result["interrupted"])
            self.assertEqual(result["observedSteps"], environment.calls)
            self.assertEqual(environment.calls, budget)
            self.assertNotIn("rows", result)
            self.assertNotIn("selectedPlan", result)

    def test_fine_position_and_heading_combination_requires_tenth_actual_episode(self):
        environment = FineMeasuredFixtureEnvironment()
        model = PublicModel()
        result = collect_demonstration(environment, model, WORLD, time.monotonic() + 10, 1000,
            "pose-deadline-margin-v2", FINE_POSE_HEADING_PROFILE)
        self.assertFalse(result["interrupted"])
        self.assertEqual(len(result["attempts"]), 10)
        self.assertEqual(result["observedSteps"], 30)
        self.assertEqual(environment.calls, 30)
        self.assertEqual(result["selectedPlan"]["kind"], "fine-combined")
        self.assertEqual(result["rows"][-1]["teacherAction"][1:], [-0.125, 0.0, 0.1])
        self.assertEqual(result["reason"], "success")
        self.assertEqual(len(model.seen), environment.calls)
        self.assertTrue(all(len(observation) == 34 for observation in model.seen))
        for episode in environment.episodes[3:]:
            self.assertEqual(episode[0][1:], [0.0, 0.0, 0.0])
            self.assertEqual(episode[1][1:], [0.0, 0.0, 0.0])

    def test_fine_partial_probe_and_combination_work_is_charged_without_labels(self):
        for budget in (0, 1, 8, 9, 10, 26, 27, 29):
            environment = FineMeasuredFixtureEnvironment()
            result = collect_demonstration(environment, PublicModel(), WORLD, time.monotonic() + 10, budget,
                "pose-deadline-margin-v2", FINE_POSE_HEADING_PROFILE)
            self.assertTrue(result["interrupted"])
            self.assertEqual(result["observedSteps"], budget)
            self.assertEqual(environment.calls, budget)
            self.assertNotIn("rows", result)
            self.assertNotIn("selectedPlan", result)

    def test_default_native_demonstrations_are_identical_and_probes_use_original_engine(self):
        outputs = []
        for profile in (DEFAULT_TEACHER_PROFILE, DEFAULT_TEACHER_PROFILE, MEASURED_LATE_POSE_PROFILE,
                        FAILURE_GATED_LATE_POSE_PROFILE, FINE_POSE_HEADING_PROFILE):
            model = MotionActorCritic("broad-pace-v1", "deadline-context-v1")
            with MotionEnvironment() as underlying:
                environment = CountedEnvironment(underlying)
                result = collect_demonstration(environment, model, WORLD, time.monotonic() + 60, 2000,
                    "pose-deadline-margin-v2", profile)
                self.assertEqual(result["observedSteps"], environment.calls)
                self.assertEqual(len(result["attempts"]), environment.resets)
            self.assertFalse(result["interrupted"])
            self.assertTrue(all(len(row["modelInput"]) == 145 for row in result["rows"]))
            outputs.append(result)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(outputs[0]["attempts"]), 3)
        self.assertTrue(all(row["teacherAction"][1:] == [0, 0, 0] for row in outputs[0]["rows"]))
        self.assertGreaterEqual(len(outputs[2]["attempts"]), 7)
        self.assertLessEqual(len(outputs[2]["attempts"]), 8)
        self.assertTrue(any(attempt["plan"]["right"] or attempt["plan"]["forward"] for attempt in outputs[2]["attempts"]))
        self.assertEqual(len(outputs[3]["attempts"]), 3)
        for field in ("rows", "selectedPace", "selectedMarginScore", "canonicalReward", "reason", "observedSteps"):
            self.assertEqual(outputs[0][field], outputs[3][field])
            self.assertEqual(outputs[0][field], outputs[4][field])

    def test_teacher_profile_and_definition_are_source_bound_resume_configuration(self):
        arguments = dict(seed=210001, seed_count=1, updates=4, max_steps=3000, rollout_steps=16,
            warm_start_updates=1, feature_transform="deadline-context-v1", objective="finite-total-return-settle-margin-v4",
            curriculum="mixed-full-reach-v5", gradient_steps=4, replay_capacity=64, exploration_std=0.05,
            demonstration_anchor_weight=1, policy_regularization="public-deadline-residual-v1", deadline_pace_loss_weight=4)
        original = configuration(**arguments)
        measured = configuration(**arguments, teacher_profile=MEASURED_LATE_POSE_PROFILE)
        self.assertFalse(compatible_config(original, measured))
        self.assertFalse(compatible_config(measured, original))
        changed = copy.deepcopy(measured)
        changed["teacherProfileDefinition"]["latePose"]["probeMagnitude"] = 0.1
        self.assertFalse(compatible_config(measured, changed))
        self.assertEqual(measured["teacherProfileDefinition"]["maximumEpisodesPerWorld"], 8)
        gated = configuration(**arguments, teacher_profile=FAILURE_GATED_LATE_POSE_PROFILE)
        self.assertFalse(compatible_config(gated, measured))
        self.assertFalse(compatible_config(measured, gated))
        self.assertFalse(compatible_config(original, gated))
        changed_gate = copy.deepcopy(gated)
        changed_gate["teacherProfileDefinition"]["latePose"]["probeGate"] = "Always probe"
        self.assertFalse(compatible_config(gated, changed_gate))
        fine = configuration(**arguments, teacher_profile=FINE_POSE_HEADING_PROFILE)
        self.assertFalse(compatible_config(fine, gated))
        self.assertFalse(compatible_config(gated, fine))
        changed_fine = copy.deepcopy(fine)
        changed_fine["teacherProfileDefinition"]["latePose"]["probeMagnitudes"]["headingUnits"] = 0.2
        self.assertFalse(compatible_config(fine, changed_fine))
        with self.assertRaises(ValueError):
            configuration(**arguments, teacher_profile="privileged-actor-input")


if __name__ == "__main__":
    unittest.main()
