import ast
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_learning import teacher_profiles as teachers


INVALID_NUMBERS = (None, True, False, "0.5", [], {}, complex(1, 0), float("nan"), float("inf"), -float("inf"), 10 ** 1000)


def measurement(pace=0.95, position=0.03, heading=0.04, contact=0.0, effort=4.0, elapsed=5.0, reason="time_limit"):
    return {"pace": pace, "reason": reason, "metrics": {"positionErrorMeters": position,
            "headingErrorRadians": heading, "contactSeconds": contact, "effortProxyVAs": effort, "elapsedSeconds": elapsed}}


def measured_probes(baseline):
    result = []
    for plan in teachers.probe_plans([baseline], teachers.MEASURED_LATE_POSE_PROFILE):
        probe = copy.deepcopy(baseline)
        probe["plan"] = plan
        result.append(probe)
    return result


def raw_observation(reference_seconds=0.0):
    observation = [0.0] * teachers.OBSERVATION_SIZE
    observation[teachers.REFERENCE_SECONDS_INDEX] = reference_seconds
    return observation


def fine_measured_probes(baseline):
    candidates = [measurement(0.85, position=1, heading=1), measurement(1.0, position=1, heading=1),
                  {**copy.deepcopy(baseline), "reason": "time_limit"}]
    result = []
    for plan in teachers.probe_plans(candidates, teachers.FINE_POSE_HEADING_PROFILE):
        result.append({**copy.deepcopy(baseline), "plan": plan})
    return result


class MotionTeacherProfilesTests(unittest.TestCase):
    def test_profile_definitions_are_json_compatible_fresh_and_bounded(self):
        default = teachers.teacher_profile_definition()
        optional = teachers.teacher_profile_definition(teachers.MEASURED_LATE_POSE_PROFILE)
        self.assertEqual(teachers.TEACHER_PROFILES, ("three-pace-v1", "measured-late-pose-v1",
                         "failure-gated-late-pose-v1", "failure-gated-fine-pose-heading-v1",
                         "robust-stop-fine-pose-heading-v1"))
        self.assertEqual(default["id"], "three-pace-v1")
        self.assertTrue(default["default"])
        self.assertIsNone(default["latePose"])
        self.assertFalse(optional["default"])
        self.assertEqual(default["maximumEpisodesPerWorld"], 3)
        self.assertEqual(optional["maximumEpisodesPerWorld"], 8)
        self.assertEqual(optional["maximumProbeEpisodes"], 4)
        self.assertEqual(optional["maximumCombinedEpisodes"], 1)
        self.assertEqual(optional["latePose"]["referenceSecondsIndex"], 25)
        self.assertEqual(optional["latePose"]["phaseStartFraction"], 0.8)
        self.assertEqual(optional["latePose"]["resetDurationKey"], "referenceDurationSeconds")
        self.assertIn("Caller charges every actual episode and step", optional["costAccounting"])
        for definition in (default, optional):
            self.assertEqual(json.loads(json.dumps(definition, allow_nan=False)), definition)
        optional["basePlans"][0]["right"] = 1
        optional["latePose"]["probeOffsets"][0]["right"] = 1
        optional["latePose"]["nonfaultReasons"].clear()
        fresh = teachers.teacher_profile_definition(teachers.MEASURED_LATE_POSE_PROFILE)
        self.assertEqual(fresh["basePlans"][0]["right"], 0)
        self.assertEqual(fresh["latePose"]["probeOffsets"][0]["right"], -0.5)
        self.assertEqual(fresh["latePose"]["nonfaultReasons"], ["success", "time_limit"])

    def test_existing_profile_definitions_remain_exactly_unchanged(self):
        expected = {"three-pace-v1": "6737106297e74efa58cc843bd6cac322f6bc929851f7d2cd01d4d88427799ced",
                    "measured-late-pose-v1": "12c6fd11c4fe42d47bd12841c53b3faf6aff763806389254da28ab54f6408da8",
                    "failure-gated-late-pose-v1": "debbb09bc4ebd8f22546ee6f5aed4c1538fccceaa9dc7af2f549dadad6a1c8f1",
                    "failure-gated-fine-pose-heading-v1": "d5b77e8c9066f6e50bcce61af1018e239caf50ffdfba2db5aafc5aa73f99c1a0"}
        for profile, checksum in expected.items():
            serialized = json.dumps(teachers.teacher_profile_definition(profile), sort_keys=True, separators=(",", ":"))
            self.assertEqual(hashlib.sha256(serialized.encode()).hexdigest(), checksum)

    def test_adaptive_stop_requires_measured_contact_free_pose_and_time_margins(self):
        candidate = measurement(position=0.005, heading=0.005, elapsed=2, reason="success")
        candidate["plan"] = teachers.base_plans()[1]
        self.assertTrue(teachers.robust_teacher_success(candidate, 4))
        for changes in ({"contactSeconds": 1e-12}, {"positionErrorMeters": 0.8 * 0.02032 + 1e-12},
                        {"headingErrorRadians": 0.8 * 0.035 + 1e-12}, {"elapsedSeconds": 3 + 1e-12}):
            changed = copy.deepcopy(candidate)
            changed["metrics"].update(changes)
            self.assertFalse(teachers.robust_teacher_success(changed, 4))
        for reason in ("time_limit", "controller_or_sensor_fault", "stale_policy_action", "stopped"):
            self.assertFalse(teachers.robust_teacher_success({**candidate, "reason": reason}, 4))
        for invalid in INVALID_NUMBERS + (0, -1):
            with self.assertRaises(ValueError):
                teachers.robust_teacher_success(candidate, invalid)
        for missing in ("plan", "metrics", "reason", "pace"):
            changed = copy.deepcopy(candidate)
            del changed[missing]
            with self.assertRaises(ValueError):
                teachers.robust_teacher_success(changed, 4)

    def test_adaptive_profile_keeps_fine_probe_order_and_bounded_fallback(self):
        profile = teachers.ADAPTIVE_FINE_POSE_HEADING_PROFILE
        definition = teachers.teacher_profile_definition(profile)
        previous = teachers.teacher_profile_definition(teachers.FINE_POSE_HEADING_PROFILE)
        self.assertEqual(definition["basePlans"], previous["basePlans"])
        self.assertEqual(definition["latePose"], previous["latePose"])
        self.assertEqual(definition["maximumEpisodesPerWorld"], 10)
        self.assertEqual(definition["earlyStop"]["requiredReason"], "success")
        self.assertEqual(definition["earlyStop"]["contactSeconds"], 0)
        self.assertEqual(json.loads(json.dumps(definition, allow_nan=False)), definition)
        candidates = [measurement(pace) for pace in teachers.TEACHER_PACES]
        self.assertEqual(teachers.probe_plans(candidates, profile),
                         teachers.probe_plans(candidates, teachers.FINE_POSE_HEADING_PROFILE))
        with self.assertRaises(ValueError):
            teachers.probe_plans(candidates[:1], profile)

    def test_failure_gated_profile_is_opt_in_bounded_and_source_described(self):
        profile = teachers.FAILURE_GATED_LATE_POSE_PROFILE
        definition = teachers.teacher_profile_definition(profile)
        self.assertFalse(definition["default"])
        self.assertEqual(definition["maximumEpisodesPerWorld"], 8)
        self.assertEqual(definition["basePlans"], teachers.base_plans())
        self.assertIn("All three measured base paces", definition["latePose"]["probeGate"])
        self.assertIn("exactly zero contact", definition["latePose"]["probeGate"])
        self.assertEqual(json.loads(json.dumps(definition, allow_nan=False)), definition)

    def test_failure_gating_skips_only_measured_success_with_exact_zero_contact(self):
        profile = teachers.FAILURE_GATED_LATE_POSE_PROFILE
        candidates = [measurement(pace) for pace in teachers.TEACHER_PACES]
        for reason, contact, expected_count in (("time_limit", 0.0, 4), ("success", 0.0, 0),
                                                ("success", 1e-12, 4), ("controller_or_sensor_fault", 0.0, 4)):
            candidates[1] = measurement(0.95, reason=reason, contact=contact)
            for permutation in itertools.permutations(candidates):
                plans = teachers.probe_plans(permutation, profile)
                self.assertEqual(len(plans), expected_count)
                if expected_count:
                    self.assertEqual(plans, teachers.probe_plans(permutation, teachers.MEASURED_LATE_POSE_PROFILE))
            self.assertEqual(len(teachers.probe_plans(candidates, teachers.MEASURED_LATE_POSE_PROFILE)), 4)
        faults = [measurement(pace, reason="stopped") for pace in teachers.TEACHER_PACES]
        self.assertEqual(teachers.probe_plans(faults, profile), [])

    def test_failure_gating_never_uses_partial_fallback_or_skips_measurement_validation(self):
        profile = teachers.FAILURE_GATED_LATE_POSE_PROFILE
        success = measurement(0.85, reason="success", contact=0.0)
        for partial in ([], [success], [success, measurement(0.95)]):
            for fallback in (False, True):
                with self.subTest(partial=partial, fallback=fallback), self.assertRaises(ValueError):
                    teachers.probe_plans(partial, profile, allow_fallback=fallback)
        for field, value in itertools.product(teachers.METRIC_NAMES, INVALID_NUMBERS):
            corrupted = measurement(0.95)
            corrupted["metrics"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                teachers.probe_plans([success, corrupted, measurement(1.0)], profile)

    def test_failure_gated_combined_plan_cannot_bypass_success_gate_or_probe_validation(self):
        profile = teachers.FAILURE_GATED_LATE_POSE_PROFILE
        baseline = measurement(reason="success", position=0.01, contact=0.0)
        probes = measured_probes(baseline)
        probes[0]["metrics"]["positionErrorMeters"] = 0.001
        self.assertIsNone(teachers.combined_plan(baseline, probes, profile))
        probes[1]["metrics"]["contactSeconds"] = float("nan")
        with self.assertRaises(ValueError):
            teachers.combined_plan(baseline, probes, profile)

    def test_fine_profile_preserves_bases_and_defines_measured_small_angular_actions(self):
        profile = teachers.FINE_POSE_HEADING_PROFILE
        definition = teachers.teacher_profile_definition(profile)
        self.assertEqual(definition["maximumEpisodesPerWorld"], 10)
        self.assertEqual(definition["maximumProbeEpisodes"], 6)
        self.assertEqual(definition["basePlans"], teachers.base_plans())
        self.assertEqual(json.loads(json.dumps(definition, allow_nan=False)), definition)
        baseline = measurement()
        plans = [probe["plan"] for probe in fine_measured_probes(baseline)]
        self.assertEqual(len(plans), 6)
        for plan in plans:
            self.assertEqual(teachers.validate_plan(plan), plan)
            self.assertEqual(teachers.teacher_action(plan, raw_observation(7.99), 10)[1:], [0.0, 0.0, 0.0])
            action = teachers.teacher_action(plan, raw_observation(8), 10)
            target = teachers.reference_offset_target(action, 0.7)
            if plan["heading"]:
                self.assertEqual(target[:2], [0.0, 0.0])
                self.assertAlmostEqual(abs(target[2]), 0.01)
            else:
                self.assertAlmostEqual(math.hypot(*target[:2]), 0.01)
                self.assertEqual(target[2], 0.0)
        definition["latePose"]["probeOffsets"][0]["right"] = 1
        self.assertEqual(teachers.teacher_profile_definition(profile)["latePose"]["probeOffsets"][0]["right"], -0.125)

    def test_fine_profile_requires_all_valid_measured_bases_and_only_skips_zero_contact_success(self):
        profile = teachers.FINE_POSE_HEADING_PROFILE
        for size in (0, 1, 2):
            with self.assertRaises(ValueError):
                teachers.probe_plans([measurement(pace) for pace in teachers.TEACHER_PACES[:size]], profile,
                                     allow_fallback=True)
        candidates = [measurement(pace) for pace in teachers.TEACHER_PACES]
        self.assertEqual(len(teachers.probe_plans(candidates, profile)), 6)
        candidates[1]["reason"] = "success"
        self.assertEqual(teachers.probe_plans(candidates, profile), [])
        candidates[1]["metrics"]["contactSeconds"] = 1e-12
        self.assertEqual(len(teachers.probe_plans(candidates, profile)), 6)
        candidates[1]["metrics"]["contactSeconds"] = 0
        candidates[2]["metrics"]["headingErrorRadians"] = float("nan")
        with self.assertRaises(ValueError):
            teachers.probe_plans(candidates, profile)

    def test_fine_combination_ranks_both_measured_errors_and_requires_actual_probe_shapes(self):
        baseline = measurement()
        probes = fine_measured_probes(baseline)
        probes[0]["metrics"].update(positionErrorMeters=0.025, headingErrorRadians=0.08)
        probes[1]["metrics"].update(positionErrorMeters=0.027, headingErrorRadians=0.03)
        probes[3]["metrics"].update(positionErrorMeters=0.015, headingErrorRadians=0.045)
        probes[4]["metrics"]["headingErrorRadians"] = 0.005
        probes[5]["metrics"]["headingErrorRadians"] = 0.01
        before = copy.deepcopy(probes)
        plan = teachers.combined_plan(baseline, probes, teachers.FINE_POSE_HEADING_PROFILE)
        self.assertEqual((plan["right"], plan["forward"], plan["heading"]), (0.125, 0.125, -0.1))
        self.assertEqual(teachers.validate_plan(plan), plan)
        self.assertEqual(probes, before)
        for invalid in (probes[:5], probes + [probes[0]], [probes[0]] * 6):
            with self.assertRaises(ValueError):
                teachers.combined_plan(baseline, invalid, teachers.FINE_POSE_HEADING_PROFILE)
        for field in teachers.METRIC_NAMES:
            corrupted = copy.deepcopy(probes)
            corrupted[5]["metrics"][field] = float("nan")
            with self.assertRaises(ValueError):
                teachers.combined_plan(baseline, corrupted, teachers.FINE_POSE_HEADING_PROFILE)

    def test_fine_ties_faults_contacts_and_success_do_not_authorize_unmeasured_corrections(self):
        profile = teachers.FINE_POSE_HEADING_PROFILE
        baseline = measurement()
        probes = fine_measured_probes(baseline)
        self.assertIsNone(teachers.combined_plan(baseline, probes, profile))
        for reason, contact in (("stopped", 0), ("time_limit", 0.001)):
            altered = copy.deepcopy(probes)
            altered[4]["reason"] = reason
            altered[4]["metrics"].update(headingErrorRadians=0, positionErrorMeters=0, contactSeconds=contact)
            self.assertIsNone(teachers.combined_plan(baseline, altered, profile))
        for probe in probes[4:]:
            probe["metrics"]["headingErrorRadians"] = 0.01
        self.assertEqual(teachers.combined_plan(baseline, probes, profile)["heading"], -0.1)
        baseline["reason"] = "success"
        self.assertIsNone(teachers.combined_plan(baseline, probes, profile))
        probes[0]["pace"] = 1
        with self.assertRaises(ValueError):
            teachers.combined_plan(baseline, probes, profile)

    def test_fine_plans_and_counts_cannot_expand_legacy_actions_or_query_caps(self):
        plans = [probe["plan"] for probe in fine_measured_probes(measurement())]
        for plan in plans:
            for field in ("right", "forward", "heading", "phaseStartFraction"):
                for invalid in INVALID_NUMBERS:
                    with self.assertRaises(ValueError):
                        teachers.validate_plan({**plan, field: invalid})
        for changes in ({"heading": 0.11}, {"right": 0.126}, {"heading": -0.1}, {"phaseStartFraction": 0.79}):
            with self.assertRaises(ValueError):
                teachers.validate_plan({**plans[0], **changes})
        with self.assertRaises(ValueError):
            teachers.validate_plan({**teachers.base_plans()[0], "heading": 0.1})
        for count in range(11):
            self.assertEqual(teachers.validate_episode_count(count, teachers.FINE_POSE_HEADING_PROFILE), count)
        for invalid in (-1, 11, True, 10.0):
            with self.assertRaises(ValueError):
                teachers.validate_episode_count(invalid, teachers.FINE_POSE_HEADING_PROFILE)
        with self.assertRaises(ValueError):
            teachers.validate_episode_count(9, teachers.FAILURE_GATED_LATE_POSE_PROFILE)

    def test_default_and_optional_base_actions_equal_original_three_paces(self):
        default = teachers.base_plans()
        self.assertEqual(default, teachers.base_plans(teachers.MEASURED_LATE_POSE_PROFILE))
        self.assertEqual([plan["pace"] for plan in default], [0.85, 0.95, 1.0])
        for plan, reference_seconds in itertools.product(default, (0, 7.99, 8, 10)):
            expected = [2 * (plan["pace"] - 0.25) / 0.75 - 1, 0, 0, 0]
            self.assertEqual(teachers.teacher_action(plan, raw_observation(reference_seconds), 10), expected)
        baseline = measurement()
        self.assertEqual(teachers.probe_plans([baseline]), [])
        self.assertIsNone(teachers.combined_plan(baseline, measured_probes(baseline)))

    def test_pace_selection_prioritizes_contact_then_normalized_squared_pose(self):
        touching = measurement(1.0, position=0.001, heading=0.001, contact=0.1, effort=0, elapsed=0, reason="success")
        no_contact = measurement(0.85, position=0.03, heading=0.04)
        self.assertEqual(teachers.select_probe_pace([touching, no_contact]), 0.85)
        better_position = measurement(0.85, position=0.01, heading=0.06, effort=0, elapsed=0, reason="success")
        better_pose = measurement(0.95, position=0.02, heading=0.01, effort=100, elapsed=100)
        self.assertLess(teachers.terminal_pose_error_squared(better_pose["metrics"]),
                        teachers.terminal_pose_error_squared(better_position["metrics"]))
        self.assertEqual(teachers.select_probe_pace([better_position, better_pose]), 0.95)
        normalized = measurement(position=teachers.POSITION_ERROR_SCALE, heading=teachers.HEADING_ERROR_SCALE)
        self.assertEqual(teachers.terminal_pose_error_squared(normalized["metrics"]), 2.0)

    def test_pace_ties_use_effort_time_then_pace_independent_of_order(self):
        candidates = [measurement(0.85, effort=5), measurement(0.95, effort=4, elapsed=4),
                      measurement(1.0, effort=4, elapsed=3)]
        for permutation in itertools.permutations(candidates):
            self.assertEqual(teachers.select_probe_pace(permutation), 1.0)
        for candidate in candidates:
            candidate["metrics"]["effortProxyVAs"] = 4
            candidate["metrics"]["elapsedSeconds"] = 3
        for permutation in itertools.permutations(candidates):
            self.assertEqual(teachers.select_probe_pace(permutation), 0.85)

    def test_faults_skip_and_only_explicit_validated_empty_input_can_fallback(self):
        self.assertIsNone(teachers.select_probe_pace([]))
        self.assertEqual(teachers.select_probe_pace([], allow_fallback=True), 0.95)
        self.assertEqual(teachers.probe_plans([], teachers.MEASURED_LATE_POSE_PROFILE), [])
        fallback = teachers.probe_plans([], teachers.MEASURED_LATE_POSE_PROFILE, allow_fallback=True)
        self.assertEqual(len(fallback), 4)
        self.assertTrue(all(plan["pace"] == 0.95 for plan in fallback))
        for reason in teachers.FAULT_REASONS:
            fault = measurement(reason=reason, position=0, heading=0, effort=0, elapsed=0)
            self.assertIsNone(teachers.select_probe_pace([fault], allow_fallback=True))
            self.assertEqual(teachers.probe_plans([fault], teachers.MEASURED_LATE_POSE_PROFILE, allow_fallback=True), [])
            valid = measurement(0.85)
            self.assertEqual(teachers.select_probe_pace([fault, valid]), 0.85)
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                teachers.select_probe_pace([], allow_fallback=value)
        with self.assertRaises(ValueError):
            teachers.select_probe_pace([{}], allow_fallback=True)

    def test_exact_four_signed_single_axis_plans_use_selected_pace(self):
        baseline = measurement(0.85)
        plans = teachers.probe_plans([baseline], teachers.MEASURED_LATE_POSE_PROFILE)
        self.assertEqual([(plan["right"], plan["forward"]) for plan in plans],
                         [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)])
        for plan in plans:
            self.assertEqual(plan["pace"], 0.85)
            self.assertEqual(plan["heading"], 0)
            self.assertEqual(plan["phaseStartFraction"], 0.8)
            self.assertEqual(teachers.validate_plan(plan), plan)
            target = teachers.reference_offset_target(teachers.teacher_action(plan, raw_observation(10), 10), 0.7)
            self.assertAlmostEqual(math.hypot(target[0], target[1]), 0.04)
        self.assertEqual(json.loads(json.dumps(plans, allow_nan=False)), plans)

    def test_raw_public_reference_time_not_wall_time_pace_or_pose_gates_offsets(self):
        plan = teachers.probe_plans([measurement()], teachers.MEASURED_LATE_POSE_PROFILE)[1]
        before = raw_observation(math.nextafter(8.0, 0.0))
        before[0:3] = [1000, -1000, math.pi]
        before[18] = -1000
        before[19:25] = [500, 600, 1.4, 20, 30, 40]
        before[26] = 1.0
        before[33] = 1000
        self.assertEqual(teachers.teacher_action(plan, before, 10)[1:], [0, 0, 0])
        boundary = list(before)
        boundary[25] = 8.0
        self.assertEqual(teachers.teacher_action(plan, boundary, 10)[1:], [0.5, 0, 0])
        boundary[0:3] = [-500, 500, -math.pi]
        boundary[18] = 1000
        boundary[21] = -1.4
        boundary[26] = 0.25
        self.assertEqual(teachers.teacher_action(plan, boundary, 10)[1:], [0.5, 0, 0])
        self.assertEqual(teachers.teacher_action(plan, boundary, 20)[1:], [0, 0, 0])
        for seconds in (8.5, 9, 10):
            self.assertEqual(teachers.teacher_action(plan, raw_observation(seconds), 10)[1:], [0.5, 0, 0])
        self.assertEqual(teachers.teacher_action(plan, raw_observation(), 0)[1:], [0.5, 0, 0])

    def test_original_current_reference_body_rotation_and_signs(self):
        pace = teachers.base_plans()[1]["pace"]
        pace_unit = 2 * (pace - 0.25) / 0.75 - 1
        cases = (([pace_unit, 0.5, 0, 0], 0, [0.04, 0, 0]),
                 ([pace_unit, -0.5, 0, 0], 0, [-0.04, 0, 0]),
                 ([pace_unit, 0, 0.5, 0], 0, [0, 0.04, 0]),
                 ([pace_unit, 0.5, 0, 0], math.pi / 2, [0, -0.04, 0]),
                 ([pace_unit, 0, 0.5, 0], math.pi / 2, [0.04, 0, 0]),
                 ([pace_unit, 0.5, -0.5, 0], math.pi / 2, [-0.04, -0.04, 0]),
                 ([pace_unit, 0, 0, 0.5], 1.2, [0, 0, 0.05]))
        for action, heading, expected in cases:
            with self.subTest(action=action, heading=heading):
                actual = teachers.reference_offset_target(action, heading)
                for actual_value, expected_value in zip(actual, expected):
                    self.assertAlmostEqual(actual_value, expected_value, places=14)
        self.assertNotEqual(teachers.reference_offset_target(cases[0][0], 0),
                            teachers.reference_offset_target(cases[0][0], math.pi / 2))

    def test_combined_selects_axes_independently_and_does_not_synthesize_results(self):
        baseline = measurement()
        probes = measured_probes(baseline)
        for probe, error in zip(probes, (0.024, 0.015, 0.018, 0.022)):
            probe["metrics"]["positionErrorMeters"] = error
        combined = teachers.combined_plan(baseline, probes, teachers.MEASURED_LATE_POSE_PROFILE)
        self.assertEqual(combined["right"], 0.5)
        self.assertEqual(combined["forward"], -0.5)
        self.assertEqual(combined["heading"], 0)
        self.assertEqual(combined["pace"], baseline["pace"])
        self.assertEqual(set(combined), set(teachers.PLAN_FIELDS))
        self.assertNotIn("reason", combined)
        self.assertNotIn("reward", combined)
        self.assertEqual(teachers.teacher_action(combined, raw_observation(7.99), 10)[1:], [0, 0, 0])
        self.assertEqual(teachers.teacher_action(combined, raw_observation(8), 10)[1:], [0.5, -0.5, 0])
        self.assertAlmostEqual(math.hypot(*teachers.reference_offset_target(
            teachers.teacher_action(combined, raw_observation(8), 10), 0.7)[:2]), math.sqrt(2) * 0.04)

    def test_combined_requires_exact_same_contact_even_if_probe_has_less_contact(self):
        baseline = measurement(contact=0.1)
        probes = measured_probes(baseline)
        probes[0]["metrics"].update(positionErrorMeters=0.001, contactSeconds=0)
        probes[1]["metrics"].update(positionErrorMeters=0.002, contactSeconds=0.2)
        probes[2]["metrics"].update(positionErrorMeters=0.003, contactSeconds=math.nextafter(0.1, 0.0))
        self.assertIsNone(teachers.combined_plan(baseline, probes, teachers.MEASURED_LATE_POSE_PROFILE))
        probes[3]["metrics"]["positionErrorMeters"] = 0.025
        combined = teachers.combined_plan(baseline, probes, teachers.MEASURED_LATE_POSE_PROFILE)
        self.assertEqual((combined["right"], combined["forward"]), (0, 0.5))

    def test_combined_ties_keep_baseline_zero_then_negative_sign(self):
        baseline = measurement()
        probes = measured_probes(baseline)
        for probe in probes:
            probe["metrics"]["effortProxyVAs"] = 0
            probe["metrics"]["elapsedSeconds"] = 0
            probe["metrics"]["headingErrorRadians"] = 0
        self.assertIsNone(teachers.combined_plan(baseline, probes, teachers.MEASURED_LATE_POSE_PROFILE))
        probes[0]["metrics"]["positionErrorMeters"] = 0.02
        probes[1]["metrics"]["positionErrorMeters"] = 0.02
        probes[0]["metrics"]["effortProxyVAs"] = 100
        for permutation in itertools.permutations(probes):
            combined = teachers.combined_plan(baseline, permutation, teachers.MEASURED_LATE_POSE_PROFILE)
            self.assertEqual((combined["right"], combined["forward"]), (-0.5, 0))

    def test_combined_fault_probes_are_ignored_and_fault_baselines_skip(self):
        baseline = measurement()
        probes = measured_probes(baseline)
        for probe in probes:
            probe["metrics"]["positionErrorMeters"] = 0.001
            probe["reason"] = "controller_or_sensor_fault"
        self.assertIsNone(teachers.combined_plan(baseline, probes, teachers.MEASURED_LATE_POSE_PROFILE))
        probes[0]["reason"] = "time_limit"
        combined = teachers.combined_plan(baseline, probes, teachers.MEASURED_LATE_POSE_PROFILE)
        self.assertEqual((combined["right"], combined["forward"]), (-0.5, 0))
        baseline["reason"] = "stale_policy_action"
        self.assertIsNone(teachers.combined_plan(baseline, probes, teachers.MEASURED_LATE_POSE_PROFILE))

    def test_all_helpers_are_deterministic_without_input_mutation(self):
        candidates = [measurement(pace) for pace in teachers.TEACHER_PACES]
        candidates_before = copy.deepcopy(candidates)
        baseline = candidates[0]
        probes = measured_probes(baseline)
        probes[0]["metrics"]["positionErrorMeters"] = 0.01
        probes[2]["metrics"]["positionErrorMeters"] = 0.02
        probes_before = copy.deepcopy(probes)
        observation = raw_observation(8)
        observation_before = list(observation)
        first = teachers.combined_plan(baseline, probes, teachers.MEASURED_LATE_POSE_PROFILE)
        action = teachers.teacher_action(first, observation, 10)
        self.assertEqual(first, teachers.combined_plan(baseline, probes, teachers.MEASURED_LATE_POSE_PROFILE))
        self.assertEqual(action, teachers.teacher_action(first, observation, 10))
        self.assertEqual(candidates, candidates_before)
        self.assertEqual(probes, probes_before)
        self.assertEqual(observation, observation_before)
        normalized = teachers.validate_plan(first)
        normalized["right"] = 0
        self.assertEqual(first["right"], -0.5)
        copied_action = teachers.validate_action(action)
        copied_action[1] = 0
        self.assertEqual(action[1], -0.5)

    def test_actor_actions_ignore_extra_training_and_privileged_measurement_fields(self):
        baseline = measurement()
        original_plans = teachers.probe_plans([baseline], teachers.MEASURED_LATE_POSE_PROFILE)
        baseline.update(truePose={"x": 1000}, hiddenSimulatorState={"theta": 100},
                        evaluatorWorld={"goal": -1000}, corpus={"success": True}, canonicalReward=12345)
        baseline["metrics"]["worldGoalX"] = -1000
        self.assertEqual(original_plans, teachers.probe_plans([baseline], teachers.MEASURED_LATE_POSE_PROFILE))
        for plan in original_plans:
            self.assertEqual(set(plan), set(teachers.PLAN_FIELDS))

    def test_invalid_profiles_reject_for_all_profile_aware_helpers(self):
        baseline = measurement()
        probes = measured_probes(baseline)
        for value in (None, True, 1, [], {}, "unknown", "measured-late-pose"):
            for function in (lambda: teachers.teacher_profile_definition(value), lambda: teachers.base_plans(value),
                             lambda: teachers.probe_plans([baseline], value),
                             lambda: teachers.combined_plan(baseline, probes, value),
                             lambda: teachers.validate_episode_count(0, value)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    function()

    def test_invalid_candidate_shapes_reasons_and_base_paces_reject(self):
        for candidates in (None, {}, "", iter([]), [measurement()] * 4, [measurement()] * 2, [None], [[]], [{}]):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                teachers.select_probe_pace(candidates)
        for field in ("pace", "reason", "metrics"):
            candidate = measurement()
            del candidate[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                teachers.select_probe_pace([candidate])
        for reason in (None, True, [], {}, "running", "failed", "Success"):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                teachers.select_probe_pace([measurement(reason=reason)])
        for pace in INVALID_NUMBERS + (-1, 0.25, 0.5, 0.849, 1.01):
            with self.subTest(pace=pace), self.assertRaises(ValueError):
                teachers.select_probe_pace([measurement(pace=pace)])

    def test_every_required_metric_rejects_missing_invalid_negative_or_overflow_values(self):
        for field in teachers.METRIC_NAMES:
            for value in INVALID_NUMBERS + (-0.001,):
                candidate = measurement()
                candidate["metrics"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    teachers.select_probe_pace([candidate], allow_fallback=True)
            candidate = measurement()
            del candidate["metrics"][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                teachers.terminal_pose_error_squared(candidate["metrics"])
        for metrics in (None, [], "", 1):
            candidate = measurement()
            candidate["metrics"] = metrics
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                teachers.select_probe_pace([candidate])
        for field in ("positionErrorMeters", "headingErrorRadians"):
            candidate = measurement()
            candidate["metrics"][field] = 1e308
            with self.subTest(field=field), self.assertRaises(ValueError):
                teachers.select_probe_pace([candidate])

    def test_fault_metrics_do_not_bypass_finite_measurement_validation(self):
        for field, value in itertools.product(teachers.METRIC_NAMES, INVALID_NUMBERS):
            candidate = measurement(reason="controller_or_sensor_fault")
            candidate["metrics"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                teachers.select_probe_pace([candidate], allow_fallback=True)

    def test_base_candidates_with_attached_plans_require_base_kind_and_matching_pace(self):
        baseline = measurement()
        baseline["plan"] = teachers.base_plans()[1]
        self.assertEqual(teachers.select_probe_pace([baseline]), 0.95)
        baseline["plan"] = teachers.base_plans()[0]
        with self.assertRaises(ValueError):
            teachers.select_probe_pace([baseline])
        baseline["plan"] = teachers.probe_plans([measurement()], teachers.MEASURED_LATE_POSE_PROFILE)[0]
        with self.assertRaises(ValueError):
            teachers.select_probe_pace([baseline])

    def test_actions_reject_bad_shapes_units_and_bool_without_clamping(self):
        for action in (None, {}, "0000", [0] * 3, [0] * 5, iter([0] * 4)):
            with self.subTest(action=action), self.assertRaises(ValueError):
                teachers.validate_action(action)
        for index, value in itertools.product(range(4), INVALID_NUMBERS + (-1.001, 1.001)):
            action = [0] * 4
            action[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                teachers.validate_action(action)
        self.assertEqual(teachers.validate_action((-1, 1, -1, 1)), [-1.0, 1.0, -1.0, 1.0])

    def test_raw_observations_reject_wrong_shapes_nonfinite_and_bool_in_every_slot(self):
        plan = teachers.base_plans()[0]
        for observation in (None, {}, "0" * 34, [0] * 33, [0] * 35, [[0] * 34], iter([0] * 34)):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                teachers.teacher_action(plan, observation, 10)
        for index, value in itertools.product(range(34), INVALID_NUMBERS):
            observation = raw_observation()
            observation[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                teachers.teacher_action(plan, observation, 10)
        for seconds in (-0.001, 10.001):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                teachers.teacher_action(plan, raw_observation(seconds), 10)
        for duration in INVALID_NUMBERS + (-0.001,):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                teachers.teacher_action(plan, raw_observation(), duration)

    def test_plans_reject_wrong_shapes_units_signs_and_all_changed_phases(self):
        probe = teachers.probe_plans([measurement()], teachers.MEASURED_LATE_POSE_PROFILE)[0]
        for plan in (None, [], {}, {**probe, "truePose": 0}, {field: value for field, value in probe.items() if field != "heading"}):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                teachers.validate_plan(plan)
        for field in ("pace", "right", "forward", "heading", "phaseStartFraction"):
            for value in INVALID_NUMBERS:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    teachers.validate_plan({**probe, field: value})
        for phase in (-0.01, 0, 0.799, 0.801, 1, 1.01):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                teachers.validate_plan({**probe, "phaseStartFraction": phase})
        for changes in ({"right": 0.5}, {"right": -0.25}, {"right": -0.501}, {"forward": 0.5},
                        {"heading": 0.1}, {"pace": 0.9}, {"kind": None}, {"kind": []}, {"kind": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                teachers.validate_plan({**probe, **changes})
        base = teachers.base_plans()[0]
        for changes in ({"right": 0.5}, {"forward": -0.5}, {"phaseStartFraction": 0.8}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                teachers.validate_plan({**base, **changes})
        for changes in ({"right": 0, "forward": 0}, {"right": 0.1}, {"forward": -0.1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                teachers.validate_plan({**probe, "kind": "combined", **changes})

    def test_combined_requires_four_distinct_measured_same_pace_probe_plans(self):
        baseline = measurement()
        probes = measured_probes(baseline)
        for invalid in (None, {}, "", probes[:3], probes + [probes[0]], [probes[0]] * 4,
                        [None] + probes[1:], [{key: value for key, value in probes[0].items() if key != "plan"}] + probes[1:]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                teachers.combined_plan(baseline, invalid, teachers.MEASURED_LATE_POSE_PROFILE)
        for change in ("measurement-pace", "plan-pace", "base-plan", "phase", "metric", "reason"):
            altered = copy.deepcopy(probes)
            if change == "measurement-pace":
                altered[0]["pace"] = 0.85
            elif change == "plan-pace":
                altered[0]["plan"]["pace"] = 0.85
            elif change == "base-plan":
                altered[0]["plan"] = teachers.base_plans()[1]
            elif change == "phase":
                altered[0]["plan"]["phaseStartFraction"] = 0.7
            elif change == "metric":
                altered[0]["metrics"]["positionErrorMeters"] = float("nan")
            else:
                altered[0]["reason"] = "running"
            with self.subTest(change=change), self.assertRaises(ValueError):
                teachers.combined_plan(baseline, altered, teachers.MEASURED_LATE_POSE_PROFILE)

    def test_reference_target_rejects_invalid_heading_and_action(self):
        for heading in INVALID_NUMBERS:
            with self.subTest(heading=heading), self.assertRaises(ValueError):
                teachers.reference_offset_target([0, 0.5, 0, 0], heading)
        with self.assertRaises(ValueError):
            teachers.reference_offset_target([0, 0.5, 0, float("nan")], 0)

    def test_attempt_budget_includes_three_base_four_probe_and_at_most_one_combined(self):
        baseline = measurement()
        plans = teachers.base_plans(teachers.MEASURED_LATE_POSE_PROFILE)
        plans += teachers.probe_plans([baseline], teachers.MEASURED_LATE_POSE_PROFILE)
        probes = measured_probes(baseline)
        probes[0]["metrics"]["positionErrorMeters"] = 0.01
        plans.append(teachers.combined_plan(baseline, probes, teachers.MEASURED_LATE_POSE_PROFILE))
        self.assertEqual(len(plans), 8)
        for count in range(9):
            self.assertEqual(teachers.validate_episode_count(count, teachers.MEASURED_LATE_POSE_PROFILE), count)
        self.assertEqual(teachers.validate_episode_count(3), 3)
        with self.assertRaises(ValueError):
            teachers.validate_episode_count(4)
        for count in INVALID_NUMBERS + (-1, 9, 8.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                teachers.validate_episode_count(count, teachers.MEASURED_LATE_POSE_PROFILE)

    def test_module_has_only_standard_library_math_import_and_no_execution_dependencies(self):
        source = Path(teachers.__file__).read_text(encoding="utf-8")
        parsed = ast.parse(source)
        imported = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module)
        self.assertEqual(imported, ["math"])
        forbidden = ("roboproof.motion_env", "roboproof.motion_learning.train", "torch")
        script = f"import json, sys; sys.path.insert(0, {str(ROOT)!r}); " \
                 "from roboproof.motion_learning import teacher_profiles; " \
                 f"print(json.dumps([name for name in {forbidden!r} if name in sys.modules]))"
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=ROOT,
                                capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
