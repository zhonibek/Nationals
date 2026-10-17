import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.expert_data import FEATURE_SIZE, accepted_rows, collect, iter_demonstrations, read_json, validate_worlds


def packet():
    return {"schemaVersion": 1, "cases": [
        {"id": f"expert-data-fixture-{index}", "group": f"expert-data-family-{index}", "seed": 880010 + index,
         "task": {"start": {"xIn": 0.125, "yIn": -0.25, "headingDeg": 0.5},
                  "goal": {"xIn": 0.375, "yIn": 12.25 * (index + 1), "headingDeg": 1}, "deadlineSeconds": 10},
         "configuration": {}} for index in range(3)]}


def save(filename, value):
    filename.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class ExpertDataTests(unittest.TestCase):
    def test_actual_complete_trajectories_and_clean_slice_resume_are_identical(self):
        observed = []

        class CountedEnvironment(MotionEnvironment):
            def step(self, action):
                result = super().step(action)
                observed.append(1)
                return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worlds = root / "input.json"
            save(worlds, packet())
            with mock.patch("roboproof.motion_learning.expert_data.MotionEnvironment", CountedEnvironment):
                complete = collect(worlds, root / "complete", max_queries=5000, max_seconds=90)
                full_count = len(observed)
                prefix = collect(worlds, root / "resumed", max_queries=5000, max_seconds=90, worlds_per_call=1)
                self.assertEqual(prefix["status"], "budget-stopped")
                self.assertEqual(len(prefix["records"]), 1)
                self.assertIsNone(prefix["pendingWorld"])
                resumed = collect(worlds, root / "resumed", max_queries=5000, max_seconds=90, resume=True)
            self.assertEqual(complete["status"], "completed")
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(complete["actualQueries"], full_count)
            self.assertEqual(resumed["actualQueries"], len(observed) - full_count)
            self.assertEqual(complete["records"], resumed["records"])
            full_rows = list(iter_demonstrations(root / "complete"))
            resumed_rows = list(iter_demonstrations(root / "resumed"))
            self.assertEqual(full_rows, resumed_rows)
            self.assertEqual(len(full_rows), 3)
            self.assertTrue(all(row["rows"] and row["featureSize"] == FEATURE_SIZE for row in full_rows))
            self.assertEqual(complete["actualQueries"], sum(row["measurement"]["observedSteps"] for row in full_rows))
            self.assertFalse(complete["gradientsPerformed"])
            self.assertFalse(complete["learnedImprovementVerified"])
            with self.assertRaisesRegex(ValueError, "Never overwrite"):
                collect(worlds, root / "complete", max_queries=5000, max_seconds=90)
            damaged = root / "complete/world-0002.json"
            original = damaged.read_text()
            damaged.write_text(original + " ")
            with self.assertRaisesRegex(ValueError, "hash/world mismatch"):
                next(iter_demonstrations(root / "complete"))
            changed = json.loads(original)
            changed["featureSize"] -= 1
            save(damaged, changed)
            declaration, _ = read_json(root / "complete/dataset.json")
            declaration["records"][2]["sha256"] = hashlib.sha256(damaged.read_bytes()).hexdigest()
            save(root / "complete/dataset.json", declaration)
            with self.assertRaisesRegex(ValueError, "hash/world mismatch"):
                next(iter_demonstrations(root / "complete"))

    def test_partial_teacher_cost_is_retained_without_any_admitted_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worlds = root / "input.json"
            save(worlds, packet())
            saved = collect(worlds, root / "partial", max_queries=1, max_seconds=90)
            self.assertEqual(saved["status"], "budget-stopped")
            self.assertEqual(saved["records"], [])
            self.assertEqual(saved["actualQueries"], 1)
            self.assertEqual(saved["discardedQueries"], 1)
            self.assertEqual(saved["discardedAttempts"][0]["observedSteps"], 1)
            exposure, _ = read_json(root / "partial/exposure.json")
            self.assertEqual(len(exposure["cases"]), 1)
            with self.assertRaisesRegex(ValueError, "Complete exact"):
                next(iter_demonstrations(root / "partial"))
            with self.assertRaisesRegex(ValueError, "continuation"):
                collect(worlds, root / "partial", max_queries=2, max_seconds=90, resume=True)

    def test_unknown_transport_interruption_cannot_silently_resume(self):
        original_step = MotionEnvironment.step

        def lost_response(environment, action):
            original_step(environment, action)
            raise RuntimeError("Synthetic lost-response fixture; no physics claim")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worlds = root / "input.json"
            save(worlds, packet())
            with mock.patch.object(MotionEnvironment, "step", lost_response), self.assertRaisesRegex(RuntimeError, "lost-response"):
                collect(worlds, root / "unknown", max_queries=5000, max_seconds=90)
            saved, _ = read_json(root / "unknown/dataset.json")
            self.assertEqual(saved["status"], "interrupted-or-failed")
            self.assertTrue(saved["unknownInterruptedCost"])
            self.assertEqual(saved["pendingWorld"], 0)
            with self.assertRaisesRegex(ValueError, "unknown interrupted cost"):
                collect(worlds, root / "unknown", max_queries=5000, max_seconds=90, resume=True)

    def test_only_real_zero_contact_success_rows_and_valid_public_worlds_are_admitted(self):
        plan = {"kind": "base", "pace": 0.85, "right": 0, "forward": 0, "heading": 0, "phaseStartFraction": 0}
        rows = [{"modelInput": [0.0] * FEATURE_SIZE, "teacherAction": [0.6, 0, 0, 0]}]
        measured = {"rows": rows, "reason": "success", "selectedPlan": plan,
                    "attempts": [{"reason": "success", "plan": plan, "metrics": {"contactSeconds": 0}}]}
        self.assertEqual(accepted_rows(measured, FEATURE_SIZE, 10), (rows, "success-zero-contact"))
        contact = copy.deepcopy(measured)
        contact["attempts"][0]["metrics"]["contactSeconds"] = 1
        self.assertEqual(accepted_rows(contact, FEATURE_SIZE, 10), ([], "selected-contact-success-excluded"))
        contact["rows"] = []
        self.assertEqual(accepted_rows(contact, FEATURE_SIZE, 10), ([], "selected-contact-success-excluded"))
        with self.assertRaisesRegex(ValueError, "Complete selected"):
            accepted_rows({**measured, "rows": []}, FEATURE_SIZE, 10)
        invalid = copy.deepcopy(measured)
        invalid["rows"][0]["teacherAction"][0] = 1.1
        with self.assertRaises(ValueError):
            accepted_rows(invalid, FEATURE_SIZE, 10)
        for change in ("duplicate-seed", "bad-pose", "boolean-seed"):
            invalid_worlds = packet()
            if change == "duplicate-seed":
                invalid_worlds["cases"][1]["seed"] = invalid_worlds["cases"][0]["seed"]
            elif change == "bad-pose":
                invalid_worlds["cases"][1]["task"]["goal"]["xIn"] = 61
            else:
                invalid_worlds["cases"][1]["seed"] = True
            with self.assertRaises(ValueError):
                validate_worlds(invalid_worlds)
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "nonfinite.json"
            filename.write_text('{"value":NaN}')
            with self.assertRaisesRegex(ValueError, "Finite"):
                read_json(filename)


if __name__ == "__main__":
    unittest.main()
