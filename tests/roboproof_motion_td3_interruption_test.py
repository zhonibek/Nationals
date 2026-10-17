import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.td3_train import run


def snapshot(directory):
    return {str(filename.relative_to(directory)): hashlib.sha256(filename.read_bytes()).hexdigest()
            for filename in directory.rglob("*") if filename.is_file()}


def arguments(directory):
    return dict(directory=directory, seed=890061, seed_count=1, updates=2, max_steps=512,
                rollout_steps=16, max_seconds=90, warm_start_updates=0, gradient_steps=2,
                demonstration_anchor_weight=0, policy_regularization="none", deadline_pace_loss_weight=1)


class TD3InterruptionTests(unittest.TestCase):
    def test_unknown_step_failure_rejects_resume_without_rewriting_evidence(self):
        original_step = MotionEnvironment.step
        calls = []

        def lose_response(environment, action):
            original_step(environment, action)
            calls.append(1)
            raise RuntimeError("Simulated loss after an actual original-engine step")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(MotionEnvironment, "step", lose_response), self.assertRaisesRegex(RuntimeError, "Simulated loss"):
                run(**arguments(directory))
            self.assertEqual(len(calls), 1)
            run_id = json.loads((directory / "latest.json").read_text())["runId"]
            manifest = json.loads((directory / run_id / "run.json").read_text())
            self.assertEqual(manifest["status"], "interrupted-or-failed")
            self.assertEqual(manifest["models"][0]["collectionSteps"], 0)
            exposures = json.loads((directory / run_id / "seed-890061/exposure.json").read_text())["cases"]
            self.assertEqual(exposures[-1]["update"], 1)
            before = snapshot(directory)
            with patch.object(MotionEnvironment, "reset") as reset, patch.object(MotionEnvironment, "step") as step:
                with self.assertRaisesRegex(ValueError, "exposure.*checkpoint"):
                    run(**arguments(directory), resume=run_id)
                reset.assert_not_called()
                step.assert_not_called()
            self.assertEqual(before, snapshot(directory))

    def test_changed_missing_and_corrupt_exposure_reject_before_queries_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = run(**arguments(directory))
            self.assertEqual(result["status"], "completed")
            exposure_path = directory / result["runId"] / "seed-890061/exposure.json"
            original = exposure_path.read_bytes()
            for changed in (original + b" ", b"\x00" * len(original), None):
                with self.subTest(mutation="missing" if changed is None else "changed"):
                    if changed is None:
                        exposure_path.unlink()
                    else:
                        exposure_path.write_bytes(changed)
                    before = snapshot(directory)
                    with patch.object(MotionEnvironment, "reset") as reset, patch.object(MotionEnvironment, "step") as step:
                        with self.assertRaisesRegex(ValueError, "exposure.*checkpoint"):
                            run(**arguments(directory), resume=result["runId"])
                        reset.assert_not_called()
                        step.assert_not_called()
                    self.assertEqual(before, snapshot(directory))
                    exposure_path.write_bytes(original)

    def test_later_seed_exposure_mismatch_rejects_before_first_seed_is_republished(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            actual = {**arguments(directory), "seed_count": 2}
            result = run(**actual)
            self.assertEqual(result["status"], "completed")
            exposure_path = directory / result["runId"] / "seed-890062/exposure.json"
            exposure_path.write_bytes(exposure_path.read_bytes() + b" ")
            before = snapshot(directory)
            with patch.object(MotionEnvironment, "reset") as reset, patch.object(MotionEnvironment, "step") as step:
                with self.assertRaisesRegex(ValueError, "exposure.*checkpoint"):
                    run(**actual, resume=result["runId"])
                reset.assert_not_called()
                step.assert_not_called()
            self.assertEqual(before, snapshot(directory))


if __name__ == "__main__":
    unittest.main()
