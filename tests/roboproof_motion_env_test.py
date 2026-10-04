import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment


class MotionEnvironmentTests(unittest.TestCase):
    def test_real_bridge_reset_step_report_and_exact_node_replay(self):
        task = {"start": {"xIn": 0, "yIn": 0, "headingDeg": 0},
                "goal": {"xIn": 0, "yIn": 24, "headingDeg": 0}, "deadlineSeconds": 0.03}
        with MotionEnvironment() as environment:
            observation, info = environment.reset(seed=123, task=task)
            self.assertEqual(len(observation), 34)
            self.assertTrue(info["identity"]["canonicalSimulator"])
            self.assertFalse(environment.contract["motionTrainingImplemented"])
            observation, reward, terminated, truncated, info = environment.step([1, 0, 0, 0])
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["physicsTicks"], 3)
            report = environment.report()
            self.assertEqual(len(report["transitions"]), 3)
            self.assertEqual(report["totalReward"], reward)
            with self.assertRaisesRegex(RuntimeError, "Reset"):
                environment.step([1, 0, 0, 0])
            completed = subprocess.run(
                [shutil.which("node"), "-e", "let data='';process.stdin.on('data',part=>data+=part);"
                 "process.stdin.on('end',()=>{require('./roboproof/motion').replay(JSON.parse(data));"
                 "process.stdout.write('verified');});"],
                cwd=str(ROOT), input=json.dumps(report), capture_output=True, text=True, check=True,
                timeout=30,
            )
            self.assertEqual(completed.stdout, "verified")
        self.assertIsNotNone(environment._process.poll())

    def test_invalid_actions_stop_and_require_reset(self):
        with MotionEnvironment() as environment:
            for action in ([float("nan"), 0, 0, 0], [2, 0, 0, 0], [1, 0], None, [True, 0, 0, 0]):
                environment.reset()
                with self.assertRaisesRegex(ValueError, "four finite"):
                    environment.step(action)
                self.assertEqual(environment.report()["reason"], "stopped")
                with self.assertRaisesRegex(RuntimeError, "Reset"):
                    environment.step([1, 0, 0, 0])

    def test_resets_are_isolated_and_close_is_idempotent(self):
        environment = MotionEnvironment()
        try:
            first, _ = environment.reset(seed=456, configuration={"massKg": [5, 8]})
            environment.step([1, 1, -1, 1])
            second, _ = environment.reset(seed=456, configuration={"massKg": [5, 8]})
            self.assertEqual(first, second)
            self.assertEqual(environment.report()["ticks"], 0)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                environment.reset(configuration={"batteryVoltage": 10})
            with self.assertRaisesRegex(RuntimeError, "Reset"):
                environment.step([1, 0, 0, 0])
        finally:
            environment.close()
            environment.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            environment.report()

    def test_explicit_limits_and_missing_repository_fail_without_starting(self):
        with self.assertRaises(ValueError):
            MotionEnvironment(timeout_seconds=0)
        with self.assertRaisesRegex(ValueError, "Repository"):
            MotionEnvironment(repository=ROOT / "does-not-exist")


if __name__ == "__main__":
    unittest.main()
