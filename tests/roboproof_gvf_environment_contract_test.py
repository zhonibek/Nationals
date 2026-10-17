import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment


def contract():
    return {**MotionEnvironment.GVF_TAGS, "protocolVersion": 1, "episodeSchemaVersion": 2, "controlMode": "gvf-residual",
            "policyContractVersion": 2, "policyContractName": "gvf-residual",
            "actionSize": 3, "observationSize": 34, "observationFields": copy.deepcopy(MotionEnvironment.GVF_FIELDS),
            "actionNames": ["dRight", "dForward", "dHeading"], "actionBounds": [-1, 1],
            "zeroAction": [0, 0, 0], "fixedDt": 0.01, "policyIntervalTicks": 5, "actionTimeoutTicks": 10,
            "policyLimits": copy.deepcopy(MotionEnvironment.GVF_LIMITS),
            "identity": {"canonicalSimulator": True, "sourceHash": "original-runner-fixture"}}


def transition():
    observation = {**MotionEnvironment.GVF_TAGS,
                   **{name: 0 if width == 1 else [0] * width for name, width in MotionEnvironment.GVF_FIELDS}}
    return {**MotionEnvironment.GVF_TAGS, "observation": observation, "vector": [0] * 34,
            "identity": contract()["identity"], "options": {"controlMode": "gvf-residual"},
            "reward": 2, "terminated": False, "truncated": False, "info": {"physicsTicks": 5}}


class GvfEnvironmentContractTests(unittest.TestCase):
    def environment(self, mode="gvf-residual", response=None):
        process = Mock()
        process.poll.return_value = None
        process.stdin = io.BytesIO()
        process.stdout = io.BytesIO()
        spawn = patch("roboproof.motion_env.subprocess.Popen", return_value=process).start()
        reader = patch("roboproof.motion_env.threading.Thread").start()
        request = patch.object(MotionEnvironment, "_request", return_value=response or contract()).start()
        self.addCleanup(patch.stopall)
        environment = MotionEnvironment(repository=ROOT, node_executable="fixture-node", control_mode=mode)
        self.addCleanup(environment.close)
        return environment, process, spawn, reader, request

    def test_fixed_selector_only_appends_new_mode_cli_and_never_supplies_environment(self):
        environment, _, spawn, _, _ = self.environment()
        self.assertEqual(spawn.call_args.args[0], ["fixture-node", str(ROOT / "roboproof/motion-bridge.js"),
                                                 "--contract=gvf-residual"])
        self.assertNotIn("env", spawn.call_args.kwargs)
        self.assertFalse(spawn.call_args.kwargs["shell"])
        self.assertEqual(environment.contract["actionSize"], 3)

    def test_legacy_spawning_reset_info_and_four_actions_remain_unchanged(self):
        environment, _, spawn, _, request = self.environment("policy", {"actionSize": 4, "observationSize": 34})
        self.assertEqual(len(spawn.call_args.args[0]), 2)
        result = {"vector": [7] * 34, "identity": {"legacy": True}, "options": {"policyIntervalTicks": 3},
                  "referenceDurationSeconds": 12, "reward": 1, "terminated": False, "truncated": False, "info": {}}
        request.return_value = result
        _, info = environment.reset(options={"policyIntervalTicks": 3})
        self.assertEqual(info["referenceDurationSeconds"], 12)
        self.assertNotIn("contractName", info)
        environment.step([1, 0, 0, 0])
        request.assert_called_with("step", action=[1, 0, 0, 0])

    def test_unknown_control_mode_rejects_before_spawn(self):
        with patch("roboproof.motion_env.subprocess.Popen") as spawn:
            for mode in ("gvf", "", None, True):
                with self.assertRaisesRegex(ValueError, "control_mode"):
                    MotionEnvironment(control_mode=mode)
            spawn.assert_not_called()

    def test_equal_length_never_substitutes_for_contract_identity_schema_or_cadence(self):
        for field, bad in [("contractName", "policy"), ("contractVersion", 1), ("observationSchema", "legacy34"),
                           ("policyContractVersion", 1), ("policyContractName", "policy"), ("controlMode", "policy"),
                           ("actionSize", 4), ("actionNames", ["pace", "right", "forward"]),
                           ("observationFields", [["oldSensors", 34]]), ("zeroAction", [1, 0, 0]), ("zeroAction", [False, 0, 0]),
                           ("episodeSchemaVersion", 1),
                           ("policyIntervalTicks", 4), ("actionTimeoutTicks", 11), ("policyLimits", {"minimumPace": 0.25})]:
            with self.subTest(field=field):
                response = contract()
                response[field] = bad
                with self.assertRaisesRegex(RuntimeError, "identity/schema/action"):
                    self.environment(response=response)

    def test_gvf_reset_returns_tagged_sensor_vector_without_reference_duration(self):
        environment, _, _, _, request = self.environment()
        request.return_value = transition()
        options = {"policyIdentity": {"kind": "untrained", "id": "gvf-fixture"}}
        vector, info = environment.reset(seed=13, options=options)
        self.assertEqual(vector, [0] * 34)
        self.assertNotIn("referenceDurationSeconds", info)
        self.assertEqual(info["observationSchema"], "gvf-residual-sensor34-v2")
        request.assert_called_with("reset", seed=13, configuration={}, options=options)
        self.assertFalse(environment._needs_reset)
        environment.step([0, 0, 0])
        request.assert_called_with("step", action=[0, 0, 0])

    def test_bad_actions_stop_and_require_reset_even_if_stop_fails(self):
        environment, _, _, _, request = self.environment()
        for action in ([1, 0, 0, 0], [True, 0, 0], [float("nan"), 0, 0], [2, 0, 0], [0, 0], None):
            request.return_value = transition()
            environment.reset()
            with self.assertRaisesRegex(ValueError, "three finite"):
                environment.step(action)
            request.assert_called_with("stop")
            with self.assertRaisesRegex(RuntimeError, "Reset"):
                environment.step([0, 0, 0])
        environment.reset()
        request.side_effect = ValueError("stop rejected")
        with self.assertRaisesRegex(ValueError, "stop rejected"):
            environment.step([True, 0, 0])
        self.assertTrue(environment._needs_reset)

    def test_invalid_reset_vector_or_rejected_contract_switch_latches_reset(self):
        environment, _, _, _, request = self.environment()
        for alter in (lambda row: row.pop("contractVersion"),
                      lambda row: row["observation"].update(contractName="policy"),
                      lambda row: row["observation"].update(truthPose=[0, 0, 0]),
                      lambda row: row["observation"].update(previousAction=[1, 0, 0, 0]),
                      lambda row: row["observation"].update(pose=[True, 0, 0]),
                      lambda row: row.update(vector=[False] * 34),
                      lambda row: row.update(vector=[1] * 34)):
            response = transition()
            alter(response)
            request.return_value = response
            with self.assertRaises(ValueError):
                environment.reset()
            self.assertTrue(environment._needs_reset)
        request.side_effect = ValueError("Process-fixed controlMode mismatch")
        with self.assertRaisesRegex(ValueError, "Process-fixed"):
            environment.reset(options={"controlMode": "policy"})
        self.assertTrue(environment._needs_reset)

    def test_terminal_and_bridge_rejections_require_reset_and_wrong_step_tags_stop(self):
        environment, _, _, _, request = self.environment()
        for terminal_field in ("terminated", "truncated"):
            request.return_value = transition()
            environment.reset()
            request.return_value[terminal_field] = True
            self.assertTrue(environment.step([0, 0, 0])[2 if terminal_field == "terminated" else 3])
            with self.assertRaisesRegex(RuntimeError, "Reset"):
                environment.step([0, 0, 0])
        request.return_value = transition()
        environment.reset()
        request.side_effect = [ValueError("Bridge rejected action"), {}]
        with self.assertRaisesRegex(ValueError, "Bridge rejected"):
            environment.step([0, 0, 0])
        request.assert_called_with("stop")
        self.assertTrue(environment._needs_reset)

        request.side_effect = None
        request.return_value = transition()
        environment.reset()
        request.return_value["contractName"] = "policy"
        with self.assertRaisesRegex(ValueError, "identity/schema"):
            environment.step([0, 0, 0])
        request.assert_called_with("stop")
        self.assertTrue(environment._needs_reset)

    def test_step_failure_keeps_original_error_and_closes_when_stop_cannot_be_confirmed(self):
        environment, process, _, _, request = self.environment()
        request.return_value = transition()
        environment.reset()
        request.side_effect = [ValueError("original bridge failure"), ValueError("stop failure")]
        with self.assertRaisesRegex(ValueError, "original bridge failure"):
            environment.step([0, 0, 0])
        self.assertTrue(environment._needs_reset)
        self.assertTrue(environment._closed)
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
