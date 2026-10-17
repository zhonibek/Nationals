import copy
import inspect
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

from roboproof.motion_learning.evaluation_exclusion import geometry_keys, reject_consumed_final
from roboproof.motion_learning.rollout_retention import retained_success_rows
from roboproof.motion_learning.checkpoint_limits import checkpoint_tensor_bytes
from roboproof.motion_learning.td3_train import configuration, run

ROOT = Path(__file__).resolve().parents[1]
TASK = {"start": {"xIn": 0, "yIn": 0, "headingDeg": 0},
        "goal": {"xIn": 0, "yIn": 12, "headingDeg": 90}, "deadlineSeconds": 4}


def report():
    return {"reason": "success", "metrics": {"contactSeconds": 0, "positionErrorMeters": 0.005,
        "headingErrorRadians": 0.005, "elapsedSeconds": 2, "effortProxyVAs": 3}}


def rows():
    return [{"modelInput": [index / 100] * 34, "action": [0.3, 0.1, -0.1, 0.0], "terminal": index == 31}
            for index in range(32)]


class MotionRetentionTests(unittest.TestCase):
    def test_only_actual_complete_robust_success_rows_are_retained(self):
        measured = rows()
        selected = retained_success_rows(measured, report(), TASK)
        self.assertEqual(len(selected), 16)
        self.assertEqual(selected[0]["modelInput"], measured[0]["modelInput"])
        self.assertEqual(selected[-1]["modelInput"], measured[-1]["modelInput"])
        self.assertTrue(all(set(row) == {"modelInput", "teacherAction"} for row in selected))
        self.assertTrue(all(row["teacherAction"] == measured[0]["action"] for row in selected))
        selected[0]["modelInput"][0] = 99
        self.assertEqual(measured[0]["modelInput"][0], 0)

    def test_failure_contact_or_weak_margins_never_supply_self_labels(self):
        for changes in ({"contactSeconds": 1e-12}, {"positionErrorMeters": 0.019},
                        {"headingErrorRadians": 0.034}, {"elapsedSeconds": 3.5}):
            measured = report()
            measured["metrics"].update(changes)
            self.assertEqual(retained_success_rows(rows(), measured, TASK), [])
        for reason in ("time_limit", "stopped", "controller_or_sensor_fault", "stale_policy_action"):
            self.assertEqual(retained_success_rows(rows(), {**report(), "reason": reason}, TASK), [])

    def test_partial_cross_episode_and_malformed_actions_fail_closed(self):
        mutations = [[], rows()[:-1], rows(), rows(), rows(), rows()]
        mutations[2][0]["terminal"] = True
        mutations[3][3]["action"][1] = 1.01
        mutations[4][3]["modelInput"][1] = float("nan")
        mutations[5][4]["modelInput"] = [0.1]
        for measured in mutations:
            with self.assertRaises(ValueError):
                retained_success_rows(measured, report(), TASK)

    def test_optional_bank_is_source_config_bound_and_capacity_accounted(self):
        defaults = {name: parameter.default for name, parameter in inspect.signature(run).parameters.items()}
        arguments = {name: defaults[name] for name in inspect.signature(configuration).parameters}
        baseline = configuration(**arguments)
        self.assertNotIn("selfRetentionWeight", baseline)
        candidate = configuration(**{**arguments, "self_retention_weight": 1})
        self.assertEqual(candidate["selfRetentionDefinition"]["capacity"], 512)
        self.assertEqual(candidate.get("networkDepth", 2), 2)
        self.assertEqual(checkpoint_tensor_bytes(candidate) - checkpoint_tensor_bytes(baseline), 512 * (145 + 4) * 4)
        for invalid in (True, -1, 101, "1", float("nan")):
            with self.assertRaises(ValueError):
                configuration(**{**arguments, "self_retention_weight": invalid})

    def test_body_geometry_matches_shared_node_exclusion_and_conservative_ties(self):
        generator = random.Random(343052)
        tasks = [copy.deepcopy(TASK)]
        for _ in range(100):
            tasks.append({"start": {"xIn": generator.uniform(-5, 5), "yIn": generator.uniform(-5, 5),
                                    "headingDeg": generator.uniform(-180, 180)},
                "goal": {"xIn": generator.uniform(-40, 40), "yIn": generator.uniform(-40, 40),
                         "headingDeg": generator.uniform(-180, 180)}, "deadlineSeconds": 6})
        for distance in (0.05, -0.05, 0.05 - 1e-12, -0.05 + 1e-12):
            tasks.append({**copy.deepcopy(TASK), "goal": {"xIn": distance, "yIn": 12, "headingDeg": 90}})
        program = "const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));console.log(JSON.stringify(input.map(require('./roboproof/motion-exposures').geometryFingerprint)));"
        hashes = json.loads(subprocess.run(["node", "-e", program], input=json.dumps(tasks), text=True,
            capture_output=True, cwd=ROOT, check=True, timeout=30).stdout)
        for task, hashed in zip(tasks, hashes):
            self.assertIn(hashed, geometry_keys(task))

    def test_consumed_seed_and_rotated_deadline_aliases_are_blocked_without_current_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "roboproof/runs/motion/holdout/final"
            filename = root / "candidate.json"
            filename.write_text('{"synthetic":true}')
            program = "const fs=require('node:fs');const input=JSON.parse(fs.readFileSync(0,'utf8'));const audit=require('./roboproof/motion-holdout').createLocalFilesystemHoldoutAudit(input.folder,{readCurrentSourceRuntimeIdentity:()=>({sources:{synthetic:'a'.repeat(64)},runtime:{node:process.version}})});audit.commitFreshCorpus({cases:[{id:'synthetic-only',seed:7,task:input.task}]});audit.freezeCandidateFilesBeforeReveal([{id:'synthetic',path:input.candidate}]);audit.consumeAndRevealOnce();"
            subprocess.run(["node", "-e", program], input=json.dumps({"folder": str(folder), "candidate": str(filename), "task": TASK}),
                           cwd=ROOT, text=True, capture_output=True, check=True, timeout=30)
            with self.assertRaisesRegex(RuntimeError, "excluded"):
                reject_consumed_final({"seed": 7, "task": {**TASK, "deadlineSeconds": 10}}, root)
            rotated = {"start": {"xIn": 10, "yIn": 10, "headingDeg": 90},
                       "goal": {"xIn": 22, "yIn": 10, "headingDeg": 180}, "deadlineSeconds": 10}
            with self.assertRaisesRegex(RuntimeError, "geometry alias"):
                reject_consumed_final({"seed": 8, "task": rotated}, root)
            different = {"seed": 8, "task": {**TASK, "goal": {"xIn": 0, "yIn": 20, "headingDeg": 0}}}
            self.assertIs(reject_consumed_final(different, root), different)
            (folder / "commitment.json").write_text((folder / "commitment.json").read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash linkage"):
                reject_consumed_final(different, root)


if __name__ == "__main__":
    unittest.main()
