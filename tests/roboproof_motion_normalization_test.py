import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import (
    MotionActorCritic, SCALES, DEADLINE_CONTEXT, SETTLING_NORMALIZATION, SETTLING_FRAME_SCALES,
    normalization_definition,
)
from roboproof.motion_learning.objective import SETTLING_MARGIN_OBJECTIVE, objective_definition
from roboproof.motion_learning.train import compatible_config, run, source_identity, train_seed


class SensorNormalizationTests(unittest.TestCase):
    def test_scales_change_only_six_sensor_axes_per_frame_with_unchanged_capacity_and_encoding(self):
        torch.manual_seed(98001)
        baseline = MotionActorCritic("broad-pace-v1", "deadline-context-v1")
        torch.manual_seed(98001)
        candidate = MotionActorCritic("broad-pace-v1", SETTLING_NORMALIZATION)
        self.assertEqual(sum(parameter.numel() for parameter in baseline.parameters()),
                         sum(parameter.numel() for parameter in candidate.parameters()))
        for before, after in zip(baseline.parameters(), candidate.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        for frame in range(4):
            for index in range(35):
                expected = SETTLING_FRAME_SCALES.get(index, float(baseline.scales[frame * 35 + index]))
                self.assertEqual(float(candidate.scales[frame * 35 + index]), float(torch.tensor(expected)))
        torch.testing.assert_close(candidate.scales[140:], baseline.scales[140:], rtol=0, atol=0)
        self.assertEqual(len(candidate.scales), 145)
        baseline.reset_policy_state(4)
        candidate.reset_policy_state(4)
        for index in range(24):
            raw = [0.03 * (column - 15) for column in range(34)]
            raw[10] = index * 0.0254
            raw[12] = 0.1
            raw[13] = 0.99
            raw[18] = 10
            raw[25] = index * 0.05
            baseline_input = baseline.encode_observation(raw)
            candidate_input = candidate.encode_observation(raw)
            torch.testing.assert_close(candidate_input, baseline_input, rtol=0, atol=0)
            distribution, _ = candidate(candidate_input, encoded=True)
            reference, _ = baseline(baseline_input, encoded=True)
            torch.testing.assert_close(distribution.mean, reference.mean, rtol=0, atol=0)
        frozen_history = copy.deepcopy(candidate.observation_history)
        frozen_context = candidate.initial_context.clone()
        preview = candidate.encode_observation(raw, advance_state=False)
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(frozen_history, candidate.observation_history)))
        torch.testing.assert_close(candidate.initial_context, frozen_context, rtol=0, atol=0)
        torch.testing.assert_close(candidate.encode_observation(raw), preview, rtol=0, atol=0)
        candidate.reset_policy_state(4)
        self.assertEqual(len(candidate.observation_history), 0)
        self.assertIsNone(candidate.initial_context)

    def test_sensor_units_match_independent_component_formula_and_clip_without_truth(self):
        model = MotionActorCritic("broad-pace-v1", SETTLING_NORMALIZATION)
        model.reset_policy_state(2)
        raw = [0.0] * 34
        raw[18] = 10
        raw[3], raw[4], raw[5] = 0.0254, -0.0127, 0.0873
        raw[10], raw[11], raw[12], raw[13] = 0.02032, -0.01016, 0.035, 0.999
        encoded = model.encode_observation(raw)
        normalized = torch.clamp(encoded / model.scales, -10, 10)
        for frame in range(4):
            for index, expected in ((3, 1), (4, -0.5), (5, 1), (10, 1), (11, -0.5), (12, 1)):
                self.assertAlmostEqual(float(normalized[frame * 35 + index]), expected, places=6)
        self.assertAlmostEqual(float(normalized[140]), 0.02032 / SCALES[10], places=7)
        raw[10], raw[3], raw[5] = 3, -3, 4
        encoded = model.encode_observation(raw)
        normalized = torch.clamp(encoded / model.scales, -10, 10)
        self.assertEqual(float(normalized[115]), 10)
        self.assertEqual(float(normalized[108]), -10)
        self.assertEqual(float(normalized[110]), 10)
        definition = normalization_definition(SETTLING_NORMALIZATION)
        definition["frameScales"]["10"] = 1
        self.assertEqual(SETTLING_FRAME_SCALES[10], 0.02032)
        self.assertIsNone(normalization_definition("deadline-context-v1"))
        with self.assertRaises(ValueError):
            normalization_definition("unknown")

    def test_real_training_all_heads_exact_resume_and_python_javascript_trajectory_parity(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 4, "maxSteps": 4000, "rolloutSteps": 16, "seed": 98002, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": SETTLING_NORMALIZATION, "normalizationDefinition": normalization_definition(SETTLING_NORMALIZATION),
                  "deadlineContextDefinition": dict(DEADLINE_CONTEXT), "explorationConcentration": 8,
                  "objective": SETTLING_MARGIN_OBJECTIVE, "objectiveDefinition": objective_definition(SETTLING_MARGIN_OBJECTIVE),
                  "policyRegularization": "public-deadline-residual-v1", "deadlinePaceLossWeight": 40,
                  "warmStartUpdates": 1, "demonstrationAnchorWeight": 10, "demonstrationAnchorLossProfile": "pose-budget-v2"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 98002, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 98002, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 98002, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            warm = json.loads((directory / "complete/policy-0001.json").read_text())
            trained = json.loads((directory / "complete/policy-0004.json").read_text())
            for before, after in zip(warm["layers"][2]["weight"], trained["layers"][2]["weight"]):
                self.assertNotEqual(before, after)
            checkpoint = torch.load(directory / "complete/checkpoint.pt", weights_only=True)
            model = MotionActorCritic("broad-pace-v1", SETTLING_NORMALIZATION)
            model.load_state_dict(checkpoint["model"])
            raw, info = environment.reset(seed=98003)
            duration = info["referenceDurationSeconds"]
            model.reset_policy_state(duration)
            observations, expected = [], []
            for _ in range(220):
                observations.append(list(raw))
                action, _, _, _ = model.act(raw, deterministic=True)
                expected.append(action)
                raw, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    break
            self.assertTrue(terminated or truncated)
            self.assertGreater(len(observations), 24)
            script = "const fs=require('fs');const p=require('./roboproof/motion-learner').policy(process.argv[1]);" \
                     "const data=JSON.parse(fs.readFileSync(0,'utf8'));p.act.reset({referenceDurationSeconds:data.duration});" \
                     "const first=data.observations.map(row=>p.act(row));p.act.reset({referenceDurationSeconds:data.duration});" \
                     "const second=data.observations.map(row=>p.act(row));console.log(JSON.stringify({first,second}));"
            result = subprocess.run(["node", "-e", script, str(directory / "complete" / complete["policyFile"])], cwd=ROOT,
                                    input=json.dumps({"duration": duration, "observations": observations}), capture_output=True,
                                    text=True, check=True, timeout=30)
            decoded = json.loads(result.stdout)
            self.assertEqual(decoded["first"], decoded["second"])
            for actual, reference in zip(decoded["first"], expected):
                for actual_axis, expected_axis in zip(actual, reference):
                    self.assertAlmostEqual(actual_axis, expected_axis, places=5)
            corrupted = copy.deepcopy(checkpoint)
            corrupted["model"]["scales"][3] *= 2
            (directory / "corrupted").mkdir()
            torch.save(corrupted, directory / "corrupted/checkpoint.pt")
            with self.assertRaisesRegex(ValueError, "sensor normalization mismatch"):
                train_seed(environment, directory / "corrupted", 98002, config, sources, time.monotonic() + 90, resume=True)
            with self.assertRaisesRegex(ValueError, "normalization definition mismatch"):
                train_seed(environment, directory / "resumed", 98002,
                           {**config, "normalizationDefinition": None}, sources, time.monotonic() + 90, resume=True)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 98002,
                           {**config, "featureTransform": "deadline-context-v1", "normalizationDefinition": None},
                           sources, time.monotonic() + 90, resume=True)
        self.assertFalse(compatible_config(config, {**config, "normalizationDefinition": None}))

    def test_official_export_and_config_bind_fixed_units_and_unchanged_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = run(directory=temporary, seed=98004, seed_count=1, updates=1, max_steps=1000, rollout_steps=16,
                        max_seconds=90, feature_transform=SETTLING_NORMALIZATION)
            actor = json.loads((Path(temporary) / saved["runId"] / "seed-98004" / saved["models"][0]["policyFile"]).read_text())
        self.assertEqual(actor["observationSize"], 34)
        self.assertEqual(actor["actionSize"], 4)
        self.assertEqual(actor["featureInputSize"], 145)
        self.assertEqual(actor["historyFrames"], 4)
        self.assertEqual(actor["normalizationDefinition"], normalization_definition(SETTLING_NORMALIZATION))
        self.assertEqual(saved["config"]["normalizationDefinition"], actor["normalizationDefinition"])
        self.assertEqual(saved["config"]["deadlineContextDefinition"], DEADLINE_CONTEXT)
        self.assertFalse(saved["learnedImprovementVerified"])


if __name__ == "__main__":
    unittest.main()
