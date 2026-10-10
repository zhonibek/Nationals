import copy
import json
import math
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
    MotionActorCritic, COARSE_AUGMENTATION, MULTISCALE_FEATURES, AUGMENTATION_INDICES, DEADLINE_CONTEXT,
    augmentation_units, normalization_definition, reference_frame_features,
)
from roboproof.motion_learning.objective import SETTLING_MARGIN_OBJECTIVE, objective_definition
from roboproof.motion_learning.train import compatible_config, run, source_identity, train_seed


class MultiscaleTests(unittest.TestCase):
    def test_capacity_matched_profiles_keep_every_original_frame_and_initial_context(self):
        torch.manual_seed(99001)
        coarse = MotionActorCritic("broad-pace-v1", COARSE_AUGMENTATION)
        torch.manual_seed(99001)
        fine = MotionActorCritic("broad-pace-v1", MULTISCALE_FEATURES)
        baseline = MotionActorCritic("broad-pace-v1", "deadline-context-v1")
        self.assertEqual(len(coarse.scales), 169)
        torch.testing.assert_close(coarse.scales, fine.scales, rtol=0, atol=0)
        for before, after in zip(coarse.parameters(), fine.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(sum(parameter.numel() for parameter in coarse.parameters()) -
                         sum(parameter.numel() for parameter in baseline.parameters()), 1536)
        coarse.reset_policy_state(4)
        fine.reset_policy_state(4)
        baseline.reset_policy_state(4)
        for step in range(24):
            raw = [0.03 * (index - 15) for index in range(34)]
            raw[10] = step * 0.0254
            raw[18] = 10
            raw[25] = step * 0.05
            original = baseline.encode_observation(raw)
            for model in (coarse, fine):
                encoded = model.encode_observation(raw)
                for frame in range(4):
                    torch.testing.assert_close(encoded[frame * 41:frame * 41 + 35],
                                               original[frame * 35:frame * 35 + 35], rtol=0, atol=0)
                    torch.testing.assert_close(model.scales[frame * 41:frame * 41 + 35],
                                               baseline.scales[frame * 35:frame * 35 + 35], rtol=0, atol=0)
                    self.assertTrue(torch.all(encoded[frame * 41 + 35:frame * 41 + 41].abs() <= 1))
                torch.testing.assert_close(encoded[164:], original[140:], rtol=0, atol=0)
        history = copy.deepcopy(fine.observation_history)
        preview = fine.encode_observation(raw, advance_state=False)
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(history, fine.observation_history)))
        torch.testing.assert_close(fine.encode_observation(raw), preview, rtol=0, atol=0)
        fine.reset_policy_state(4)
        self.assertEqual(len(fine.observation_history), 0)
        self.assertIsNone(fine.initial_context)

    def test_bounded_local_units_follow_independent_formula_without_replacing_large_state_information(self):
        for profile in (COARSE_AUGMENTATION, MULTISCALE_FEATURES):
            model = MotionActorCritic("broad-pace-v1", profile)
            model.reset_policy_state(2)
            raw = [0.0] * 34
            raw[18], raw[13] = 10, 1
            raw[3], raw[4], raw[5] = 0.0254, -0.0127, 0.0873
            raw[10], raw[11], raw[12] = 0.02032, -0.01016, 0.035
            encoded = model.encode_observation(raw)
            frame = reference_frame_features(torch.tensor(raw))
            for axis, (index, unit) in enumerate(zip(AUGMENTATION_INDICES, augmentation_units(profile))):
                self.assertAlmostEqual(float(encoded[35 + axis]), math.tanh(float(frame[index]) / unit), places=6)
            raw[10] = 2
            larger = model.encode_observation(raw)
            raw[10] = 3
            largest = model.encode_observation(raw)
            self.assertEqual(float(larger[133]), 2)
            self.assertEqual(float(largest[133]), 3)
            self.assertTrue(torch.all(largest[158:164].abs() <= 1))
        units = augmentation_units(MULTISCALE_FEATURES)
        units[0] = 100
        self.assertEqual(augmentation_units(MULTISCALE_FEATURES)[0], 0.0254)
        with self.assertRaises(ValueError):
            augmentation_units("deadline-context-v1")

    def test_real_all_head_training_exact_resume_and_complete_trajectory_decoder_parity(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for offset, profile in enumerate((COARSE_AUGMENTATION, MULTISCALE_FEATURES)):
            seed = 99002 + offset
            config = {"updates": 4, "maxSteps": 4000, "rolloutSteps": 16, "seed": seed, "seedCount": 1,
                      "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                      "featureTransform": profile, "normalizationDefinition": normalization_definition(profile),
                      "deadlineContextDefinition": dict(DEADLINE_CONTEXT), "explorationConcentration": 8,
                      "objective": SETTLING_MARGIN_OBJECTIVE, "objectiveDefinition": objective_definition(SETTLING_MARGIN_OBJECTIVE),
                      "policyRegularization": "public-deadline-residual-v1", "deadlinePaceLossWeight": 40,
                      "warmStartUpdates": 1, "demonstrationAnchorWeight": 10, "demonstrationAnchorLossProfile": "pose-budget-v2"}
            with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
                directory = Path(temporary)
                sources = source_identity()
                complete = train_seed(environment, directory / "complete", seed, config, sources, time.monotonic() + 90)
                class Interrupted(Exception):
                    pass
                def interrupt(row):
                    if row["updates"] == 2:
                        raise Interrupted()
                with self.assertRaises(Interrupted):
                    train_seed(environment, directory / "resumed", seed, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
                resumed = train_seed(environment, directory / "resumed", seed, config, sources, time.monotonic() + 90, resume=True)
                self.assertEqual(complete["policySha256"], resumed["policySha256"])
                self.assertEqual(complete["history"], resumed["history"])
                warm = json.loads((directory / "complete/policy-0001.json").read_text())
                trained = json.loads((directory / "complete/policy-0004.json").read_text())
                for before, after in zip(warm["layers"][2]["weight"], trained["layers"][2]["weight"]):
                    self.assertNotEqual(before, after)
                checkpoint = torch.load(directory / "complete/checkpoint.pt", weights_only=True)
                model = MotionActorCritic("broad-pace-v1", profile)
                model.load_state_dict(checkpoint["model"])
                raw, info = environment.reset(seed=99004, configuration={"massKg": 11, "muLong": 0.44, "batteryInternalR": 0.08})
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
                script = "const fs=require('fs');const policy=require('./roboproof/motion-learner').policy(process.argv[1]);" \
                         "const request=JSON.parse(fs.readFileSync(0,'utf8'));policy.act.reset({referenceDurationSeconds:request.duration});" \
                         "const first=request.observations.map(row=>policy.act(row));policy.act.reset({referenceDurationSeconds:request.duration});" \
                         "const second=request.observations.map(row=>policy.act(row));console.log(JSON.stringify({first,second}));"
                result = subprocess.run(["node", "-e", script, str(directory / "complete" / complete["policyFile"])], cwd=ROOT,
                                        input=json.dumps({"duration": duration, "observations": observations}), capture_output=True,
                                        text=True, check=True, timeout=30)
                decoded = json.loads(result.stdout)
                self.assertEqual(decoded["first"], decoded["second"])
                for actual, reference in zip(decoded["first"], expected):
                    for actual_axis, expected_axis in zip(actual, reference):
                        self.assertAlmostEqual(actual_axis, expected_axis, places=5)
                corrupted = copy.deepcopy(checkpoint)
                corrupted["model"]["scales"][35] = 2
                (directory / "corrupted").mkdir()
                torch.save(corrupted, directory / "corrupted/checkpoint.pt")
                with self.assertRaisesRegex(ValueError, "normalization mismatch"):
                    train_seed(environment, directory / "corrupted", seed, config, sources, time.monotonic() + 90, resume=True)
                with self.assertRaisesRegex(ValueError, "definition mismatch"):
                    train_seed(environment, directory / "resumed", seed,
                               {**config, "normalizationDefinition": None}, sources, time.monotonic() + 90, resume=True)
            self.assertFalse(compatible_config(config, {**config, "featureTransform": "deadline-context-v1"}))

    def test_official_metadata_binds_profile_capacity_units_and_original_raw_contract(self):
        for offset, profile in enumerate((COARSE_AUGMENTATION, MULTISCALE_FEATURES)):
            with tempfile.TemporaryDirectory() as temporary:
                saved = run(directory=temporary, seed=99005 + offset, seed_count=1, updates=1, max_steps=1000,
                            rollout_steps=16, max_seconds=90, feature_transform=profile)
                actor = json.loads((Path(temporary) / saved["runId"] / f"seed-{99005 + offset}" / saved["models"][0]["policyFile"]).read_text())
            self.assertEqual((actor["observationSize"], actor["actionSize"], actor["featureInputSize"], actor["frameInputSize"]), (34, 4, 169, 41))
            self.assertEqual(actor["normalizationDefinition"], normalization_definition(profile))
            self.assertEqual(saved["config"]["normalizationDefinition"], actor["normalizationDefinition"])
            self.assertEqual(actor["historyFrames"], 4)
            self.assertFalse(saved["learnedImprovementVerified"])


if __name__ == "__main__":
    unittest.main()
