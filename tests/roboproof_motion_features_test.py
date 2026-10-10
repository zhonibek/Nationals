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
from roboproof.motion_learning.model import MotionActorCritic, reference_frame_features
from roboproof.motion_learning.train import compatible_config, source_identity, train_seed


class MotionFeatureTests(unittest.TestCase):
    def test_features_use_only_raw_sensors_and_known_reference_coordinates(self):
        raw = torch.zeros(34, dtype=torch.float64)
        raw[0], raw[1], raw[19], raw[20] = 1, 2, 1, 3
        raw[21] = math.pi / 2
        raw[3], raw[11], raw[27] = 1, 1, 0.08
        original = raw.clone()
        features = reference_frame_features(raw)
        self.assertTrue(torch.equal(raw, original))
        self.assertEqual(len(features), 35)
        self.assertAlmostEqual(float(features[19]), -1)
        self.assertAlmostEqual(float(features[20]), 0)
        self.assertAlmostEqual(float(features[3]), 0)
        self.assertAlmostEqual(float(features[4]), 1)
        self.assertAlmostEqual(float(features[10]), -1)
        self.assertAlmostEqual(float(features[11]), 0)
        self.assertAlmostEqual(float(features[27]), 0)
        self.assertAlmostEqual(float(features[28]), 0.08)
        self.assertAlmostEqual(float(features[21]), 1)
        self.assertAlmostEqual(float(features[34]), 0)
        self.assertTrue(torch.equal(features[:3], raw[:3]))
        with self.assertRaises(ValueError):
            reference_frame_features(torch.zeros(35))

    def test_local_motion_features_are_translation_and_rotation_consistent(self):
        raw = torch.arange(34, dtype=torch.float64) / 30
        raw[2], raw[21] = 0.2, 0.7
        shifted = raw.clone()
        shifted[0] += 5
        shifted[19] += 5
        shifted[1] -= 3
        shifted[20] -= 3
        expected = reference_frame_features(raw)
        actual = reference_frame_features(shifted)
        torch.testing.assert_close(actual[3:], expected[3:], rtol=1e-12, atol=1e-12)
        rotated = raw.clone()
        angle = 0.8
        for first, second in ((0, 1), (3, 4), (19, 20), (22, 23), (27, 28), (30, 31)):
            rotated[first] = raw[first] * math.cos(angle) + raw[second] * math.sin(angle)
            rotated[second] = -raw[first] * math.sin(angle) + raw[second] * math.cos(angle)
        rotated[2] += angle
        rotated[21] += angle
        torch.testing.assert_close(reference_frame_features(rotated)[3:], expected[3:], rtol=1e-12, atol=1e-12)
        self.assertFalse(torch.equal(reference_frame_features(rotated)[:3], expected[:3]))

    def test_relative_heading_features_remain_continuous_at_wrap(self):
        before = torch.zeros(34, dtype=torch.float64)
        after = before.clone()
        before[2], after[2] = math.pi - 1e-7, -math.pi + 1e-7
        before[21] = after[21] = math.pi
        first, second = reference_frame_features(before), reference_frame_features(after)
        self.assertLess(abs(float(first[21] - second[21])), 3e-7)
        self.assertAlmostEqual(float(first[34]), float(second[34]), places=12)

    def test_real_engine_gradients_and_exported_node_inference_agree(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 2, "maxSteps": 1000, "rolloutSteps": 16, "seed": 503, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "reference-frame-v1", "policyRegularization": "public-deadline-residual-v1"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            summary = train_seed(environment, directory, 503, config, source_identity(), time.monotonic() + 90)
            self.assertTrue(summary["actorWeightsChanged"])
            checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True)
            model = MotionActorCritic("broad-pace-v1", "reference-frame-v1")
            model.load_state_dict(checkpoint["model"])
            observations = []
            expected = []
            observation, _ = environment.reset(seed=17)
            for _ in range(20):
                observations.append(observation)
                action, _, _, _ = model.act(observation, deterministic=True)
                expected.append(action)
                observation, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    break
            script = "const model=require('./roboproof/motion-learner').policy(process.argv[1]);" \
                     "console.log(JSON.stringify(JSON.parse(process.argv[2]).map(row=>model.act(row))));"
            result = subprocess.run(["node", "-e", script, str(directory / summary["policyFile"]), json.dumps(observations)],
                                    cwd=ROOT, capture_output=True, text=True, check=True, timeout=30)
            for row, reference in zip(json.loads(result.stdout), expected):
                for actual, value in zip(row, reference):
                    self.assertAlmostEqual(actual, value, places=6)
            actor = json.loads((directory / summary["policyFile"]).read_text())
            self.assertEqual(actor["observationSize"], 34)
            self.assertEqual(actor["featureInputSize"], 35)
            self.assertEqual(len(actor["scales"]), 35)
            self.assertEqual(len(actor["layers"][0]["weight"][0]), 35)
            replay_script = "const policy=require('./roboproof/motion-learner').policy(process.argv[1]);" \
                            "const motion=require('./roboproof/motion');const task={...require('./simulator/motion').DEFAULT_TASK,deadlineSeconds:0.15};" \
                            "const report=motion.runPolicy({seed:17,task,act:policy.act,options:{policyIdentity:policy.identity}});" \
                            "motion.replay(report);console.log(JSON.stringify({replay:true,ticks:report.ticks,kind:report.options.policyIdentity.kind}));"
            replay = subprocess.run(["node", "-e", replay_script, str(directory / summary["policyFile"])], cwd=ROOT,
                                    capture_output=True, text=True, check=True, timeout=30)
            self.assertEqual(json.loads(replay.stdout), {"replay": True, "ticks": 15, "kind": "learned"})
        self.assertFalse(compatible_config(config, {**config, "featureTransform": "identity-v1"}))
        with self.assertRaises(ValueError):
            MotionActorCritic(feature_transform="evaluator-truth")


if __name__ == "__main__":
    unittest.main()
