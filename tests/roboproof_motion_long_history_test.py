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
from roboproof.motion_learning.model import MotionActorCritic, DEADLINE_CONTEXT, reference_frame_features, public_deadline_floor
from roboproof.motion_learning.train import compatible_config, curriculum_definition, run, source_identity, train_seed


class LongHistoryTests(unittest.TestCase):
    def test_sixteen_sensor_frames_padding_eviction_preview_and_episode_reset(self):
        model = MotionActorCritic("broad-pace-v1", "long-history-deadline-v1")
        self.assertEqual(len(model.scales), 565)
        self.assertEqual(model.history_frames, 16)
        raw = [0.0] * 34
        raw[18], raw[10:14] = 10, [0.1, 0.2, 0.3, 0.4]
        with self.assertRaisesRegex(ValueError, "public reference"):
            model.encode_observation(raw)
        model.reset_policy_state(4)
        frames = []
        for index in range(24):
            raw[3], raw[25], raw[10] = index * 0.1, index * 0.05, index * 0.02
            frames.append(reference_frame_features(torch.tensor(raw)))
            retained = frames[-16:]
            expected = torch.cat([*([retained[0]] * (16 - len(retained))), *retained,
                                  torch.tensor([0, 0.2, 0.3, 0.4]),
                                  torch.tensor([public_deadline_floor(4, raw)])])
            encoded = model.encode_observation(raw)
            torch.testing.assert_close(encoded, expected)
            self.assertLessEqual(len(model.observation_history), 16)
        history = [frame.clone() for frame in model.observation_history]
        context = model.initial_context.clone()
        preview = list(raw)
        preview[3] = 9
        expected = model.encode_observation(preview, advance_state=False)
        model.act(preview, deterministic=True, advance_state=False)
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(history, model.observation_history)))
        self.assertTrue(torch.equal(context, model.initial_context))
        torch.testing.assert_close(model.encode_observation(preview), expected)
        model.reset_policy_state(2)
        reset = model.encode_observation(preview)
        current = reference_frame_features(torch.tensor(preview))
        torch.testing.assert_close(reset[:560], current.repeat(16))
        self.assertEqual(len(model.observation_history), 1)
        self.assertEqual(len(MotionActorCritic("broad-pace-v1", "deadline-context-v1").scales), 145)

    def test_actual_long_history_gradients_resume_and_javascript_trace_parity(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 4, "maxSteps": 4000, "rolloutSteps": 16, "seed": 93001, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "curriculum": "mixed-full-reach-v5", "curriculumDefinition": curriculum_definition("mixed-full-reach-v5"),
                  "featureTransform": "long-history-deadline-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
                  "explorationConcentration": 8, "objective": "finite-total-return-pose-effort-v3",
                  "policyRegularization": "public-deadline-residual-v1", "deadlinePaceLossWeight": 40,
                  "warmStartUpdates": 1, "demonstrationAnchorWeight": 10, "demonstrationAnchorLossProfile": "pose-budget-v2",
                  "teacherSelection": "pose-deadline-margin-v2"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 93001, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 93001, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 93001, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(complete["actorWeightsChanged"])
            checkpoint_path = directory / "complete/checkpoint.pt"
            self.assertLess(checkpoint_path.stat().st_size, 4 * 1024 * 1024)
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model = MotionActorCritic("broad-pace-v1", "long-history-deadline-v1")
            model.load_state_dict(checkpoint["model"])
            raw, reset_info = environment.reset(seed=93002)
            duration = reset_info["referenceDurationSeconds"]
            model.reset_policy_state(duration)
            observations, expected = [], []
            for _ in range(22):
                observations.append(list(raw))
                action, _, _, _ = model.act(raw, deterministic=True)
                expected.append(action)
                raw, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    break
            script = "const fs=require('fs');const p=require('./roboproof/motion-learner').policy(process.argv[1]);" \
                     "const request=JSON.parse(fs.readFileSync(0,'utf8'));p.act.reset({referenceDurationSeconds:request.duration});" \
                     "const first=request.observations.map(row=>p.act(row));p.act.reset({referenceDurationSeconds:request.duration});" \
                     "const second=request.observations.map(row=>p.act(row));console.log(JSON.stringify({first,second}));"
            policy_path = directory / "complete" / complete["policyFile"]
            result = subprocess.run(["node", "-e", script, str(policy_path)], cwd=ROOT, capture_output=True, text=True,
                                    input=json.dumps({"duration": duration, "observations": observations}), check=True, timeout=30)
            decoded = json.loads(result.stdout)
            self.assertEqual(decoded["first"], decoded["second"])
            for actual, reference in zip(decoded["first"], expected):
                for actual_axis, expected_axis in zip(actual, reference):
                    self.assertAlmostEqual(actual_axis, expected_axis, places=5)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 93001, {**config, "featureTransform": "deadline-context-v1"},
                           sources, time.monotonic() + 90, resume=True)
        self.assertFalse(compatible_config(config, {**config, "featureTransform": "deadline-context-v1"}))

    def test_official_export_declares_fixed_window_with_unchanged_raw_action_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = run(directory=temporary, seed=93003, seed_count=1, updates=1, max_steps=1000,
                        rollout_steps=16, max_seconds=90, feature_transform="long-history-deadline-v1")
            actor = json.loads((Path(temporary) / saved["runId"] / "seed-93003" / saved["models"][0]["policyFile"]).read_text())
        self.assertEqual(actor["observationSize"], 34)
        self.assertEqual(actor["actionSize"], 4)
        self.assertEqual(actor["featureInputSize"], 565)
        self.assertEqual(actor["historyFrames"], 16)
        self.assertEqual(actor["deadlineContext"], DEADLINE_CONTEXT)
        self.assertEqual(saved["config"]["temporalDefinition"]["historyFrames"], 16)
        self.assertFalse(saved["learnedImprovementVerified"])


if __name__ == "__main__":
    unittest.main()
