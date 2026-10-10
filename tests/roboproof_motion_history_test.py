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
from roboproof.motion_learning.model import MotionActorCritic, update
from roboproof.motion_learning.train import compatible_config, source_identity, train_seed


class MotionHistoryTests(unittest.TestCase):
    def test_history_is_bounded_chronological_and_keeps_initial_task_context(self):
        model = MotionActorCritic("broad-pace-v1", "history-context-v1")
        raw = [0.0] * 34
        raw[10:14] = [0.1, 0.2, 0.3, 0.4]
        raw[3] = 1
        first = model.encode_observation(raw)
        self.assertEqual(len(first), 144)
        self.assertEqual([float(first[index]) for index in (3, 38, 73, 108)], [1, 1, 1, 1])
        for velocity in (2, 3, 4, 5):
            raw[3] = velocity
            raw[10:14] = [1, 1, 1, 1]
            current = model.encode_observation(raw)
        self.assertEqual([float(current[index]) for index in (3, 38, 73, 108)], [2, 3, 4, 5])
        torch.testing.assert_close(current[-4:], torch.tensor([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(len(model.observation_history), 4)
        with self.assertRaises(ValueError):
            model.encode_observation([0] * 144)

    def test_bootstrap_preview_does_not_advance_history_or_replace_collected_input(self):
        model = MotionActorCritic("broad-pace-v1", "history-context-v1")
        raw = [0.0] * 34
        model.act(raw, deterministic=True)
        collected = list(model.last_input)
        history = [frame.clone() for frame in model.observation_history]
        raw[3] = 2
        expected = model.encode_observation(raw, advance_state=False)
        model.act(raw, deterministic=True, advance_state=False)
        self.assertEqual(model.last_input, collected)
        self.assertEqual(len(model.observation_history), len(history))
        self.assertTrue(all(torch.equal(actual, previous) for actual, previous in zip(model.observation_history, history)))
        model.act(raw, deterministic=True)
        torch.testing.assert_close(torch.tensor(model.last_input), expected)
        model.reset_policy_state()
        self.assertEqual(model.observation_history, [])
        self.assertIsNone(model.initial_context)
        self.assertIsNone(model.last_input)

    def test_temporal_updates_reject_missing_collected_features_before_optimization(self):
        model = MotionActorCritic("broad-pace-v1", "history-context-v1")
        with self.assertRaisesRegex(ValueError, "collected"):
            update(model, torch.optim.Adam(model.parameters()), [{"observation": [0.0] * 34}])
        with self.assertRaisesRegex(ValueError, "encoded"):
            model(torch.zeros(34))

    def test_real_temporal_gradients_resume_and_node_history_inference_replay(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 2, "maxSteps": 1000, "rolloutSteps": 16, "seed": 811, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "history-context-v1", "policyRegularization": "public-deadline-residual-v1",
                  "objective": "finite-total-return-pose-effort-v3"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 811, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 1:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 811, config, sources, time.monotonic() + 90,
                           on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 811, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(complete["actorWeightsChanged"])
            checkpoint = torch.load(directory / "complete/checkpoint.pt", weights_only=True)
            model = MotionActorCritic("broad-pace-v1", "history-context-v1")
            model.load_state_dict(checkpoint["model"])
            observation, _ = environment.reset(seed=19)
            observations, expected = [], []
            for _ in range(20):
                observations.append(observation)
                action, _, _, _ = model.act(observation, deterministic=True)
                expected.append(action)
                observation, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    break
            filename = directory / "complete" / complete["policyFile"]
            script = "const policy=require('./roboproof/motion-learner').policy(process.argv[1]);policy.act.reset();" \
                     "console.log(JSON.stringify(JSON.parse(process.argv[2]).map(row=>policy.act(row))));"
            result = subprocess.run(["node", "-e", script, str(filename), json.dumps(observations)], cwd=ROOT,
                                    capture_output=True, text=True, check=True, timeout=30)
            for actual_row, expected_row in zip(json.loads(result.stdout), expected):
                for actual, reference in zip(actual_row, expected_row):
                    self.assertAlmostEqual(actual, reference, places=6)
            replay_script = "const policy=require('./roboproof/motion-learner').policy(process.argv[1]);const motion=require('./roboproof/motion');" \
                            "const task={...require('./simulator/motion').DEFAULT_TASK,deadlineSeconds:0.15};" \
                            "const run=()=>motion.runPolicy({seed:19,task,act:policy.act,options:{policyIdentity:policy.identity}});" \
                            "const first=run();policy.act(Array(34).fill(1));const second=run();" \
                            "require('node:assert/strict').deepEqual(first,second);motion.replay(second);console.log('exact-reset-and-replay');"
            replay = subprocess.run(["node", "-e", replay_script, str(filename)], cwd=ROOT,
                                    capture_output=True, text=True, check=True, timeout=30)
            self.assertEqual(replay.stdout.strip(), "exact-reset-and-replay")
            actor = json.loads(filename.read_text())
            self.assertEqual(actor["observationSize"], 34)
            self.assertEqual(actor["featureInputSize"], 144)
            self.assertEqual(actor["historyFrames"], 4)
        self.assertFalse(compatible_config(config, {**config, "featureTransform": "reference-frame-v1"}))


if __name__ == "__main__":
    unittest.main()
