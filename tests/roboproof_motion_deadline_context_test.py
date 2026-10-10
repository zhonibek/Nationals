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
from roboproof.motion_learning.model import MotionActorCritic, DEADLINE_CONTEXT, public_deadline_floor, update
from roboproof.motion_learning.imitation import collect_demonstration
from roboproof.motion_learning.train import compatible_config, source_identity, train_seed, run


class MotionDeadlineContextTests(unittest.TestCase):
    def test_public_duration_is_required_bounded_and_preview_never_mutates_history(self):
        model = MotionActorCritic("broad-pace-v1", "deadline-context-v1")
        raw = [0.0] * 34
        raw[18] = 10
        raw[10:14] = [0.1, 0.2, 0.3, 0.4]
        with self.assertRaisesRegex(ValueError, "public reference"):
            model.encode_observation(raw)
        model.reset_policy_state(4)
        model.act(raw, deterministic=True)
        self.assertEqual(len(model.last_input), 145)
        self.assertEqual(model.last_input[-1], 0.5)
        collected = list(model.last_input)
        history = [frame.clone() for frame in model.observation_history]
        raw[18], raw[25] = 7, 1
        raw[3] = 2
        expected = model.encode_observation(raw, advance_state=False)
        self.assertAlmostEqual(float(expected[-1]), 0.6, places=6)
        model.act(raw, deterministic=True, advance_state=False)
        self.assertEqual(model.last_input, collected)
        self.assertTrue(all(torch.equal(frame, previous) for frame, previous in zip(model.observation_history, history)))
        model.act(raw, deterministic=True)
        torch.testing.assert_close(torch.tensor(model.last_input), expected)
        torch.testing.assert_close(expected[-5:-1], torch.tensor([0.1, 0.2, 0.3, 0.4]))
        raw[25] = 100
        self.assertEqual(float(model.encode_observation(raw)[-1]), 0.25)
        raw[25], raw[18] = 0, 1
        self.assertEqual(float(model.encode_observation(raw)[-1]), 1)
        model.reset_policy_state(0)
        self.assertEqual(float(model.encode_observation(raw)[-1]), 0.25)
        for invalid in (True, -1, float("nan"), float("inf"), "4"):
            with self.assertRaises(ValueError):
                model.reset_policy_state(invalid)
            self.assertIsNone(model.reference_duration)
            self.assertEqual(model.observation_history, [])
            self.assertIsNone(model.last_input)
            with self.assertRaises(ValueError):
                model.encode_observation(raw)
        with self.assertRaisesRegex(ValueError, "encoded"):
            model(torch.zeros(34))
        with self.assertRaisesRegex(ValueError, "collected"):
            update(model, torch.optim.Adam(model.parameters()), [{"observation": raw}])
        self.assertEqual(len(MotionActorCritic("broad-pace-v1", "history-context-v1").encode_observation(raw)), 144)

    def test_original_planner_supplies_teacher_context_without_changing_raw_sensors(self):
        torch.set_num_threads(1)
        model = MotionActorCritic("broad-pace-v1", "deadline-context-v1")
        world = {"seed": 1905, "task": {"start": {"xIn": 0, "yIn": 0, "headingDeg": 0},
                 "goal": {"xIn": 0, "yIn": 12, "headingDeg": 0}, "deadlineSeconds": 4}, "configuration": {}}
        with MotionEnvironment() as environment:
            observation, reset_info = environment.reset(seed=world["seed"], task=world["task"])
            self.assertEqual(len(observation), 34)
            model.reset_policy_state(reset_info["referenceDurationSeconds"])
            self.assertAlmostEqual(float(model.encode_observation(observation)[-1]),
                                   public_deadline_floor(reset_info["referenceDurationSeconds"], observation), places=6)
            demonstration = collect_demonstration(environment, model, world, time.monotonic() + 60, 1000,
                                                  "pose-deadline-margin-v2")
        self.assertFalse(demonstration["interrupted"])
        self.assertEqual(len(demonstration["attempts"]), 3)
        self.assertTrue(all(len(row["modelInput"]) == 145 and 0.25 <= row["modelInput"][-1] <= 1 for row in demonstration["rows"]))

    def test_real_deadline_context_training_and_resume_keep_exact_actor_and_public_parity(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 3, "maxSteps": 2000, "rolloutSteps": 16, "seed": 1907, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "deadline-context-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
                  "policyRegularization": "public-deadline-residual-v1", "warmStartUpdates": 1,
                  "demonstrationAnchorWeight": 10, "teacherSelection": "pose-deadline-margin-v2",
                  "objective": "finite-total-return-pose-effort-v3"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 1907, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 1907, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 1907, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(complete["actorWeightsChanged"])
            checkpoint = torch.load(directory / "complete/checkpoint.pt", weights_only=True)
            model = MotionActorCritic("broad-pace-v1", "deadline-context-v1")
            model.load_state_dict(checkpoint["model"])
            observation, reset_info = environment.reset(seed=19)
            duration = reset_info["referenceDurationSeconds"]
            model.reset_policy_state(duration)
            observations, expected = [], []
            for _ in range(20):
                observations.append(observation)
                action, _, _, _ = model.act(observation, deterministic=True)
                expected.append(action)
                observation, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    break
            filename = directory / "complete" / complete["policyFile"]
            script = "const policy=require('./roboproof/motion-learner').policy(process.argv[1]);" \
                     "policy.act.reset({referenceDurationSeconds:Number(process.argv[3])});" \
                     "console.log(JSON.stringify(JSON.parse(process.argv[2]).map(row=>policy.act(row))));"
            result = subprocess.run(["node", "-e", script, str(filename), json.dumps(observations), str(duration)], cwd=ROOT,
                                    capture_output=True, text=True, check=True, timeout=30)
            for actual_row, expected_row in zip(json.loads(result.stdout), expected):
                for actual, reference in zip(actual_row, expected_row):
                    self.assertAlmostEqual(actual, reference, places=6)
            replay_script = "const policy=require('./roboproof/motion-learner').policy(process.argv[1]);const motion=require('./roboproof/motion');" \
                            "const task={...require('./simulator/motion').DEFAULT_TASK,deadlineSeconds:0.15};" \
                            "const run=()=>motion.runPolicy({seed:19,task,act:policy.act,options:{policyIdentity:policy.identity}});" \
                            "const first=run();policy.act(Array(34).fill(1));const second=run();" \
                            "require('node:assert/strict').deepEqual(first,second);motion.replay(second);console.log('exact-public-reset-and-replay');"
            replay = subprocess.run(["node", "-e", replay_script, str(filename)], cwd=ROOT,
                                    capture_output=True, text=True, check=True, timeout=30)
            self.assertEqual(replay.stdout.strip(), "exact-public-reset-and-replay")
            actor = json.loads(filename.read_text())
            self.assertEqual(actor["observationSize"], 34)
            self.assertEqual(actor["featureInputSize"], 145)
            self.assertEqual(actor["deadlineContext"], DEADLINE_CONTEXT)
            for invalid in (None, {**DEADLINE_CONTEXT, "settlingReserveSeconds": 3}, {**DEADLINE_CONTEXT, "schemaVersion": True}):
                with self.assertRaisesRegex(ValueError, "context mismatch"):
                    train_seed(environment, directory / "resumed", 1907, {**config, "deadlineContextDefinition": invalid},
                               sources, time.monotonic() + 90, resume=True)
        self.assertFalse(compatible_config(config, {**config, "featureTransform": "history-context-v1"}))

    def test_official_run_declares_both_temporal_and_public_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run(directory=temporary, seed=1908, seed_count=1, updates=1, max_steps=500,
                         rollout_steps=16, max_seconds=90, feature_transform="deadline-context-v1")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["config"]["deadlineContextDefinition"], DEADLINE_CONTEXT)
        self.assertEqual(result["config"]["temporalDefinition"]["historyFrames"], 4)


if __name__ == "__main__":
    unittest.main()
