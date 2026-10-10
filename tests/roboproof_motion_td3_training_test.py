import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import MotionActorCritic
from roboproof.motion_learning.td3 import ALGORITHM
from roboproof.motion_learning.td3_train import configuration, run, source_identity, train_seed


def config(seed, updates=4, warm=1, rollout=16):
    return configuration(seed, 1, updates, 3000, rollout, warm, "deadline-context-v1",
                         "finite-total-return-settle-margin-v4", "mixed-full-reach-v5", 4, 1024, 0.05,
                         10 if warm else 0, "public-deadline-residual-v1", 40)


class CountedEnvironment:
    def __init__(self, environment):
        self.environment = environment
        self.contract = environment.contract
        self.steps = 0

    def reset(self, **kwargs):
        return self.environment.reset(**kwargs)

    def step(self, action):
        result = self.environment.step(action)
        self.steps += 1
        return result

    def report(self):
        return self.environment.report()


class TD3TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_actual_original_simulator_training_exact_resume_all_heads_and_complete_decoder_parity(self):
        parameters = config(102001)
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            counted = CountedEnvironment(environment)
            sources = source_identity()
            complete = train_seed(counted, directory / "complete", 102001, parameters, sources, time.monotonic() + 90)
            self.assertEqual(complete["collectionSteps"], counted.steps)
            self.assertGreater(complete["collectionSteps"], complete["steps"])
            self.assertEqual((complete["td3GradientUpdates"], complete["td3ActorUpdates"]), (12, 6))
            self.assertGreater(complete["replayRows"], 0)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 102001, parameters, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 102001, parameters, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            first = torch.load(directory / "complete/checkpoint.pt", weights_only=True)
            second = torch.load(directory / "resumed/checkpoint.pt", weights_only=True)
            for name in first["replay"]:
                if isinstance(first["replay"][name], torch.Tensor):
                    torch.testing.assert_close(first["replay"][name], second["replay"][name], rtol=0, atol=0)
                else:
                    self.assertEqual(first["replay"][name], second["replay"][name])
            warm = json.loads((directory / "complete/policy-0001.json").read_text())
            trained = json.loads((directory / "complete/policy-0004.json").read_text())
            self.assertEqual(trained["algorithm"], ALGORITHM)
            for before, after in zip(warm["layers"][2]["weight"], trained["layers"][2]["weight"]):
                self.assertNotEqual(before, after)
            self.assertLess((directory / "complete/checkpoint.pt").stat().st_size, 4 * 1024 * 1024)
            actor = MotionActorCritic("broad-pace-v1", "deadline-context-v1", value_critic=False)
            actor.load_state_dict(first["model"])
            observation, information = environment.reset(seed=102002, configuration={"massKg": 11, "muLong": 0.44})
            duration = information["referenceDurationSeconds"]
            actor.reset_policy_state(duration)
            observations, expected = [], []
            for _ in range(220):
                observations.append(list(observation))
                action, _, _, _ = actor.act(observation, deterministic=True)
                expected.append(action)
                observation, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    break
            self.assertTrue(terminated or truncated)
            script = "const fs=require('fs');const policy=require('./roboproof/motion-learner').policy(process.argv[1]);" \
                     "const request=JSON.parse(fs.readFileSync(0,'utf8'));policy.act.reset({referenceDurationSeconds:request.duration});" \
                     "console.log(JSON.stringify({id:policy.identity.id,algorithm:policy.data.algorithm,actions:request.observations.map(row=>policy.act(row))}));"
            result = subprocess.run(["node", "-e", script, str(directory / "complete" / complete["policyFile"])], cwd=ROOT,
                                    input=json.dumps({"duration": duration, "observations": observations}), text=True,
                                    capture_output=True, check=True, timeout=30)
            decoded = json.loads(result.stdout)
            self.assertTrue(decoded["id"].startswith("td3-"))
            self.assertEqual(decoded["algorithm"], ALGORITHM)
            for actual, reference in zip(decoded["actions"], expected):
                for actual_axis, expected_axis in zip(actual, reference):
                    self.assertAlmostEqual(actual_axis, expected_axis, places=5)
            malformed = copy.deepcopy(parameters)
            malformed["td3Definition"]["policyDelay"] = 1
            with self.assertRaisesRegex(ValueError, "definition mismatch"):
                train_seed(environment, directory / "resumed", 102001, malformed, sources, time.monotonic() + 90, resume=True)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 102001, parameters, {**sources, "changed": "source"}, time.monotonic() + 90, resume=True)

    def test_partial_collection_rolls_back_rng_replay_and_all_learning_states_with_spent_work_retained(self):
        parameters = config(102003, updates=2, warm=0, rollout=128)
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 102003, parameters, sources, time.monotonic() + 90)
            counted = CountedEnvironment(environment)
            initial = {}
            def capture(row):
                if row["updates"] == 0 and not initial:
                    initial.update(torch.load(directory / "partial/checkpoint.pt", weights_only=True))
            with patch("roboproof.motion_learning.td3_train.time.monotonic", side_effect=lambda: 0 if counted.steps < 75 else 100):
                partial = train_seed(counted, directory / "partial", 102003, parameters, sources, 50, on_checkpoint=capture)
            self.assertEqual((partial["updates"], partial["collectionSteps"], partial["discardedUncommittedSteps"]), (0, 75, 75))
            saved = torch.load(directory / "partial/checkpoint.pt", weights_only=True)
            self.assertEqual(saved["replay"]["count"], 0)
            self.assertEqual(saved["learner"]["gradientUpdates"], 0)
            self.assertEqual(saved["discardedCollectionStepsTotal"], 75)
            torch.testing.assert_close(saved["torchRng"], initial["torchRng"], rtol=0, atol=0)
            self.assertEqual(saved["worldRng"], initial["worldRng"])
            for name in initial["model"]:
                torch.testing.assert_close(saved["model"][name], initial["model"][name], rtol=0, atol=0)
            exposures = json.loads((directory / "partial/exposure.json").read_text())["cases"]
            self.assertGreaterEqual(len(exposures), 2)
            resumed = train_seed(environment, directory / "partial", 102003, parameters, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(resumed["policySha256"], complete["policySha256"])
            self.assertEqual(resumed["history"], complete["history"])
            self.assertEqual(resumed["collectionSteps"], complete["collectionSteps"] + 75)

    def test_official_manifest_checkpoint_and_existing_node_reader_keep_td3_experimental_and_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = run(directory=temporary, seed=102004, seed_count=1, updates=2, max_steps=2000, rollout_steps=16,
                        max_seconds=90, warm_start_updates=1, gradient_steps=4)
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["config"]["algorithm"], ALGORITHM)
            self.assertFalse(saved["learnedImprovementVerified"])
            self.assertEqual(saved["device"], "cpu")
            script = "const status=require('./roboproof/motion-learner').status(process.argv[1]);console.log(JSON.stringify(status));"
            result = subprocess.run(["node", "-e", script, temporary], cwd=ROOT, text=True, capture_output=True, check=True, timeout=30)
            status = json.loads(result.stdout)
            self.assertTrue(status["available"])
            self.assertTrue(status["motionPolicyTrained"])
            self.assertFalse(status["learnedImprovementVerified"])
            self.assertFalse(status["hardwareExecutionEnabled"])
            self.assertFalse(status["cloudExecutionEnabled"])
            checkpoint_path = Path(temporary) / saved["runId"] / "seed-102004/checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(checkpoint["algorithm"], ALGORITHM)
            self.assertEqual(checkpoint["config"]["td3Definition"]["policyDelay"], 2)
            corrupted = copy.deepcopy(checkpoint)
            corrupted["model"]["scales"][3] *= 2
            torch.save(corrupted, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "scale mismatch"):
                run(directory=temporary, seed=102004, seed_count=1, updates=2, max_steps=2000, rollout_steps=16,
                    max_seconds=90, warm_start_updates=1, gradient_steps=4, resume=saved["runId"])

    def test_budget_and_hyperparameter_guards_do_not_start_invalid_training(self):
        for arguments in ({"max_steps": 50001}, {"seed_count": 4}, {"max_seconds": 901},
                          {"gradient_steps": 33}, {"exploration_std": 0}, {"exploration_std": float("nan")},
                          {"warm_start_updates": 64}, {"replay_capacity": 4096}):
            with self.assertRaises(ValueError):
                run(**arguments)


if __name__ == "__main__":
    unittest.main()
