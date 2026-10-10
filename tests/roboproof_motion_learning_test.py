import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import MotionActorCritic, advantages, update
from roboproof.motion_learning.train import compatible_config, run, source_identity, train_seed, training_world


class MotionLearningTests(unittest.TestCase):
    def test_terminal_and_truncation_bootstrap_are_distinct(self):
        advantage, _ = advantages([1, 1], [0, 0], [10, 10], [True, False], [True, True])
        self.assertAlmostEqual(float(advantage[0]), 1)
        self.assertAlmostEqual(float(advantage[1]), 10.9, places=5)

    def test_training_worlds_are_bounded_curriculum_not_frozen_tasks(self):
        generator = random.Random(42)
        frozen = [world for version in ("", "-v2") for world in
                  json.loads((ROOT / f"tests/fixtures/motion-evaluation-worlds{version}.json").read_text())["cases"]]
        for index in range(100):
            world = training_world(generator, index)
            self.assertTrue(all(world["seed"] != row["seed"] and world["task"] != row["task"] for row in frozen))
            self.assertLessEqual(abs(world["task"]["goal"]["xIn"]), 36)
            self.assertLessEqual(abs(world["task"]["goal"]["yIn"]), 36)
            self.assertEqual(world["stage"], 0 if index < 4 else (1, 2, 0, 2)[(index - 4) % 4])
        early = [training_world(random.Random(index), index)["stage"] for index in range(8)]
        self.assertEqual(early.count(2), 2)
        self.assertEqual(early.count(0), 5)
        field = [training_world(random.Random(index), index, "field-reach-v4") for index in range(8, 40)]
        self.assertTrue(any(abs(world["task"]["goal"]["xIn"]) == 60 or abs(world["task"]["goal"]["yIn"]) == 60 for world in field))
        self.assertTrue(all(0.02 <= world["configuration"]["muLat"] <= 0.15 for world in field if world["stage"] == 2))

    def test_real_gradients_checkpoint_resume_and_node_inference_parity(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 2, "maxSteps": 1000, "rolloutSteps": 16,
                  "seed": 84, "seedCount": 1, "algorithm": "ppo-beta-reference-v1"}
        sources = source_identity()
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            complete = train_seed(environment, directory / "complete", 84, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 1:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 84, config, sources, time.monotonic() + 90,
                           on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 84, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["steps"], resumed["steps"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertTrue(complete["actorWeightsChanged"])
            self.assertGreater(complete["history"][0]["maximumGradientNorm"], 0)
            diagnostic = complete["history"][0]
            for field in ("meanActorLoss", "meanCriticLoss", "meanEntropy", "meanApproxKL", "clipFraction"):
                self.assertTrue(torch.isfinite(torch.tensor(diagnostic[field])))
            self.assertGreaterEqual(diagnostic["clipFraction"], 0)
            self.assertLessEqual(diagnostic["clipFraction"], 1)
            self.assertEqual(len(diagnostic["sampledActionStd"]), 4)
            self.assertGreater(sum(diagnostic["sampledActionStd"]), 0)
            self.assertEqual(sum(diagnostic["curriculumStageCounts"].values()), len(diagnostic["worlds"]))
            checkpoint = torch.load(directory / "complete/checkpoint.pt", weights_only=True)
            model = MotionActorCritic()
            model.load_state_dict(checkpoint["model"])
            observation, _ = environment.reset(seed=22)
            expected, _, _, _ = model.act(observation, deterministic=True)
            filename = directory / "complete" / complete["policyFile"]
            script = "const model=require('./roboproof/motion-learner').policy(process.argv[1]);" \
                     "console.log(JSON.stringify(model.act(JSON.parse(process.argv[2]))));"
            result = subprocess.run(["node", "-e", script, str(filename), json.dumps(observation)],
                                    cwd=ROOT, capture_output=True, text=True, check=True, timeout=30)
            for actual, reference in zip(json.loads(result.stdout), expected):
                self.assertAlmostEqual(actual, reference, places=6)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 84, config, {"wrong": "source"},
                           time.monotonic() + 90, resume=True)

    def test_budget_rejects_bad_requests_and_does_not_claim_uncommitted_learning(self):
        configuration = {"rolloutSteps": 16, "seed": 18, "seedCount": 1, "algorithm": "ppo-beta-reference-v1", "updates": 2, "maxSteps": 1000}
        self.assertTrue(compatible_config(configuration, {**configuration, "updates": 3, "maxSteps": 2000}))
        self.assertFalse(compatible_config(configuration, {**configuration, "rolloutSteps": 32}))
        self.assertFalse(compatible_config(configuration, {**configuration, "maxSteps": 1}))
        self.assertFalse(compatible_config(configuration, {**configuration, "curriculum": "different"}))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "budget"):
                run(directory=directory, max_steps=0)
            result = run(directory=directory, seed=18, seed_count=1, updates=2, max_steps=1,
                         rollout_steps=16, max_seconds=30)
            self.assertEqual(result["status"], "budget-stopped")
            self.assertFalse(result["models"][0]["trained"])
            self.assertEqual(result["models"][0]["steps"], 0)
            self.assertEqual(result["models"][0]["discardedUncommittedSteps"], 1)
            self.assertFalse(result["learnedImprovementVerified"])

    def test_constant_returns_have_no_invented_explained_variance(self):
        torch.manual_seed(12)
        model = MotionActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        observation = [0.0] * 34
        rows = []
        for _ in range(16):
            _, action, log_prob, value = model.act(observation)
            rows.append({"observation": observation, "action": action, "logProb": float(log_prob),
                         "value": float(value), "nextValue": 0, "reward": 1,
                         "terminated": True, "ended": True})
        result = update(model, optimizer, rows, epochs=1)
        self.assertIsNone(result["valueExplainedVariance"])
        self.assertGreaterEqual(result["meanApproxKL"], -1e-6)


if __name__ == "__main__":
    unittest.main()
