import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
from torch import nn

from roboproof.motion_learning.model import MotionActorCritic, NETWORK_DEPTHS, export, validate_network_depth
from roboproof.motion_learning.td3 import ALGORITHM, ReplayBuffer, TD3Learner, TwinQ
from roboproof.motion_learning.checkpoint_limits import checkpoint_tensor_bytes, checkpoint_size_limit
from roboproof.motion_learning.train import atomic_json, compatible_config

ROOT = Path(__file__).resolve().parents[1]


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(entry) for entry in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(entry) for entry in value)
    return 0


class MotionDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_prechange_two_layer_state_rng_and_export_goldens_remain_exact(self):
        reference = json.loads((ROOT / "tests/fixtures/motion-depth-two-layer-reference.json").read_text())
        if torch.__version__ != reference["torch"]:
            self.skipTest("Frozen CPU initialization reference requires its original Torch runtime")
        for case in reference["cases"]:
            torch.manual_seed(reference["seed"])
            model = MotionActorCritic("broad-pace-v1", case["featureTransform"], case["valueCritic"], case["width"])
            state_hash = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                state_hash.update(name.encode())
                state_hash.update(str(tuple(tensor.shape)).encode())
                state_hash.update(tensor.detach().numpy().tobytes())
            self.assertEqual(state_hash.hexdigest(), case["stateSha256"])
            self.assertEqual(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(), case["rngSha256"])
            payload = export(model, {"synthetic": "two-layer-reference-only"}, reference["seed"], 0, 0)
            self.assertNotIn("hiddenLayers", payload)
            self.assertEqual(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), case["exportSha256"])

    def test_depth_is_explicit_bounded_and_defaults_to_two(self):
        self.assertEqual(NETWORK_DEPTHS, (2, 10))
        for invalid in (None, True, False, 1, 3, 11, 0, -1, "10", 10.0):
            with self.assertRaises(ValueError):
                validate_network_depth(invalid)
            with self.assertRaises(ValueError):
                MotionActorCritic(network_depth=invalid)
        self.assertEqual(MotionActorCritic().network_depth, 2)

    def test_actor_and_separate_critics_have_requested_depth_at_every_width(self):
        for width in (32, 64, 128, 256):
            for depth in NETWORK_DEPTHS:
                model = MotionActorCritic("broad-pace-v1", "deadline-context-v1", True, width, depth)
                self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in model.body), depth)
                self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in model.critic_body), depth)
                critics = TwinQ(model.scales, width, depth)
                self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in critics.first), depth + 1)
                self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in critics.second), depth + 1)
                payload = export(model, {}, 41, 0, 0)
                self.assertEqual(len(payload["layers"]), depth + 1)
                self.assertEqual(payload.get("hiddenLayers", 2), depth)
        model = MotionActorCritic("broad-pace-v1", "deadline-context-v1", False, 256, 10)
        self.assertEqual(sum(parameter.numel() for parameter in model.body.parameters()) +
                         sum(parameter.numel() for parameter in model.actor.parameters()), 631560)

    def test_deep_td3_update_and_checkpoint_budget_are_genuine_but_not_movement_proof(self):
        actor = MotionActorCritic("broad-pace-v1", "deadline-context-v1", False, 256, 10)
        learner = TD3Learner(actor)
        replay = ReplayBuffer(len(actor.scales), 64)
        replay.append([{"modelInput": [index / 100] * len(actor.scales),
            "nextModelInput": [(index + 1) / 100] * len(actor.scales), "action": [0.3, -0.1, 0.2, -0.3],
            "reward": index / 10, "terminal": index % 8 == 0, "requiredPace": 0.95, "horizon": 1}
            for index in range(32)])
        for _ in range(4):
            learner.update(replay.sample(8))
        self.assertEqual(learner.gradient_updates, 4)
        self.assertEqual(learner.actor_updates, 2)
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in actor.parameters()))
        state = {"model": actor.state_dict(), "learner": learner.state(), "replay": replay.state()}
        config = {"algorithm": ALGORITHM, "networkWidth": 256, "networkDepth": 10,
                  "featureTransform": "deadline-context-v1", "replayCapacity": 64, "demonstrationAnchorWeight": 0}
        self.assertLessEqual(tensor_bytes(state), checkpoint_tensor_bytes(config))
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "checkpoint.pt"
            torch.save(state, filename)
            self.assertLess(filename.stat().st_size, checkpoint_size_limit(config))
            restored = torch.load(filename, weights_only=True, map_location="cpu")
            fresh = TD3Learner(MotionActorCritic("broad-pace-v1", "deadline-context-v1", False, 256, 10))
            fresh.restore(restored["learner"])
            self.assertEqual(fresh.gradient_updates, 4)
            shallow = TD3Learner(MotionActorCritic("broad-pace-v1", "deadline-context-v1", False, 256))
            with self.assertRaisesRegex(ValueError, "model states"):
                shallow.restore(restored["learner"])

    def test_depth_change_cannot_resume_as_the_same_ppo_method(self):
        base = {"updates": 4, "maxSteps": 1000, "networkWidth": 256,
                "rolloutSteps": 32, "seed": 42, "seedCount": 1, "algorithm": "ppo-beta-reference-v1"}
        self.assertTrue(compatible_config(base, {**base, "networkDepth": 2}))
        self.assertFalse(compatible_config(base, {**base, "networkDepth": 10}))
        with self.assertRaises(ValueError):
            checkpoint_size_limit({"networkDepth": 3})

    def test_largest_deep_export_and_public_history_match_javascript(self):
        result = subprocess.run(["node", "-e", "console.log(JSON.stringify(require('./simulator/headless').loadHeadless().identity))"],
                                cwd=ROOT, capture_output=True, text=True, check=True, timeout=30)
        torch.manual_seed(734)
        actor = MotionActorCritic("broad-pace-v1", "long-history-deadline-v1", False, 256, 10)
        with torch.no_grad():
            actor.actor.weight.uniform_(-0.2, 0.2)
            actor.actor.bias.uniform_(-0.3, 0.3)
        payload = export(actor, json.loads(result.stdout), 734, 0, 0, ALGORITHM)
        self.assertEqual(sum(len(layer["bias"]) + sum(len(row) for row in layer["weight"]) for layer in payload["layers"]), 739080)
        observations, expected = [], []
        actor.reset_policy_state(4)
        for tick in range(8):
            raw = [math.sin(index * 0.3 + tick * 0.2) * 0.2 for index in range(34)]
            raw[18], raw[21], raw[25] = 8, 0.4, tick * 0.3
            observations.append(raw)
            expected.append(actor.act(raw, deterministic=True)[0])
        with tempfile.TemporaryDirectory() as directory:
            policy_file = Path(directory) / "policy.json"
            request_file = Path(directory) / "request.json"
            atomic_json(policy_file, payload, actor=True)
            self.assertGreater(policy_file.stat().st_size, 8 * 1024 * 1024)
            self.assertLess(policy_file.stat().st_size, 24 * 1024 * 1024)
            request_file.write_text(json.dumps({"path": str(policy_file), "observations": observations}))
            program = "const fs=require('node:fs');const input=JSON.parse(fs.readFileSync(process.argv[1]));const model=require('./roboproof/motion-learner').policy(input.path);model.act.reset({referenceDurationSeconds:4});console.log(JSON.stringify(input.observations.map(raw=>model.act(raw))));"
            actual = json.loads(subprocess.run(["node", "-e", program, str(request_file)], cwd=ROOT,
                capture_output=True, text=True, check=True, timeout=60).stdout)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
