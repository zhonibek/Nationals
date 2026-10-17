"""Network checks are intentionally unavailable on unapproved/local hosts."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def observation():
    return {"contractName": "gvf-residual", "contractVersion": 2,
            "observationSchema": "gvf-residual-sensor34-v2",
            "pose": [0, 0, 0], "velocity": [0, 0, 0], "wheelRimSpeeds": [0] * 4,
            "targetError": [0] * 4, "previousAction": [0] * 3, "remainingSeconds": 5,
            "baseReference": [0] * 6, "projection": 0, "remainingArcMeters": 0,
            "crossTrackMeters": 0, "appliedFieldOffset": [0] * 3,
            "offsetVelocity": [0] * 3, "actionAgeTicks": 0}


@unittest.skipUnless(sys.platform == "linux" and os.environ.get("ROBOPROOF_APPROVED_REMOTE") == "1",
                     "Requires approved Linux remote; no local Torch import or model execution")
class GvfModelRemoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, model, contract
        import torch
        from roboproof.motion_learning import gvf_model as model, gvf_contract as contract

    def test_zero_forward_raw_encoding_and_all_architectures(self):
        for width in (32, 64, 128, 256, 512):
            for depth in (2, 10):
                actor = model.GvfActor(width, depth)
                features = actor.encode_observation(observation())
                self.assertEqual(features.shape, (36,))
                self.assertEqual(features.device.type, "cpu")
                self.assertEqual(actor.act(observation()), [0, 0, 0])
                output = actor(torch.zeros(2, 3, 36))
                self.assertEqual(output.shape, (2, 3, 3))
                self.assertEqual(torch.count_nonzero(output).item(), 0)
                self.assertEqual(sum(isinstance(layer, torch.nn.Linear) for layer in actor.layers), depth + 1)
                payload = model.export(actor, {"sourceHash": "remote-test-only"}, 19, 0, 0)
                self.assertEqual(len(payload["layers"]), depth + 1)
                critics = model.TwinQ(width, depth)
                for result in critics(torch.zeros(2, 3, 36), output):
                    self.assertEqual(result.shape, (2, 3, 1))
                    self.assertTrue(torch.isfinite(result).all())

    def test_invalid_features_actions_architecture_and_raw_tags(self):
        actor, critics = model.GvfActor(32), model.TwinQ(32)
        for value in ([0] * 34, torch.zeros(2, 35), [float("nan")] * 36,
                      [float("inf")] * 36, torch.ones(36, dtype=torch.bool), [True] + [0.0] * 35,
                      [[False] + [0.0] * 35]):
            with self.assertRaises(ValueError):
                actor(value)
        for action in (torch.zeros(2, 4), torch.zeros(3), torch.ones(2, 3) * 2,
                       torch.full((2, 3), float("nan"))):
            with self.assertRaises(ValueError):
                critics(torch.zeros(2, 36), action)
        with self.assertRaises(ValueError):
            critics(torch.zeros(36), [True, 0.0, 0.0])
        for width, depth in ((True, 2), (33, 2), (32, 3), (32, 2.0)):
            with self.assertRaises(ValueError):
                model.GvfActor(width, depth)
        for change in ({"contractName": "policy"}, {"contractVersion": 1},
                       {"truthPose": [0] * 3}, {"previousAction": [0] * 4}):
            with self.assertRaises(ValueError):
                actor.act({**observation(), **change})

    def test_normalization_bounded_inference_and_independent_clones(self):
        actor, critics = model.GvfActor(32), model.TwinQ(32)
        baseline, target = copy.deepcopy(actor), copy.deepcopy(critics)
        for original, cloned in ((actor, baseline), (critics, target)):
            for name, state in original.state_dict().items():
                self.assertNotEqual(state.data_ptr(), cloned.state_dict()[name].data_ptr())
        self.assertNotEqual(critics.first[0].weight.data_ptr(), critics.second[0].weight.data_ptr())
        self.assertFalse(torch.equal(critics.first[0].weight, critics.second[0].weight))
        with torch.no_grad():
            actor.layers[-2].weight.fill_(0.1)
            actor.layers[-2].bias.fill_(0.5)
            critics.first[0].weight.add_(1)
        self.assertEqual(baseline.act(observation()), [0, 0, 0])
        self.assertFalse(torch.equal(actor.layers[-2].bias, baseline.layers[-2].bias))
        self.assertFalse(torch.equal(critics.first[0].weight, target.first[0].weight))
        features = torch.tensor(contract.SCALES) * 100
        torch.testing.assert_close(actor(features), actor.layers(torch.full((36,), 10.0)))
        self.assertTrue(all(-1 <= action <= 1 for action in actor.act(observation())))

    def test_export_exact_tags_validation_and_atomic_failure(self):
        actor, identity = model.GvfActor(32), {"sourceHash": "remote-test-only"}
        payload = model.export(actor, identity, 19, 0, 0)
        header = contract.artifact_header(identity, 19, 0, 0, 32, 2)
        self.assertEqual(set(payload), set(header) | {"layers"})
        self.assertEqual({key: payload[key] for key in header}, header)
        self.assertEqual(payload["algorithm"], "td3-tanh-gvf-residual-v1")
        self.assertFalse(payload["trained"])
        self.assertTrue(model.export(actor, identity, 19, 7, 1)["trained"])
        with self.assertRaises(ValueError):
            model.export(actor, identity, True, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.json"
            model.atomic_actor_json(path, payload)
            self.assertEqual(json.loads(path.read_text()), payload)
            before = path.read_bytes()
            with patch.object(model.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    model.atomic_actor_json(path, payload)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with patch.object(model.json.JSONEncoder, "iterencode", return_value=iter([" " * (8 * 1024 * 1024 + 1)])):
                with self.assertRaises(ValueError):
                    model.atomic_actor_json(path, payload)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])
        with torch.no_grad():
            actor.layers[-2].weight[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model.export(actor, identity, 19, 0, 0)

    def test_nonzero_synthetic_actor_export_matches_javascript_inference(self):
        torch.manual_seed(1147)
        samples = []
        for index in range(8):
            sensor = observation()
            sensor.update(pose=[0.1 * index, -0.03 * index, 0.4 * index], velocity=[0.02, -0.01, 0.04],
                          targetError=[0.2, -0.3, 0.5, 0.8660254037844386],
                          baseReference=[0.05 * index, 0.01, -0.2 * index, 0.1, 0.02, -0.1])
            samples.append(sensor)
        for width, depth in ((32, 2), (128, 10)):
            actor = model.GvfActor(width, depth)
            with torch.no_grad():
                actor.layers[-2].weight.uniform_(-0.1, 0.1)
                actor.layers[-2].bias.uniform_(-0.1, 0.1)
            payload = model.export(actor, {"scope": "synthetic remote arithmetic test, not a trained actor"}, 19, 0, 0)
            expected = [actor.act(sensor) for sensor in samples]
            request = {"artifact": payload, "observations": samples}
            program = "const fs=require('node:fs');const input=JSON.parse(fs.readFileSync(0,'utf8'));const model=require('./roboproof/motion-gvf-learner').fromArtifact(input.artifact,input.artifact.identity);console.log(JSON.stringify(input.observations.map(value=>model.act(value))));"
            completed = subprocess.run(["node", "-e", program], input=json.dumps(request), text=True,
                                       capture_output=True, check=True, timeout=30, cwd=ROOT)
            actual = json.loads(completed.stdout)
            self.assertEqual(len(actual), len(expected))
            for observed, predicted in zip(actual, expected):
                self.assertEqual(len(observed), 3)
                self.assertLessEqual(max(abs(left - right) for left, right in zip(observed, predicted)), 1e-5)


if __name__ == "__main__":
    unittest.main()
