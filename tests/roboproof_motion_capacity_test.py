import copy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.distributions import Beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_learning.imitation import MeasuredAnchor, POSE_BUDGET_SCALES, imitation_update
from roboproof.motion_learning.model import (
    AUGMENTED_FEATURES, DEADLINE_CONTEXT, DEADLINE_FEATURES, FEATURE_TRANSFORMS, SCALES,
    TEMPORAL_FEATURES, MotionActorCritic, export, normalization_definition,
)
from roboproof.motion_learning.td3 import ALGORITHM, ReplayBuffer, TD3Learner, TwinQ, clipped_target


class OriginalActorCritic(nn.Module):
    def __init__(self, initialization, feature_transform, value_critic):
        super().__init__()
        self.feature_transform = feature_transform
        self.history_frames = 16 if feature_transform == "long-history-deadline-v1" else (
            4 if feature_transform in TEMPORAL_FEATURES else 0)
        scales = list(SCALES)
        if feature_transform != "identity-v1":
            scales[21] = 1
            scales.append(1)
        if feature_transform == "settling-normalized-deadline-v1":
            for index, scale in {3: 0.0254, 4: 0.0254, 5: 0.0873, 10: 0.02032, 11: 0.02032, 12: 0.035}.items():
                scales[index] = scale
        if feature_transform in AUGMENTED_FEATURES:
            scales.extend([1] * 6)
        if self.history_frames:
            scales = scales * self.history_frames + SCALES[10:14]
        if feature_transform in DEADLINE_FEATURES:
            scales.append(1)
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.body = nn.Sequential(nn.Linear(len(scales), 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.actor = nn.Linear(32, 8)
        self.critic_body = nn.Sequential(nn.Linear(len(scales), 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()) if value_critic else None
        self.critic = nn.Linear(32, 1) if value_critic else None
        nn.init.zeros_(self.actor.weight)
        biases = [18, 19, 19, 19, -2, 19, 19, 19] if initialization == "conservative-v1" else [
            math.log(math.expm1(3.5)), 59, 59, 59, math.log(math.expm1(0.5)), 59, 59, 59]
        with torch.no_grad():
            self.actor.bias.copy_(torch.tensor(biases, dtype=torch.float32))
        self.initialization = initialization

    def forward(self, observations, encoded=True):
        normalized = torch.clamp(observations / self.scales, -10, 10)
        concentrations = torch.nn.functional.softplus(self.actor(self.body(normalized))) + 1
        distribution = Beta(concentrations[..., :4], concentrations[..., 4:])
        value = self.critic(self.critic_body(normalized)).squeeze(-1) if self.critic is not None else torch.zeros_like(concentrations[..., 0])
        return distribution, value


class OriginalTwinQ(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.register_buffer("scales", scales.detach().clone())
        inputs = len(scales) + 4
        self.first = nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.second = nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, observations, actions):
        inputs = torch.cat((torch.clamp(observations / self.scales, -10, 10), actions), dim=-1)
        return self.first(inputs).squeeze(-1), self.second(inputs).squeeze(-1)


def original_export(model, algorithm):
    def layer(module):
        return {"weight": module.weight.detach().tolist(), "bias": module.bias.detach().tolist()}
    return {"schemaVersion": 1, "algorithm": algorithm, "policyContractVersion": 1,
            "observationSize": 34, "actionSize": 4, "width": 32, "scales": model.scales.detach().tolist(),
            "featureTransform": model.feature_transform, "featureInputSize": len(model.scales),
            **({"historyFrames": model.history_frames, "initialContextSize": 4} if model.feature_transform in TEMPORAL_FEATURES else {}),
            **({"deadlineContext": dict(DEADLINE_CONTEXT)} if model.feature_transform in DEADLINE_FEATURES else {}),
            **({"normalizationDefinition": normalization_definition(model.feature_transform)}
               if normalization_definition(model.feature_transform) is not None else {}),
            **({"frameInputSize": 41} if model.feature_transform in AUGMENTED_FEATURES else {}),
            "layers": [layer(model.body[0]), layer(model.body[2]), layer(model.actor)],
            "identity": {"id": "capacity-math"}, "trainingSeed": 117001, "steps": 16, "updates": 2,
            "trained": True, "initialization": model.initialization, "inference": "deterministic Beta mean; bounded reference actions",
            "normalization": "fixed engineering-unit scales; no evaluation-fitted statistics"}


def populated_replay(actor):
    replay = ReplayBuffer(len(actor.scales), 64)
    replay.append([{"modelInput": [(index + axis % 7) / 100 for axis in range(len(actor.scales))],
                   "nextModelInput": [(index + 1 + axis % 7) / 100 for axis in range(len(actor.scales))],
                   "action": [0.1, -0.2, 0.05, -0.1], "reward": index / 20 - 1.5,
                   "terminal": index % 16 == 0, "requiredPace": 0.95, "horizon": index % 4 + 1}
                  for index in range(64)])
    return replay


def populated_anchor(actor):
    anchor = MeasuredAnchor(len(actor.scales))
    anchor.append([{"modelInput": [index / 100] * len(actor.scales), "teacherAction": [0.4, -0.2, 0.1, -0.3]}
                   for index in range(40)])
    return anchor


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: cpu_state(item) for name, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


class MotionCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_state_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for name in expected:
                self.assert_state_equal(actual[name], expected[name])
        elif isinstance(expected, (list, tuple)):
            self.assertIs(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for current, original in zip(actual, expected):
                self.assert_state_equal(current, original)
        else:
            self.assertEqual(actual, expected)

    def test_default32_exact_initialization_rng_forward_and_json_against_original_architecture(self):
        for profile in FEATURE_TRANSFORMS:
            for initialization in ("conservative-v1", "broad-pace-v1"):
                for value_critic in (False, True):
                    torch.manual_seed(117001)
                    original = OriginalActorCritic(initialization, profile, value_critic)
                    expected_rng = torch.get_rng_state()
                    for width_arguments in ({}, {"network_width": 32}):
                        with self.subTest(profile=profile, initialization=initialization, value_critic=value_critic, width=width_arguments):
                            torch.manual_seed(117001)
                            actor = MotionActorCritic(initialization, profile, value_critic, **width_arguments)
                            self.assertEqual(actor.network_width, 32)
                            self.assert_state_equal(actor.state_dict(), original.state_dict())
                            self.assert_state_equal(torch.get_rng_state(), expected_rng)
                            reference = copy.deepcopy(original)
                            for trained_weights in (False, True):
                                if trained_weights:
                                    with torch.no_grad():
                                        weights = torch.linspace(-0.2, 0.3, 8 * 32).reshape(8, 32)
                                        actor.actor.weight.copy_(weights)
                                        reference.actor.weight.copy_(weights)
                                inputs = torch.linspace(-20, 20, 3 * len(actor.scales)).reshape(3, -1)
                                actual, values = actor(inputs, encoded=True)
                                expected, original_values = reference(inputs)
                                for current, matching in ((actual.concentration1, expected.concentration1),
                                                          (actual.concentration0, expected.concentration0), (values, original_values)):
                                    self.assert_state_equal(current, matching)
                                algorithm = "ppo-beta-reference-v1" if value_critic else ALGORITHM
                                payload = export(actor, {"id": "capacity-math"}, 117001, 16, 2, algorithm)
                                self.assertEqual(json.dumps(payload), json.dumps(original_export(reference, algorithm)))
                            self.assert_state_equal(torch.get_rng_state(), expected_rng)

    def test_default32_twin_q_and_target_noise_preserve_exact_values_and_cpu_rng(self):
        actor = MotionActorCritic("broad-pace-v1", value_critic=False)
        torch.manual_seed(117002)
        original = OriginalTwinQ(actor.scales)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(117002)
        critics = TwinQ(actor.scales)
        self.assert_state_equal(critics.state_dict(), original.state_dict())
        self.assert_state_equal(torch.get_rng_state(), expected_rng)
        replay = populated_replay(actor)
        torch.manual_seed(117003)
        indices = torch.randint(replay.count, (16,))
        expected_rng = torch.get_rng_state()
        torch.manual_seed(117003)
        batch = replay.sample(16)
        self.assert_state_equal(torch.get_rng_state(), expected_rng)
        for name, tensor in batch.items():
            self.assert_state_equal(tensor, getattr(replay, name)[indices])
        self.assert_state_equal(critics(batch["observations"], batch["actions"]),
                                original(batch["observations"], batch["actions"]))
        torch.manual_seed(117004)
        noise = (torch.randn_like(batch["actions"]) * 0.05).clamp(-0.1, 0.1)
        expected_rng = torch.get_rng_state()
        with torch.no_grad():
            distribution, _ = actor(batch["next_observations"], encoded=True)
            actions = (distribution.mean * 2 - 1 + noise).clamp(-1, 1)
            first, second = original(batch["next_observations"], actions)
            targets = batch["rewards"] + torch.pow(0.99, batch["horizons"].float()) * (~batch["terminals"]).float() * torch.minimum(first, second)
        torch.manual_seed(117004)
        actual_targets, actual_actions = clipped_target(actor, critics, batch, 0.99, 0.05, 0.1)
        self.assert_state_equal((actual_targets, actual_actions), (targets, actions))
        self.assert_state_equal(torch.get_rng_state(), expected_rng)

    def test_widths_keep_feature_contract_shapes_and_eight_unmasked_outputs(self):
        sizes = {"identity-v1": 34, "reference-frame-v1": 35, "history-context-v1": 144,
                 "deadline-context-v1": 145, "long-history-deadline-v1": 565,
                 "settling-normalized-deadline-v1": 145, "coarse-augmentation-deadline-v1": 169,
                 "multiscale-deadline-v1": 169}
        for width in (32, 64, 128, 256):
            for profile in FEATURE_TRANSFORMS:
                for value_critic in (False, True):
                    with self.subTest(width=width, profile=profile, value_critic=value_critic):
                        actor = MotionActorCritic("broad-pace-v1", profile, value_critic, width)
                        self.assertEqual(actor.network_width, width)
                        self.assertEqual(len(actor.scales), sizes[profile])
                        self.assertEqual(actor.body[0].weight.shape, (width, sizes[profile]))
                        self.assertEqual(actor.body[2].weight.shape, (width, width))
                        self.assertEqual(actor.actor.weight.shape, (8, width))
                        if value_critic:
                            self.assertEqual(actor.critic_body[0].weight.shape, (width, sizes[profile]))
                            self.assertEqual(actor.critic_body[2].weight.shape, (width, width))
                            self.assertEqual(actor.critic.weight.shape, (1, width))
                        else:
                            self.assertIsNone(actor.critic_body)
                            self.assertIsNone(actor.critic)
                        distribution, values = actor(torch.zeros(3, sizes[profile]), encoded=True)
                        self.assertEqual(distribution.mean.shape, (3, 4))
                        self.assertEqual(values.shape, (3,))
                        self.assertTrue(torch.isfinite(distribution.mean).all())
                        critics = TwinQ(actor.scales, width)
                        for branch in (critics.first, critics.second):
                            self.assertEqual(branch[0].weight.shape, (width, sizes[profile] + 4))
                            self.assertEqual(branch[2].weight.shape, (width, width))
                            self.assertEqual(branch[4].weight.shape, (1, width))
                        payload = export(actor, {}, 1, 0, 0)
                        self.assertEqual(payload["width"], width)
                        self.assertEqual(len(payload["layers"][0]["weight"]), width)
                        self.assertEqual(len(payload["layers"][2]["weight"]), 8)
                        with self.assertRaisesRegex(ValueError, "feature count"):
                            actor(torch.zeros(3, sizes[profile] + 1), encoded=True)
                        if profile in TEMPORAL_FEATURES:
                            with self.assertRaisesRegex(ValueError, "encoded observations"):
                                actor(torch.zeros(3, 34))

    def test_invalid_widths_fail_before_tensor_allocation_or_rng_consumption(self):
        scales = torch.tensor(SCALES)
        invalid_widths = (True, False, None, 0, -32, 16, 255, 257, 512, 32.0, 64.0, 128.0, 256.0,
                          "32", "256", [], torch.tensor(32))
        saved_rng = torch.get_rng_state()
        for width in invalid_widths:
            with self.subTest(width=width), patch("torch.tensor", side_effect=AssertionError("tensor allocation")), \
                    patch("torch.Tensor.clone", side_effect=AssertionError("buffer allocation")), \
                    patch("torch.nn.Linear", side_effect=AssertionError("network allocation")):
                with self.assertRaisesRegex(ValueError, "Network width"):
                    MotionActorCritic(network_width=width)
                with self.assertRaisesRegex(ValueError, "Network width"):
                    TwinQ(scales, width)
        self.assert_state_equal(torch.get_rng_state(), saved_rng)

    def test_default32_td3_updates_match_independent_original_networks_exactly(self):
        torch.manual_seed(117005)
        original = OriginalActorCritic("broad-pace-v1", "identity-v1", False)
        original.network_width = 32
        with patch("roboproof.motion_learning.td3.TwinQ", side_effect=lambda scales, width: OriginalTwinQ(scales)):
            reference = TD3Learner(original)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(117005)
        actor = MotionActorCritic("broad-pace-v1", value_critic=False)
        learner = TD3Learner(actor)
        self.assert_state_equal(torch.get_rng_state(), expected_rng)
        replay = populated_replay(actor)
        for iteration in range(2):
            critic_batch, actor_batch = replay.sample(16, 0.25), replay.sample(16)
            rng = torch.get_rng_state()
            expected = reference.update(critic_batch, actor_batch=actor_batch)
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            actual = learner.update(critic_batch, actor_batch=actor_batch)
            self.assertEqual(actual, expected)
            self.assertEqual(actual["actorUpdated"], iteration == 1)
            self.assert_state_equal(actor.state_dict(), original.state_dict())
            self.assert_state_equal(learner.state(), reference.state())
            self.assert_state_equal(torch.get_rng_state(), expected_rng)

    def test_all_eight_rows_have_gradients_and_serialized_checkpoint_continues_exactly_at_each_width(self):
        for width in (32, 64, 128, 256):
            with self.subTest(width=width):
                torch.manual_seed(117006)
                actor = MotionActorCritic("broad-pace-v1", "deadline-context-v1", False, width)
                learner = TD3Learner(actor)
                replay, anchor = populated_replay(actor), populated_anchor(actor)
                def advance(current, buffer, demonstrations):
                    return current.update(buffer.sample(16, 0.25), measured_anchor=demonstrations,
                        anchor_weight=1, anchor_profile="pose-budget-v2", policy_regularization="public-deadline-residual-v1",
                        pace_weight=4, actor_batch=buffer.sample(16))
                for _ in range(3):
                    advance(learner, replay, anchor)
                self.assertEqual((learner.gradient_updates, learner.actor_updates), (3, 1))
                checkpoint = {"model": actor.state_dict(), "learner": learner.state(), "replay": replay.state(),
                              "anchor": anchor.state(), "rng": torch.get_rng_state()}
                stream = io.BytesIO()
                torch.save(checkpoint, stream)
                stream.seek(0)
                saved = torch.load(stream, weights_only=True, map_location="cpu")
                before = actor.actor.weight.detach().clone()
                expected = advance(learner, replay, anchor)
                expected_rng = torch.get_rng_state()
                self.assertTrue(torch.all(actor.actor.weight.grad.abs().sum(-1) > 0))
                self.assertTrue(torch.all((actor.actor.weight.detach() - before).abs().sum(-1) > 0))
                self.assertGreater(float(actor.body[0].weight.grad.abs().sum()), 0)
                for model in (learner.critics, learner.target_critics):
                    self.assertEqual(model.network_width, width)
                    self.assertEqual(model.first[2].weight.shape, (width, width))
                self.assertEqual(learner.target_actor.network_width, width)
                restored_actor = MotionActorCritic("broad-pace-v1", "deadline-context-v1", False, width)
                restored_actor.load_state_dict(saved["model"])
                restored = TD3Learner(restored_actor)
                restored.restore(saved["learner"])
                restored_replay, restored_anchor = ReplayBuffer(len(actor.scales), 64), MeasuredAnchor(len(actor.scales))
                restored_replay.restore(saved["replay"])
                restored_anchor.restore(saved["anchor"])
                torch.set_rng_state(saved["rng"])
                self.assertEqual(advance(restored, restored_replay, restored_anchor), expected)
                self.assert_state_equal(restored_actor.state_dict(), actor.state_dict())
                self.assert_state_equal(restored.state(), learner.state())
                self.assert_state_equal(restored_replay.state(), replay.state())
                self.assert_state_equal(restored_anchor.state(), anchor.state())
                self.assert_state_equal(torch.get_rng_state(), expected_rng)
                for name in ("targetActor", "critics", "targetCritics"):
                    corrupted = copy.deepcopy(saved["learner"])
                    corrupted[name]["scales"][0] *= 2
                    with self.assertRaisesRegex(ValueError, "sensor scales"):
                        restored.restore(corrupted)
                wrong_shape = copy.deepcopy(saved["learner"])
                wrong_shape["critics"]["first.0.weight"] = wrong_shape["critics"]["first.0.weight"][:-1]
                with self.assertRaisesRegex(ValueError, "model states"):
                    restored.restore(wrong_shape)

    def test_largest_actual_float32_dense_export_passes_strict_javascript_decoder_and_history_parity(self):
        identity_result = subprocess.run(["node", "-e",
            "console.log(JSON.stringify(require('./simulator/headless').loadHeadless().identity));"],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=30)
        identity = json.loads(identity_result.stdout)
        actor = MotionActorCritic("broad-pace-v1", "long-history-deadline-v1", False, 256)
        generator = torch.Generator().manual_seed(117010)
        with torch.no_grad():
            for module in (actor.body[0], actor.body[2], actor.actor):
                for parameter in module.parameters():
                    parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.01)
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertTrue(torch.all(parameter != 0))
        payload = export(actor, identity, 117010, 16, 2, ALGORITHM)
        parameter_count = sum(len(layer["bias"]) + sum(len(row) for row in layer["weight"]) for layer in payload["layers"])
        self.assertEqual(parameter_count, 212744)
        encoded = (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.assertGreater(len(encoded), 4 * 1024 * 1024)
        self.assertLessEqual(len(encoded), 8 * 1024 * 1024)
        observations, expected = [], []
        actor.reset_policy_state(4)
        for tick in range(18):
            raw = [math.sin((index + 1) * 0.41 + tick * 0.13) * 0.35 for index in range(34)]
            raw[0], raw[1], raw[18], raw[21], raw[25] = 20, -20, 8, 0.6, tick * 0.17
            observations.append(raw)
            action, _, _, _ = actor.act(raw, deterministic=True)
            expected.append(action)
        script = "const assert=require('node:assert/strict');const fs=require('node:fs');" \
                 "const learner=require('./roboproof/motion-learner');" \
                 "assert.throws(()=>learner.readJson(process.argv[1],4*1024*1024),/size budget/);" \
                 "const actor=learner.policy(process.argv[1],process.argv[2]);" \
                 "const request=JSON.parse(fs.readFileSync(0,'utf8'));" \
                 "actor.act.reset({referenceDurationSeconds:4});" \
                 "const first=request.map(row=>actor.act(row));actor.act.reset({referenceDurationSeconds:4});" \
                 "const second=request.map(row=>actor.act(row));assert.deepEqual(first,second);" \
                 "console.log(JSON.stringify({bytes:fs.statSync(process.argv[1]).size,actions:first}));"
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "actor.json"
            filename.write_bytes(encoded)
            result = subprocess.run(["node", "-e", script, str(filename), hashlib.sha256(encoded).hexdigest()],
                cwd=ROOT, input=json.dumps(observations), capture_output=True, text=True, check=True, timeout=30)
        decoded = json.loads(result.stdout)
        self.assertEqual(decoded["bytes"], len(encoded))
        self.assertEqual(len(decoded["actions"]), len(expected))
        for actual, reference in zip(decoded["actions"], expected):
            self.assertEqual(len(actual), 4)
            for value, matching in zip(actual, reference):
                self.assertTrue(math.isfinite(value) and -1 <= value <= 1)
                self.assertAlmostEqual(value, matching, places=6)
        print(json.dumps({"actorCapacity": "actual-float32-export", "width": 256, "featureInputSize": 565,
                          "parameterCount": parameter_count, "bytes": len(encoded)}))

    def test_invalid_importance_weights_fail_before_transfers_updates_and_rng_draws(self):
        actor = MotionActorCritic("broad-pace-v1", value_critic=False, network_width=64)
        learner = TD3Learner(actor)
        replay = populated_replay(actor)
        batch, actor_batch = replay.sample(16, 0.25), replay.sample(16)
        saved, rng = copy.deepcopy(learner.state()), torch.get_rng_state()
        invalid = [({**batch, "importance_weights": value}, actor_batch) for value in (
            torch.zeros(16), torch.full((16,), math.nan), torch.full((16,), 2.1), torch.ones(15), torch.ones(16, dtype=torch.float64))]
        invalid.extend(((batch, None), (batch, batch)))
        with patch("torch.Tensor.to", side_effect=AssertionError("premature device transfer")):
            for critic_data, actor_data in invalid:
                with self.assertRaises(ValueError):
                    learner.update(critic_data, actor_batch=actor_data)
        self.assert_state_equal(learner.state(), saved)
        self.assert_state_equal(torch.get_rng_state(), rng)

    def test_anchor_loss_keeps_exact_cpu_sampling_and_independent_weighted_math(self):
        actor = MotionActorCritic("broad-pace-v1", value_critic=False, network_width=128)
        anchor = populated_anchor(actor)
        for profile in ("uniform-action-v1", "pose-budget-v2"):
            torch.manual_seed(117007)
            indices = torch.randperm(len(anchor.features))[:32]
            expected_rng = torch.get_rng_state()
            distribution, _ = actor(anchor.features[indices], encoded=True)
            scales = torch.tensor(POSE_BUDGET_SCALES if profile == "pose-budget-v2" else (1, 1, 1, 1), dtype=torch.float32)
            expected = ((distribution.mean * 2 - 1 - anchor.targets[indices]) * scales).square().mean()
            torch.manual_seed(117007)
            self.assert_state_equal(anchor.loss(actor, profile), expected)
            self.assert_state_equal(torch.get_rng_state(), expected_rng)
        self.assertEqual(anchor.features.device.type, "cpu")
        self.assertEqual(anchor.targets.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; no GPU execution claimed")
    def test_cuda_td3_transfer_cpu_history_anchor_rng_and_cpu_checkpoint_restore(self):
        torch.manual_seed(117008)
        actor = MotionActorCritic("broad-pace-v1", "multiscale-deadline-v1", False, 64)
        reference = copy.deepcopy(actor)
        actor = actor.to("cuda")
        for model in (actor, reference):
            model.reset_policy_state(5)
        for index in range(6):
            raw = [index / 100] * 34
            raw[18], raw[25] = 8, index / 20
            features = actor.encode_observation(raw)
            self.assert_state_equal(features, reference.encode_observation(raw))
            self.assertEqual(features.device.type, "cpu")
            self.assertTrue(all(frame.device.type == "cpu" for frame in actor.observation_history))
            self.assertEqual(actor.initial_context.device.type, "cpu")
        distribution, values = actor(features, encoded=True)
        self.assertEqual(distribution.mean.device.type, "cuda")
        self.assertEqual(values.device.type, "cuda")
        learner = TD3Learner(actor, policy_delay=1)
        for model in (learner.actor, learner.target_actor, learner.critics, learner.target_critics):
            self.assertTrue(all(tensor.device.type == "cuda" for tensor in model.state_dict().values()))
        replay, anchor = populated_replay(actor), populated_anchor(actor)
        anchor.restore({**anchor.state(), "features": anchor.features.to("cuda"), "targets": anchor.targets.to("cuda")})
        self.assertEqual(anchor.features.device.type, "cpu")
        self.assertEqual(anchor.targets.device.type, "cpu")
        rows = [{"modelInput": feature.tolist(), "teacherAction": target.tolist()}
                for feature, target in zip(anchor.features[:16], anchor.targets[:16])]
        torch.manual_seed(117009)
        torch.randperm(len(rows))
        expected_rng = torch.get_rng_state()
        torch.manual_seed(117009)
        result = imitation_update(actor, learner.actor_optimizer, rows, epochs=1)
        self.assertTrue(math.isfinite(result["meanLoss"]))
        self.assert_state_equal(torch.get_rng_state(), expected_rng)
        learner.sync_actor_target()
        critic_batch, actor_batch = replay.sample(16, 0.25), replay.sample(16)
        actor_batch["observations"] = actor_batch["observations"] + 0.3
        rng = torch.get_rng_state()
        torch.randn_like(critic_batch["actions"])
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        cuda_rng = torch.cuda.get_rng_state()
        with patch.object(actor, "forward", wraps=actor.forward) as forward:
            result = learner.update(critic_batch, actor_batch=actor_batch)
            observations = forward.call_args.args[0]
            self.assertEqual(observations.device.type, "cuda")
            self.assert_state_equal(observations.cpu(), actor_batch["observations"])
        self.assertTrue(result["actorUpdated"])
        self.assertEqual((result["paceLoss"], result["anchorLoss"]), (0, 0))
        self.assert_state_equal(torch.get_rng_state(), expected_rng)
        self.assert_state_equal(torch.cuda.get_rng_state(), cuda_rng)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in critic_batch.values()))
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in replay.state().values() if isinstance(tensor, torch.Tensor)))
        result = learner.update(critic_batch, actor_batch=actor_batch, measured_anchor=anchor, anchor_weight=1,
            anchor_profile="pose-budget-v2", policy_regularization="public-deadline-residual-v1", pace_weight=4)
        self.assertTrue(math.isfinite(result["actorLoss"]))
        self.assertTrue(torch.all(actor.actor.weight.grad.abs().sum(-1) > 0))
        restored_actor = MotionActorCritic("broad-pace-v1", "multiscale-deadline-v1", False, 64).to("cuda")
        restored_actor.load_state_dict(cpu_state(actor.state_dict()))
        restored = TD3Learner(restored_actor, policy_delay=1)
        saved = cpu_state(learner.state())
        restored.restore(saved)
        self.assert_state_equal(cpu_state(restored.state()), saved)
        corrupted = copy.deepcopy(saved)
        corrupted["targetCritics"]["scales"][0] *= 2
        with self.assertRaisesRegex(ValueError, "sensor scales"):
            restored.restore(corrupted)
        self.assertTrue(math.isfinite(restored.update(critic_batch, actor_batch=actor_batch)["criticLoss"]))


if __name__ == "__main__":
    unittest.main()
