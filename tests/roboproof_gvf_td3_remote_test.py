"""Synthetic gradient checks only; never import Torch on unapproved/local hosts."""

import copy
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(sys.platform == "linux" and os.environ.get("ROBOPROOF_APPROVED_REMOTE") == "1",
                     "Requires approved Linux remote; no local Torch import or model execution")
class GvfTD3RemoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, learner, trainer
        import torch
        from roboproof.motion_learning import gvf_td3 as learner
        from roboproof.motion_learning import gvf_train as trainer

    def setUp(self):
        torch.manual_seed(73019)
        self.core = learner.GvfTD3(32)
        self.batch = (torch.randn(8, 36), torch.rand(8, 3) * 2 - 1, torch.randn(8),
                      torch.randn(8, 36), torch.tensor([True, False] * 4))

    def assert_tree_equal(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        if isinstance(actual, dict):
            self.assertEqual(set(actual), set(expected))
            for name in actual:
                self.assert_tree_equal(actual[name], expected[name])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_tree_equal(left, right)
        elif isinstance(actual, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
            self.assertTrue(torch.equal(actual, expected))
        else:
            self.assertEqual(actual, expected)

    def test_critic_gradients_actor_delay_and_polyak_only_every_second_update(self):
        before = self.core.state_dict()
        first = self.core.update(*self.batch)
        after = self.core.state_dict()
        self.assertEqual((first["updates"], first["actor_updates"]), (1, 0))
        self.assertIsNone(first["actor_loss"])
        for name in ("actor", "actor_target", "critics_target"):
            self.assert_tree_equal(before["models"][name], after["models"][name])
        for prefix in ("first.", "second."):
            self.assertTrue(any(not torch.equal(value, after["models"]["critics"][name])
                for name, value in before["models"]["critics"].items() if name.startswith(prefix)))
        second = self.core.update(*self.batch)
        self.assertEqual((second["updates"], second["actor_updates"]), (2, 1))
        self.assertIsInstance(second["actor_loss"], float)
        self.assertTrue(any(not torch.equal(value, self.core.actor.state_dict()[name])
                            for name, value in before["models"]["actor"].items()))
        for source, destination, name in ((self.core.actor, self.core.actor_target, "actor_target"),
                                          (self.core.critics, self.core.critics_target, "critics_target")):
            for key, value in destination.named_parameters():
                expected = after["models"][name][key] * (1 - self.core.config["tau"]) + (
                    source.state_dict()[key] * self.core.config["tau"])
                torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-8)
        for model in (self.core.actor, self.core.critics):
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters()))
        third_before = self.core.state_dict()
        self.core.update(*self.batch)
        third_after = self.core.state_dict()
        for name in ("actor", "actor_target", "critics_target"):
            self.assert_tree_equal(third_before["models"][name], third_after["models"][name])

    def test_independent_targets_frozen_before_updates_and_after_restore(self):
        for source, target in ((self.core.actor, self.core.actor_target),
                               (self.core.critics, self.core.critics_target)):
            for original, cloned in zip(source.parameters(), target.parameters()):
                self.assertNotEqual(original.data_ptr(), cloned.data_ptr())
                self.assertTrue(torch.equal(original, cloned))
                self.assertFalse(cloned.requires_grad)
                self.assertIsNone(cloned.grad)
        self.assertNotEqual(self.core.critics.first[0].weight.data_ptr(),
                            self.core.critics.second[0].weight.data_ptr())
        for _ in range(2):
            self.core.update(*self.batch)
        self.core.load_state_dict(self.core.state_dict())
        for target in (self.core.actor_target, self.core.critics_target):
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                                for parameter in target.parameters()))

    def test_native_reward_terminal_deadline_fault_truncation_and_twin_min(self):
        with torch.no_grad():
            for parameter in self.core.critics_target.parameters():
                parameter.zero_()
            self.core.critics_target.first[-1].bias.fill_(7)
            self.core.critics_target.second[-1].bias.fill_(2)
        rewards = torch.tensor([1.25, -2.5, -3.75, -4.0, 5.0])
        done = torch.tensor([True, True, True, True, False])
        batch = (self.batch[0][:5], self.batch[1][:5], rewards, self.batch[3][:5], done)
        before = rewards.clone()
        target = self.core._target(rewards, batch[3], done)
        expected = rewards + (~done).float() * self.core.config["gamma"] * 2
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
        torch.testing.assert_close(target[:4], rewards[:4], rtol=0, atol=0)
        self.assertFalse(target.requires_grad)
        first, second = self.core.critics(batch[0], batch[1])
        expected_loss = ((first.squeeze(-1) - expected).square().mean() +
                         (second.squeeze(-1) - expected).square().mean()).item()
        result = self.core.update(*batch)
        self.assertAlmostEqual(result["critic_loss"], expected_loss, places=6)
        self.assertAlmostEqual(result["target_mean"], expected.mean().item(), places=6)
        self.assertTrue(torch.equal(rewards, before))

    def test_smoothing_clips_noise_and_final_actions(self):
        captured = []
        handle = self.core.critics_target.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[1].detach().clone()))
        try:
            with torch.no_grad():
                self.core.actor_target.layers[-2].bias.copy_(torch.tensor([10.0, -10.0, 0.0]))
            noise = torch.tensor([[100.0, -100.0, 100.0]]).expand(8, 3)
            with patch.object(learner.torch, "randn_like", return_value=noise):
                self.core.update(*self.batch)
            expected = torch.tensor([[1.0, -1.0, 0.5]]).expand(8, 3)
            torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
        finally:
            handle.remove()

    def test_malformed_nonfinite_empty_mismatched_non_tensor_and_four_action_batches(self):
        cases = [(0, torch.zeros(0, 36)), (0, torch.zeros(8, 34)), (0, torch.zeros(8, 1, 36)),
                 (0, torch.zeros(36)), (0, torch.zeros(8, 36, dtype=torch.float64)),
                 (0, torch.zeros(8, 36, dtype=torch.bool)), (0, self.batch[0].tolist()),
                 (0, torch.zeros(8, 36, device="meta")),
                 (1, torch.zeros(8, 4)), (1, torch.zeros(7, 3)), (1, torch.full((8, 3), 1.01)),
                 (2, torch.zeros(8, 1)), (3, torch.zeros(7, 36)),
                 (4, torch.zeros(8)), (4, torch.zeros(8, 1, dtype=torch.bool))]
        for index in range(4):
            for bad in (float("nan"), float("inf"), -float("inf")):
                cases.append((index, torch.full_like(self.batch[index], bad)))
        original = self.core.state_dict()
        for index, value in cases:
            batch = list(self.batch)
            batch[index] = value
            with self.subTest(index=index, shape=getattr(value, "shape", None)), self.assertRaises(ValueError):
                self.core.update(*batch)
            self.assert_tree_equal(self.core.state_dict(), original)
        with self.assertRaises(ValueError):
            self.core.update(*(value[:0] for value in self.batch))

    def test_hyperparameter_ranges_and_explicit_supported_device(self):
        for name in ("actor_lr", "critic_lr", "gamma", "tau", "policy_noise", "noise_clip"):
            bad_values = (-0.1, 1.01, float("nan"), float("inf"), True, "0.1")
            if name in ("actor_lr", "critic_lr", "tau"):
                bad_values += (0,)
            for value in bad_values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    learner.GvfTD3(32, **{name: value})
        for device in ("mps", "cuda", "cpu:0"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                learner.GvfTD3(32, device=device)
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with self.assertRaisesRegex(ValueError, "default float32"):
                learner.GvfTD3(32)
        finally:
            torch.set_default_dtype(default_dtype)

    def test_failed_optimizer_cannot_continue_or_publish_a_partial_checkpoint(self):
        checkpoint = self.core.state_dict()
        with patch.object(self.core.critic_optimizer, "step", side_effect=RuntimeError("synthetic optimizer failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic optimizer failure"):
                self.core.update(*self.batch)
        with self.assertRaisesRegex(ValueError, "checkpoint restore"):
            self.core.update(*self.batch)
        with self.assertRaisesRegex(ValueError, "partial update"):
            self.core.state_dict()
        self.core.load_state_dict(checkpoint)
        self.assertEqual(self.core.update(*self.batch)["updates"], 1)

    def test_host_guard_precedes_tensor_operations_even_with_invalid_arguments(self):
        for platform, flag in (("win32", "1"), ("darwin", "1"), ("linux", "0"), ("linux", "")):
            with patch.object(learner.sys, "platform", platform), patch.dict(
                    learner.os.environ, {"ROBOPROOF_APPROVED_REMOTE": flag}):
                with self.assertRaises(RuntimeError):
                    self.core.update(None, None, None, None, None)
        self.assertEqual(self.core.updates, 0)

    def test_nonfinite_parameters_loss_and_gradients_throw(self):
        before = self.core.state_dict()
        with torch.no_grad():
            self.core.critics.first[0].weight[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.core.update(*self.batch)
        self.core.load_state_dict(before)
        hook = self.core.critics.first[0].weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("nan")))
        try:
            with self.assertRaises(ValueError):
                self.core.update(*self.batch)
            self.assertEqual(self.core.updates, 0)
            self.assert_tree_equal(self.core.critics.state_dict(), before["models"]["critics"])
        finally:
            hook.remove()
        self.core.load_state_dict(before)
        batch = list(self.batch)
        batch[2] = torch.full((8,), torch.finfo(torch.float32).max)
        with self.assertRaises(ValueError):
            self.core.update(*batch)
        self.assertEqual(self.core.updates, 0)

    def test_exact_continuation_restores_rng_adam_targets_and_odd_update_delay(self):
        for initial_updates in (0, 1, 2, 3):
            torch.manual_seed(73019)
            original = learner.GvfTD3(32)
            for _ in range(initial_updates):
                original.update(*self.batch)
            checkpoint = original.state_dict()
            expected_metrics = [original.update(*self.batch) for _ in range(3)]
            expected = original.state_dict()
            restored = learner.GvfTD3(32)
            torch.rand(17)
            restored.load_state_dict(checkpoint)
            actual_metrics = [restored.update(*self.batch) for _ in range(3)]
            self.assertEqual(actual_metrics, expected_metrics)
            self.assert_tree_equal(restored.state_dict(), expected)
            self.assertEqual(checkpoint["updates"], initial_updates)
            checkpoint["models"]["actor"]["layers.0.weight"].zero_()
            self.assertFalse(torch.equal(checkpoint["models"]["actor"]["layers.0.weight"],
                                         restored.actor.state_dict()["layers.0.weight"]))
            restored_before = restored.state_dict()
            checkpoint["optimizers"]["critics"]["state"].clear()
            self.assert_tree_equal(restored.state_dict(), restored_before)

    def test_remote_replay_checkpoint_restores_ring_cursor_and_rejects_corruption(self):
        source = trainer.ReplayBuffer(torch, capacity=8)
        for index in range(5):
            source.append({"features": [index / 10] * 36, "action": [0.1, -0.2, 0.3],
                "reward": float(index), "nextFeatures": [(index + 1) / 10] * 36, "done": index == 4})
        checkpoint = source.state_dict()
        restored = trainer.ReplayBuffer(torch, capacity=8)
        restored.load_state_dict(torch, checkpoint)
        self.assertEqual((restored.count, restored.cursor), (5, 5))
        self.assert_tree_equal(restored.state_dict(), checkpoint)
        malformed = copy.deepcopy(checkpoint)
        malformed["cursor"] = 4
        before = restored.state_dict()
        with self.assertRaisesRegex(ValueError, "capacity/cursor/count"):
            restored.load_state_dict(torch, malformed)
        self.assert_tree_equal(restored.state_dict(), before)
        malformed = copy.deepcopy(checkpoint)
        malformed["observations"][-1, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "Unused GVF replay rows"):
            restored.load_state_dict(torch, malformed)
        self.assert_tree_equal(restored.state_dict(), before)

    def test_remote_saved_gvf_run_checkpoint_restores_exact_bound_state(self):
        from roboproof.motion_learning.gvf_model import atomic_actor_json, export

        seed = 50515
        config = trainer.validate_limits(seed, 128, 8, 60, network_width=64, hidden_layers=2)
        device, gpu = trainer.verify_t4(torch)
        approval = {"provider": "Google Colab", "device": "free Tesla T4", "maxCredits": 0,
            "billingBasis": "free runtime only", "paymentMethodChargeAuthorized": False,
            "autoReloadAuthorized": False}
        identity = {"testOnly": True, "simulator": "original-engine-identity-binding"}
        source_bytes = b"verified test-only GVF source manifest\n"
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        run_id = "resume-restore-test"
        rng = random.Random(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        from roboproof.motion_learning.gvf_td3 import GvfTD3
        learner = GvfTD3(width=64, hidden_layers=2, device=device)
        replay = trainer.ReplayBuffer(torch)
        row = {"features": [0.0] * 36, "action": [0.0] * 3, "reward": 1.0,
            "nextFeatures": [0.1] * 36, "done": True}
        replay.append(row)
        world = {"seed": 700001, "task": {"start": {"xIn": 0, "yIn": 0},
            "goal": {"xIn": 10, "yIn": 10}}, "configuration": {}}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("actors", "episodes", "checkpoints"):
                (directory / name).mkdir()
            (directory / "source-manifest.json").write_bytes(source_bytes)
            host = {"checkedAtUtc": trainer.utc_now(), "system": platform.platform(),
                "python": platform.python_version(), "node": __import__("subprocess").check_output(
                    ["node", "--version"], text=True).strip(), "gpu": gpu, "gpuUsedForOptimizer": True}
            trainer.write_exclusive(directory / "host.json", host)
            initial_actor = export(learner.actor, identity, seed, 0, 0)
            initial_path = directory / "actors/actor-episode-000000.json"
            atomic_actor_json(initial_path, initial_actor)
            initial_receipt = {"path": "actors/actor-episode-000000.json",
                "sha256": trainer.digest(initial_path.read_bytes())}
            actor = export(learner.actor, identity, seed, 1, 0)
            actor_path = directory / "actors/actor-episode-000001.json"
            atomic_actor_json(actor_path, actor)
            actor_receipt = {"path": "actors/actor-episode-000001.json",
                "sha256": trainer.digest(actor_path.read_bytes())}
            episode = {"schemaVersion": 1, "episode": 1, "world": world, "complete": True,
                "decisionCount": 1, "transitions": [row]}
            episode_path = directory / "episodes/episode-000001.json"
            episode_sha = trainer.write_exclusive(episode_path, episode)
            exposure_started = {"event": "started", "episode": 1, "world": world,
                "sourceManifestSha256": source_sha}
            trainer.append_jsonl(directory / "exposures.jsonl", exposure_started)
            checkpoint_path, checkpoint_sha = trainer._save_checkpoint(torch, learner, replay, rng,
                directory, identity, source_sha, config, 1, 1, run_id, set(), set(), 0)
            trainer.append_jsonl(directory / "exposures.jsonl", {"event": "completed", "episode": 1,
                "world": world, "decisionCount": 1, "episodeFileSha256": episode_sha,
                "sourceManifestSha256": source_sha, "checkpointSha256": checkpoint_sha})
            checkpoint_receipt = {"path": "checkpoints/episode-000001.pt", "sha256": checkpoint_sha}
            episode_receipt = {"path": "episodes/episode-000001.json", "sha256": episode_sha}
            record = {"schemaVersion": 1, "runId": run_id, "config": config,
                "sourceManifestSha256": source_sha, "createdAtUtc": trainer.utc_now(), "resumeCount": 0,
                "operatorDeclaredApproval": approval, "knownExposureExclusion": None, "trainingDevice": gpu,
                "trainingDriver": "roboproof.motion_learning.gvf_train", "resumeSupported": True,
                "status": "time-budget-stopped", "engineIdentity": identity, "initialActor": initial_receipt,
                "actor": actor_receipt, "checkpoint": checkpoint_receipt, "lastEpisode": episode_receipt,
                "decisions": 1, "episodes": 1, "updates": 0, "trainingExposureRejections": 0,
                "elapsedSeconds": 1.0, "cumulativeElapsedSeconds": 1.0, "unresolvedExposure": None}
            trainer.write_exclusive(directory / "run.json", record)
            trainer.write_exclusive(directory / "terminal.json", record)
            trainer.write_exclusive(directory / "segments.jsonl", {"event": "stopped"})
            trainer.write_exclusive(directory / "transitions.jsonl", {"episode": 1, **row})
            restored_learner = GvfTD3(width=64, hidden_layers=2, device=device)
            restored_replay = trainer.ReplayBuffer(torch)
            restored_rng = random.Random(0)
            result = trainer._load_resume_state(torch, restored_learner, restored_replay, restored_rng,
                directory, run_id, config, source_bytes, source_sha, gpu, approval, None, identity)
            self.assertEqual((result["record"]["decisions"], result["record"]["episodes"],
                restored_learner.updates, restored_replay.count, restored_replay.cursor), (1, 1, 0, 1, 1))
            self.assertEqual(result["checkpoint"], checkpoint_receipt)
            self.assertTrue(torch.equal(restored_learner.actor.layers[0].weight, learner.actor.layers[0].weight))
            self.assertEqual(rng.random(), restored_rng.random())
            record["status"] = "signal-stopped"
            trainer.atomic_json(directory / "run.json", record)
            trainer.atomic_json(directory / "terminal.json", record)
            signal_resumed = trainer._load_resume_state(torch, GvfTD3(width=64, hidden_layers=2, device=device),
                trainer.ReplayBuffer(torch), random.Random(0), directory, run_id, config,
                source_bytes, source_sha, gpu, approval, None, identity)
            self.assertEqual(signal_resumed["record"]["status"], "signal-stopped")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                trainer._load_resume_state(torch, GvfTD3(width=64, hidden_layers=2, device=device),
                    trainer.ReplayBuffer(torch), random.Random(0), directory, run_id, config,
                    source_bytes, source_sha, gpu, approval, None, {"testOnly": True})

    def test_legacy_metadata_config_counters_and_malformed_state_are_rejected_atomically(self):
        for _ in range(2):
            self.core.update(*self.batch)
        checkpoint = self.core.state_dict()
        metadata_changes = dict(algorithm="td3-beta-mean-reference-v1", policyContractName="policy",
            policyContractVersion=1, observationSchema="legacy", observationSize=4, actionSize=4,
            featureTransform="legacy", featureInputSize=34, width=64, hiddenLayers=10,
            activation="relu", device="cuda:0", dtype="float64", runtime={"torch": "other", "cuda": None})
        malformed = [{"actor": {}}, {**checkpoint, "unexpected": 1}, {**checkpoint, "updates": True},
                     {**checkpoint, "actor_updates": 0}, {**checkpoint, "updates": -1}]
        for name, value in metadata_changes.items():
            state = copy.deepcopy(checkpoint)
            state["metadata"][name] = value
            malformed.append(state)
        changes = (
            lambda state: state["metadata"]["config"].__setitem__("gamma", 0.5),
            lambda state: state["metadata"]["config"].__setitem__("policy_delay", 1),
            lambda state: state["models"]["actor"].__setitem__("layers.0.weight", torch.zeros(32, 4)),
            lambda state: state["models"]["actor"].__setitem__("layers.0.weight", torch.zeros(32, 36).double()),
            lambda state: state["models"]["actor"]["scales"].fill_(1),
            lambda state: state["models"]["actor_target"]["layers.0.weight"].fill_(float("nan")),
            lambda state: state["optimizers"]["critics"]["param_groups"][0].__setitem__("lr", 0.5),
            lambda state: state["optimizers"]["critics"]["state"][0]["step"].fill_(1),
            lambda state: state["optimizers"]["actor"]["state"][0]["exp_avg"].fill_(float("inf")),
            lambda state: state["optimizers"]["actor"]["state"][0]["exp_avg_sq"].fill_(-1),
            lambda state: state["rng"].__setitem__("cpu", torch.zeros(2, dtype=torch.uint8)),
            lambda state: state["rng"].__setitem__("device", torch.zeros(2, dtype=torch.uint8)),
        )
        for change in changes:
            state = copy.deepcopy(checkpoint)
            change(state)
            malformed.append(state)
        for index, state in enumerate(malformed):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.core.load_state_dict(state)
            self.assert_tree_equal(self.core.state_dict(), checkpoint)


if __name__ == "__main__":
    unittest.main()
