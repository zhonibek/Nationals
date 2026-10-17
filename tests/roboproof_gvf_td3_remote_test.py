"""Synthetic gradient checks only; never import Torch on unapproved/local hosts."""

import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(sys.platform == "linux" and os.environ.get("ROBOPROOF_APPROVED_REMOTE") == "1",
                     "Requires approved Linux remote; no local Torch import or model execution")
class GvfTD3RemoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, learner
        import torch
        from roboproof.motion_learning import gvf_td3 as learner

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
