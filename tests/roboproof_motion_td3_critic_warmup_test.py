import copy
import inspect
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_learning.imitation import MeasuredAnchor
from roboproof.motion_learning.model import MotionActorCritic
from roboproof.motion_learning.td3 import TD3Learner, mean_action, validate_critic_warmup_steps
from roboproof.motion_learning import td3_train


INVALID_WARMUPS = (True, False, -1, 513, 0.0, 1.0, 256.5, math.nan, math.inf, -math.inf, "256", None, [], {})
CRITIC_METRICS = {"criticLoss", "criticGradientNorm", "gradientUpdates", "actorUpdated",
                  "meanTarget", "maximumAbsoluteTargetAction"}


def configuration_arguments():
    defaults = {name: parameter.default for name, parameter in inspect.signature(td3_train.run).parameters.items()}
    arguments = {name: defaults[name] for name in inspect.signature(td3_train.configuration).parameters}
    arguments.update(seed=117001, updates=4, max_steps=3000, rollout_steps=16,
                     warm_start_updates=1, gradient_steps=2, replay_capacity=64)
    return arguments


def batch(actor, count=32):
    generator = torch.Generator().manual_seed(117002)
    observations = torch.randn(count, len(actor.scales), generator=generator) * actor.scales * 0.05
    return {"observations": observations, "next_observations": observations + actor.scales * 0.005,
            "actions": torch.rand(count, 4, generator=generator) - 0.5,
            "rewards": torch.linspace(-1, 2, count), "terminals": torch.arange(count) % 3 == 0,
            "required_paces": torch.full((count,), 0.95), "horizons": 1 + torch.arange(count) % 4}


def learner(warmup=0, delay=2):
    actor = MotionActorCritic("broad-pace-v1", value_critic=False)
    return TD3Learner(actor, critic_warmup_steps=warmup, policy_delay=delay, target_tau=0.1)


def anchor(actor):
    inputs = batch(actor, 48)
    result = MeasuredAnchor(len(actor.scales))
    result.restore({"schemaVersion": 1, "features": inputs["observations"], "targets": inputs["actions"]})
    return result


def snapshot(instance, metrics=()):
    return copy.deepcopy({"model": instance.actor.state_dict(), "learner": instance.state(),
                          "torchRng": torch.get_rng_state(), "metrics": list(metrics)})


class TD3CriticWarmupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(117003)
        for name in ("MotionEnvironment", "prepare_execution", "train_seed", "atomic_json"):
            guard = mock.patch.object(td3_train, name, side_effect=AssertionError("Only learner unit tests may run"))
            guard.start()
            self.addCleanup(guard.stop)

    def assert_exact(self, actual, expected, path="state"):
        self.assertIs(type(actual), type(expected), path)
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys(), path)
            for name, value in expected.items():
                self.assert_exact(actual[name], value, f"{path}.{name}")
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected), path)
            for index, (value, matching) in enumerate(zip(actual, expected)):
                self.assert_exact(value, matching, f"{path}[{index}]")
        else:
            self.assertEqual(actual, expected, path)

    def assert_polyak(self, actual, previous, current, tau):
        for name, value in actual.items():
            expected = previous[name] if name == "scales" else previous[name].mul(1 - tau).add(current[name], alpha=tau)
            torch.testing.assert_close(value, expected, rtol=0, atol=0, msg=name)

    def test_integer_bounds_and_invalid_values_reject_before_model_rng_or_run_side_effects(self):
        actor = MotionActorCritic("broad-pace-v1", value_critic=False)
        original = copy.deepcopy(actor.state_dict())
        random_state = torch.get_rng_state().clone()
        arguments = configuration_arguments()
        for value in (0, 1, 2, 255, 256, 511, 512):
            with self.subTest(accepted=value):
                self.assertEqual(validate_critic_warmup_steps(value), value)
                self.assertIs(type(validate_critic_warmup_steps(value)), int)
        for value in INVALID_WARMUPS:
            with self.subTest(rejected=value):
                with self.assertRaises(ValueError):
                    validate_critic_warmup_steps(value)
                with self.assertRaises(ValueError):
                    TD3Learner(actor, critic_warmup_steps=value)
                with self.assertRaises(ValueError):
                    td3_train.configuration(**{**arguments, "critic_warmup_steps": value})
                with self.assertRaises(ValueError):
                    td3_train.run(critic_warmup_steps=value)
                self.assert_exact(actor.state_dict(), original)
                self.assert_exact(torch.get_rng_state(), random_state)

    def test_default_schema_and_configuration_omit_warmup_and_opt_in_adds_only_declared_fields(self):
        for function in (TD3Learner, td3_train.configuration, td3_train.run):
            self.assertEqual(inspect.signature(function).parameters["critic_warmup_steps"].default, 0)
        arguments = configuration_arguments()
        omitted = {name: value for name, value in arguments.items() if name != "critic_warmup_steps"}
        baseline = td3_train.configuration(**omitted)
        self.assertEqual(baseline, td3_train.configuration(**arguments))
        self.assertNotIn("criticWarmupSteps", baseline)
        self.assertNotIn("criticWarmupDefinition", baseline)
        self.assertIsNone(td3_train.critic_warmup_definition(0))
        legacy_keys = {"schemaVersion", "targetActor", "critics", "targetCritics", "actorOptimizer",
                       "criticOptimizer", "gradientUpdates", "actorUpdates"}
        default = TD3Learner(MotionActorCritic("broad-pace-v1", value_critic=False))
        for instance in (default, learner(0)):
            self.assertEqual(instance.critic_warmup_steps, 0)
            self.assertEqual(instance.state()["schemaVersion"], 1)
            self.assertEqual(set(instance.state()), legacy_keys)
        for steps in (1, 256, 512):
            with self.subTest(warmup=steps):
                declared = td3_train.configuration(**{**arguments, "critic_warmup_steps": steps})
                self.assertEqual(set(declared) - set(baseline), {"criticWarmupSteps", "criticWarmupDefinition"})
                self.assertEqual({name: declared[name] for name in baseline}, baseline)
                self.assertEqual(declared["criticWarmupSteps"], steps)
                definition = declared["criticWarmupDefinition"]
                self.assertEqual(definition, td3_train.critic_warmup_definition(steps))
                self.assertEqual(definition["method"], "initial-native-critic-only-v1")
                self.assertEqual(definition["gradientSteps"], steps)
                self.assertEqual(set(definition), {"method", "gradientSteps", "timing", "actor", "targets", "budget", "scope"})
                state = learner(steps).state()
                self.assertEqual(set(state), legacy_keys | {"criticWarmupSteps"})
                self.assertEqual((state["schemaVersion"], state["criticWarmupSteps"]), (2, steps))

    def test_run_forwards_omitted_zero_and_positive_warmup_to_configuration_only(self):
        class ConfigurationReached(Exception):
            pass

        signature = inspect.signature(td3_train.configuration)
        for overrides in ({}, {"critic_warmup_steps": 0}, {"critic_warmup_steps": 1},
                          {"critic_warmup_steps": 256}, {"critic_warmup_steps": 512}):
            with self.subTest(overrides=overrides), \
                 mock.patch.object(td3_train, "configuration", side_effect=ConfigurationReached) as configure:
                with self.assertRaises(ConfigurationReached):
                    td3_train.run(**overrides)
                configure.assert_called_once()
                forwarded = signature.bind(*configure.call_args.args, **configure.call_args.kwargs)
                self.assertEqual(forwarded.arguments["critic_warmup_steps"], overrides.get("critic_warmup_steps", 0))

    def test_configuration_resume_binding_requires_identical_warmup_and_definition(self):
        arguments = configuration_arguments()
        baseline = td3_train.configuration(**arguments)
        enabled = td3_train.configuration(**{**arguments, "critic_warmup_steps": 256})
        self.assertTrue(td3_train.compatible_config(enabled, copy.deepcopy(enabled)))
        self.assertTrue(td3_train.compatible_config(enabled, {**enabled, "updates": 5, "maxSteps": 4000}))
        for changed in (baseline, {**enabled, "criticWarmupSteps": 255},
                        {**enabled, "criticWarmupDefinition": {}},
                        {name: value for name, value in enabled.items() if name != "criticWarmupDefinition"}):
            with self.subTest(changed=changed.get("criticWarmupSteps")):
                self.assertFalse(td3_train.compatible_config(enabled, changed))
                self.assertFalse(td3_train.compatible_config(changed, enabled))

    def test_critic_only_prefix_freezes_actor_adam_targets_and_skips_both_anchor_losses(self):
        for populated_adam in (False, True):
            with self.subTest(populated_adam=populated_adam):
                instance = learner(4)
                inputs = batch(instance.actor)
                if populated_adam:
                    for unused in range(2):
                        instance.actor_optimizer.zero_grad()
                        (mean_action(instance.actor, inputs["observations"]) - inputs["actions"]).square().mean().backward()
                        instance.actor_optimizer.step()
                    instance.sync_actor_target()
                    self.assertTrue(instance.actor_optimizer.state)
                initial = snapshot(instance)
                measured = mock.Mock()
                successful = mock.Mock(features=inputs["observations"])
                measured.loss.side_effect = AssertionError("Measured anchor loss is actor-only")
                successful.loss.side_effect = AssertionError("Successful anchor loss is actor-only")
                with mock.patch.object(instance.actor_optimizer, "step", wraps=instance.actor_optimizer.step) as actor_step, \
                     mock.patch.object(instance.actor_optimizer, "zero_grad", wraps=instance.actor_optimizer.zero_grad) as actor_zero, \
                     mock.patch.object(instance.critic_optimizer, "step", wraps=instance.critic_optimizer.step) as critic_step, \
                     mock.patch("roboproof.motion_learning.td3.control_prior_loss", side_effect=AssertionError("Prior is actor-only")):
                    for gradient_step in range(1, 5):
                        previous_critics = copy.deepcopy(instance.critics.state_dict())
                        random_state = torch.get_rng_state().clone()
                        torch.randn_like(inputs["actions"])
                        expected_rng = torch.get_rng_state().clone()
                        torch.set_rng_state(random_state)
                        result = instance.update(inputs, measured_anchor=measured, anchor_weight=3,
                                                 successful_anchor=successful, self_retention_weight=2,
                                                 policy_regularization="public-deadline-residual-v1", pace_weight=40)
                        self.assertEqual(set(result), CRITIC_METRICS)
                        self.assertFalse(result["actorUpdated"])
                        self.assertEqual((result["gradientUpdates"], instance.gradient_updates, instance.actor_updates),
                                         (gradient_step, gradient_step, 0))
                        self.assertGreater(result["criticLoss"], 0)
                        self.assertGreater(result["criticGradientNorm"], 0)
                        for branch in ("first.", "second."):
                            self.assertTrue(any(not torch.equal(value, previous_critics[name])
                                                for name, value in instance.critics.state_dict().items() if name.startswith(branch)))
                        self.assert_exact(instance.actor.state_dict(), initial["model"])
                        for name in ("actorOptimizer", "targetActor", "targetCritics"):
                            self.assert_exact(instance.state()[name], initial["learner"][name], name)
                        self.assert_exact(torch.get_rng_state(), expected_rng)
                        self.assertEqual(critic_step.call_count, gradient_step)
                        self.assertEqual({float(state["step"]) for state in instance.critic_optimizer.state.values()}, {float(gradient_step)})
                    actor_step.assert_not_called()
                    actor_zero.assert_not_called()
                    measured.loss.assert_not_called()
                    successful.loss.assert_not_called()

    def test_global_delay_release_updates_all_eight_outputs_and_only_then_polyak_targets(self):
        schedules = [(steps, delay) for steps in (1, 2, 3, 4) for delay in (1, 2, 3, 4)] + [(256, 2)]
        for steps, delay in schedules:
            with self.subTest(warmup=steps, delay=delay):
                torch.manual_seed(117004)
                instance = learner(steps, delay)
                inputs = batch(instance.actor)
                first_actor_step = (steps // delay + 1) * delay
                actor_steps = []
                for gradient_step in range(1, first_actor_step + 2 * delay + 1):
                    previous = snapshot(instance)
                    result = instance.update(inputs, policy_regularization="public-deadline-residual-v1", pace_weight=40)
                    expected_update = gradient_step > steps and gradient_step % delay == 0
                    expected_count = gradient_step // delay - min(gradient_step, steps) // delay
                    self.assertEqual(result["actorUpdated"], expected_update)
                    self.assertEqual((instance.gradient_updates, instance.actor_updates), (gradient_step, expected_count))
                    if expected_update:
                        actor_steps.append(gradient_step)
                        self.assertTrue(torch.all(instance.actor.actor.weight.grad.abs().sum(-1) > 0))
                        self.assertTrue(torch.all((instance.actor.actor.weight.detach() != previous["model"]["actor.weight"]).any(-1)))
                        self.assertTrue(torch.all(instance.actor.actor.bias.detach() != previous["model"]["actor.bias"]))
                        self.assertEqual({float(state["step"]) for state in instance.actor_optimizer.state.values()}, {float(expected_count)})
                        self.assert_polyak(instance.target_actor.state_dict(), previous["learner"]["targetActor"],
                                           instance.actor.state_dict(), instance.target_tau)
                        self.assert_polyak(instance.target_critics.state_dict(), previous["learner"]["targetCritics"],
                                           instance.critics.state_dict(), instance.target_tau)
                    else:
                        self.assertEqual(set(result), CRITIC_METRICS)
                        self.assert_exact(instance.actor.state_dict(), previous["model"])
                        for name in ("actorOptimizer", "targetActor", "targetCritics"):
                            self.assert_exact(instance.state()[name], previous["learner"][name], name)
                self.assertEqual(actor_steps, [first_actor_step, first_actor_step + delay, first_actor_step + 2 * delay])
                if steps == 256:
                    self.assertEqual(actor_steps[0], 258)
                self.assertEqual({float(state["step"]) for state in instance.critic_optimizer.state.values()}, {float(instance.gradient_updates)})

    def test_resume_before_at_and_across_boundary_is_exact_for_all_learning_state_rng_and_metrics(self):
        for steps in (3, 4):
            for delay in (1, 2, 3, 4):
                with self.subTest(warmup=steps, delay=delay):
                    torch.manual_seed(117005)
                    complete = learner(steps, delay)
                    inputs = batch(complete.actor)
                    options = {"measured_anchor": anchor(complete.actor), "anchor_weight": 3,
                               "successful_anchor": anchor(complete.actor), "self_retention_weight": 2,
                               "anchor_profile": "pose-budget-v2", "policy_regularization": "public-deadline-residual-v1",
                               "pace_weight": 40}
                    history = []
                    states = [snapshot(complete, history)]
                    first_actor_step = (steps // delay + 1) * delay
                    final_step = first_actor_step + 2 * delay
                    for unused in range(final_step):
                        history.append(complete.update(inputs, **options))
                        states.append(snapshot(complete, history))
                    checkpoints = sorted({steps - 1, steps, first_actor_step - 1, first_actor_step, first_actor_step + 1})
                    for checkpoint_step in checkpoints:
                        with self.subTest(checkpoint=checkpoint_step):
                            saved = copy.deepcopy(states[checkpoint_step])
                            resumed = learner(steps, delay)
                            resumed.actor.load_state_dict(saved["model"])
                            resumed.restore(saved["learner"])
                            torch.set_rng_state(saved["torchRng"])
                            resumed_history = list(saved["metrics"])
                            self.assert_exact(snapshot(resumed, resumed_history), saved)
                            for gradient_step in range(checkpoint_step + 1, final_step + 1):
                                resumed_history.append(resumed.update(inputs, **options))
                                self.assert_exact(snapshot(resumed, resumed_history), states[gradient_step])

    def test_rejected_updates_do_not_consume_a_warmup_step_or_mutate_state_or_rng(self):
        instance = learner(3)
        inputs = batch(instance.actor)
        for unused in range(2):
            instance.update(inputs)
        before = snapshot(instance)
        weighted = {**inputs, "importance_weights": torch.ones_like(inputs["rewards"])}
        invalid = [(inputs, {"policy_regularization": "invalid"}), (inputs, {"anchor_weight": True}),
                   (inputs, {"anchor_weight": -1}), (inputs, {"anchor_weight": 1}),
                   (inputs, {"pace_weight": math.nan}), (inputs, {"pace_weight": 2}),
                   (inputs, {"self_retention_weight": 1}), (weighted, {}),
                   (inputs, {"actor_batch": weighted}),
                   ({**weighted, "importance_weights": torch.zeros_like(inputs["rewards"])}, {"actor_batch": inputs})]
        for critic_batch, options in invalid:
            with self.subTest(options=options.keys()), self.assertRaises(ValueError):
                instance.update(critic_batch, **options)
            self.assert_exact(snapshot(instance), before)
        self.assertFalse(instance.update(inputs)["actorUpdated"])
        self.assertEqual((instance.gradient_updates, instance.actor_updates), (3, 0))
        self.assertTrue(instance.update(inputs)["actorUpdated"])
        self.assertEqual((instance.gradient_updates, instance.actor_updates), (4, 1))

    def test_restore_rejects_malformed_schema_warmup_and_counters_before_any_state_mutation(self):
        source = learner(3)
        target = learner(3)
        inputs = batch(source.actor)
        for unused in range(8):
            source.update(inputs)
        for unused in range(6):
            target.update(inputs)
        saved = copy.deepcopy(source.state())
        before = snapshot(target)
        mutations = [("schemaVersion", value) for value in (0, 1, 3, "2", None)]
        mutations += [("criticWarmupSteps", value) for value in (0, 2, -1, 513, True, 3.0, "3", None)]
        mutations += [("gradientUpdates", value) for value in (True, -1, 3201, 8.0, "8", None, math.nan, 10)]
        mutations += [("actorUpdates", value) for value in (True, -1, 0, 2, 4, 3.0, "3", None, math.nan)]
        invalid = [None, [], {}]
        invalid += [{**copy.deepcopy(saved), name: value} for name, value in mutations]
        invalid += [{name: value for name, value in copy.deepcopy(saved).items() if name != missing}
                    for missing in ("schemaVersion", "criticWarmupSteps", "gradientUpdates", "actorUpdates")]
        invalid += [{**copy.deepcopy(saved), "gradientUpdates": gradient_steps, "actorUpdates": actor_steps}
                    for gradient_steps, actor_steps in ((2, 1), (3, 1), (4, 0), (4, 2))]
        for index, malformed in enumerate(invalid):
            with self.subTest(malformed=index):
                with self.assertRaises(ValueError):
                    target.restore(malformed)
                self.assert_exact(snapshot(target), before)

    def test_restore_binds_opt_in_schema_warmup_and_global_policy_delay_atomically(self):
        source = learner(3, 2)
        inputs = batch(source.actor)
        for unused in range(8):
            source.update(inputs)
        saved = copy.deepcopy(source.state())
        for warmup, delay in ((0, 2), (1, 2), (4, 2), (3, 3), (3, 4)):
            with self.subTest(warmup=warmup, delay=delay):
                target = learner(warmup, delay)
                for unused in range(8):
                    target.update(inputs)
                before = snapshot(target)
                with self.assertRaises(ValueError):
                    target.restore(copy.deepcopy(saved))
                self.assert_exact(snapshot(target), before)
        for warmup in (0, 1):
            target = learner(warmup)
            before = snapshot(target)
            malformed = copy.deepcopy(target.state())
            malformed["criticWarmupSteps"] = 0 if warmup == 0 else True
            with self.subTest(warmup=warmup), self.assertRaises(ValueError):
                target.restore(malformed)
            self.assert_exact(snapshot(target), before)
        legacy = learner().state()
        opted_in = learner(3)
        before = snapshot(opted_in)
        with self.assertRaises(ValueError):
            opted_in.restore(legacy)
        self.assert_exact(snapshot(opted_in), before)


if __name__ == "__main__":
    unittest.main()
