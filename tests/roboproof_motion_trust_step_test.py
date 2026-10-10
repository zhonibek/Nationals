import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import MotionActorCritic, DEADLINE_CONTEXT, optimizer_step_with_kl_bound, sampling_distribution, update, validate_policy_kl_limit
from roboproof.motion_learning.train import compatible_config, optimizer_constraint_definition, run, source_identity, train_seed


def optimizer_for(model, actor_rate=0.2, critic_rate=0.01):
    return torch.optim.Adam([{"params": [*model.body.parameters(), *model.actor.parameters()], "lr": actor_rate},
                             {"params": [*model.critic_body.parameters(), *model.critic.parameters()], "lr": critic_rate}])


def set_gradients(model):
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    model.actor.bias.grad[0] = -1
    model.critic.bias.grad.fill_(1)


class TrustStepTests(unittest.TestCase):
    def test_backtracking_matches_one_adam_step_at_accepted_rate_and_preserves_rng(self):
        torch.set_num_threads(1)
        torch.manual_seed(96001)
        model = MotionActorCritic("broad-pace-v1")
        expected = copy.deepcopy(model)
        set_gradients(model)
        set_gradients(expected)
        observations = torch.zeros((128, 34))
        factors = torch.full((128,), 8.0)
        with torch.no_grad():
            distribution, _ = model(observations)
            collected = sampling_distribution(distribution, factors)
        optimizer = optimizer_for(model)
        rng = torch.get_rng_state().clone()
        divergence, step = optimizer_step_with_kl_bound(model, optimizer, observations, False, collected, factors, 0.001)
        self.assertTrue(step["accepted"])
        self.assertGreater(step["attempts"], 1)
        self.assertLess(step["actorRate"], 0.2)
        self.assertLessEqual(float(divergence.sum(-1).mean()), 0.001)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        reference_optimizer = optimizer_for(expected, step["actorRate"])
        reference_optimizer.step()
        reference_optimizer.param_groups[0]["lr"] = 0.2
        self.assertTrue(all(torch.equal(value, expected.state_dict()[name]) for name, value in model.state_dict().items()))
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.2)
        self.assertEqual(optimizer.param_groups[1]["lr"], 0.01)
        for state, other in zip(optimizer.state.values(), reference_optimizer.state.values()):
            for name, value in state.items():
                self.assertTrue(torch.equal(value, other[name]) if isinstance(value, torch.Tensor) else value == other[name])

    def test_rejected_actor_restores_parameters_moments_and_runs_critic_once(self):
        torch.set_num_threads(1)
        torch.manual_seed(96002)
        model = MotionActorCritic("broad-pace-v1")
        optimizer = optimizer_for(model, actor_rate=0.001, critic_rate=0.001)
        set_gradients(model)
        optimizer.step()
        optimizer.param_groups[0]["lr"] = 0.2
        optimizer.param_groups[1]["lr"] = 0.01
        expected = copy.deepcopy(model)
        set_gradients(model)
        set_gradients(expected)
        observations = torch.zeros((64, 34))
        factors = torch.full((64,), 8.0)
        with torch.no_grad():
            distribution, _ = model(observations)
            collected = sampling_distribution(distribution, factors)
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        actor_gradients = [parameter.grad for parameter in [*model.body.parameters(), *model.actor.parameters()]]
        divergence, step = optimizer_step_with_kl_bound(model, optimizer, observations, False, collected, factors, 0.001, backtracks=0)
        self.assertFalse(step["accepted"])
        self.assertIsNone(step["actorRate"])
        self.assertEqual(float(divergence.sum()), 0)
        for parameter in [*expected.body.parameters(), *expected.actor.parameters()]:
            parameter.grad = None
        reference_optimizer = optimizer_for(expected)
        reference_optimizer.load_state_dict(saved_optimizer)
        reference_optimizer.step()
        self.assertTrue(all(torch.equal(value, expected.state_dict()[name]) for name, value in model.state_dict().items()))
        for parameter, gradient in zip([*model.body.parameters(), *model.actor.parameters()], actor_gradients):
            self.assertIs(parameter.grad, gradient)
            self.assertEqual(float(optimizer.state[parameter]["step"]), 1)
        for state, other in zip(optimizer.state_dict()["state"].values(), reference_optimizer.state_dict()["state"].values()):
            for name, value in state.items():
                self.assertTrue(torch.equal(value, other[name]) if isinstance(value, torch.Tensor) else value == other[name])

    def test_measurement_failure_rolls_back_parameters_adam_and_learning_rates(self):
        torch.set_num_threads(1)
        torch.manual_seed(96006)
        model = MotionActorCritic("broad-pace-v1")
        optimizer = optimizer_for(model, actor_rate=0.001, critic_rate=0.001)
        set_gradients(model)
        optimizer.step()
        observations = torch.zeros((64, 34))
        factors = torch.full((64,), 8.0)
        with torch.no_grad():
            distribution, _ = model(observations)
            collected = sampling_distribution(distribution, factors)
        before = copy.deepcopy(model.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        with mock.patch("roboproof.motion_learning.model.analytic_policy_kl", side_effect=RuntimeError("measurement failed")):
            with self.assertRaisesRegex(RuntimeError, "measurement failed"):
                optimizer_step_with_kl_bound(model, optimizer, observations, False, collected, factors, 0.03)
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()))
        self.assertEqual(optimizer.state_dict()["param_groups"], saved_optimizer["param_groups"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for state, old in zip(optimizer.state_dict()["state"].values(), saved_optimizer["state"].values()):
            for name, value in state.items():
                self.assertTrue(torch.equal(value, old[name]) if isinstance(value, torch.Tensor) else value == old[name])

    def test_full_ppo_auxiliary_losses_are_bounded_over_all_collected_states(self):
        torch.set_num_threads(1)
        torch.manual_seed(96003)
        model = MotionActorCritic("broad-pace-v1")
        with torch.no_grad():
            model.actor.weight.uniform_(-0.1, 0.1)
        rows = []
        for index in range(129):
            raw = (torch.randn(34) * 0.2).tolist()
            _, action, probability, value = model.act(raw, exploration_concentration=8)
            rows.append({"observation": raw, "action": action, "logProb": float(probability), "value": float(value),
                         "nextValue": 0, "reward": float(index % 3), "terminated": True, "ended": True,
                         "requiredPace": 1.0, "explorationConcentration": 8})
        optimizer = optimizer_for(model, actor_rate=0.01, critic_rate=0.0003)
        result = update(model, optimizer, rows, epochs=2, policy_regularization="public-deadline-residual-v1",
                        deadline_pace_loss_weight=40, target_policy_kl=0.03)
        self.assertLessEqual(result["maximumRolloutMeanAnalyticKL"], 0.03)
        self.assertGreater(result["optimizerActorStepsAccepted"], 0)
        self.assertEqual(result["optimizerActorStepsAccepted"] + result["optimizerActorStepsRejected"], 6)
        self.assertGreater(result["optimizerProposalCount"], 6)
        self.assertTrue(torch.all(model.actor.weight.grad.abs().sum(-1) > 0))
        self.assertTrue(all(float(state["step"]) == 6 for parameter, state in optimizer.state.items()
                            if parameter in set([*model.critic_body.parameters(), *model.critic.parameters()])))
        for invalid in (True, 0, -1, 0.0001, 1, math.nan, math.inf, "0.03"):
            with self.assertRaises(ValueError):
                validate_policy_kl_limit(invalid)
            with self.assertRaises(ValueError):
                run(target_policy_kl=invalid)
        with self.assertRaisesRegex(ValueError, "Separate"):
            update(model, torch.optim.Adam(model.parameters()), rows, epochs=1, target_policy_kl=0.03)

    def test_real_bounded_training_exact_resume_and_official_constraint_metadata(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 3, "maxSteps": 3000, "rolloutSteps": 16, "seed": 96004, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "deadline-context-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
                  "explorationConcentration": 8, "objective": "finite-total-return-pose-effort-v3", "warmStartUpdates": 1,
                  "demonstrationAnchorWeight": 10, "demonstrationAnchorLossProfile": "pose-budget-v2",
                  "policyRegularization": "public-deadline-residual-v1", "deadlinePaceLossWeight": 40,
                  "targetPolicyKL": 0.03, "optimizerConstraintDefinition": optimizer_constraint_definition(0.03)}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 96004, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 96004, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 96004, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(complete["actorWeightsChanged"])
            self.assertGreater(sum(row["optimizerActorStepsAccepted"] for row in complete["history"][1:]), 0)
            warm_actor = json.loads((directory / "complete/policy-0001.json").read_text())["layers"]
            trained_actor = json.loads((directory / "complete/policy-0003.json").read_text())["layers"]
            self.assertNotEqual(warm_actor, trained_actor)
            self.assertTrue(all(row["maximumRolloutMeanAnalyticKL"] <= 0.03 for row in complete["history"][1:]))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_seed(environment, directory / "resumed", 96004,
                           {**config, "targetPolicyKL": 0.02, "optimizerConstraintDefinition": optimizer_constraint_definition(0.02)},
                           sources, time.monotonic() + 90, resume=True)
        self.assertFalse(compatible_config(config, {**config, "targetPolicyKL": None}))
        with tempfile.TemporaryDirectory() as temporary:
            saved = run(directory=temporary, seed=96005, seed_count=1, updates=1, max_steps=1000,
                        rollout_steps=16, max_seconds=90, target_policy_kl=0.03)
        self.assertEqual(saved["config"]["optimizerConstraintDefinition"], optimizer_constraint_definition(0.03))
        self.assertFalse(saved["learnedImprovementVerified"])


if __name__ == "__main__":
    unittest.main()
