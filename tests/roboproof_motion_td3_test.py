import copy
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_learning.model import MotionActorCritic, export
from roboproof.motion_learning.td3 import ALGORITHM, ReplayBuffer, TD3Learner, clipped_target, mean_action


def batch(actor, count=64):
    observations = torch.randn(count, len(actor.scales)) * 0.02
    return {"observations": observations, "next_observations": observations + 0.005,
            "actions": torch.randn(count, 4).clamp(-1, 1), "rewards": torch.linspace(-2, 2, count),
            "terminals": torch.arange(count) % 3 == 0, "required_paces": torch.full((count,), 0.95)}


class TD3Tests(unittest.TestCase):
    def test_actor_only_template_keeps_legacy_actor_parameters_and_export_is_not_mislabeled_ppo(self):
        torch.manual_seed(101001)
        baseline = MotionActorCritic("broad-pace-v1")
        torch.manual_seed(101001)
        actor = MotionActorCritic("broad-pace-v1", value_critic=False)
        self.assertIsNone(actor.critic)
        self.assertIsNone(actor.critic_body)
        for before, after in zip([*baseline.body.parameters(), *baseline.actor.parameters()], actor.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        inputs = torch.zeros((5, 34))
        torch.testing.assert_close(mean_action(actor, inputs), mean_action(baseline, inputs), rtol=0, atol=0)
        self.assertEqual(export(actor, {}, 1, 0, 0, ALGORITHM)["algorithm"], ALGORITHM)
        self.assertEqual(export(baseline, {}, 1, 0, 0)["algorithm"], "ppo-beta-reference-v1")
        with self.assertRaises(ValueError):
            MotionActorCritic(value_critic="false")
        with self.assertRaises(ValueError):
            export(actor, {}, 1, 0, 0, "ppo-disguised-td3")

    def test_twin_target_matches_independent_minimum_noise_bounds_and_intrinsic_terminal_formula(self):
        actor = MotionActorCritic("broad-pace-v1", value_critic=False)
        inputs = batch(actor, 7)
        class Critics(torch.nn.Module):
            def forward(self, observations, actions):
                return torch.full((len(observations),), 4.0), torch.full((len(observations),), -2.0)
        torch.manual_seed(101002)
        noise = (torch.randn_like(inputs["actions"]) * 0.2).clamp(-0.1, 0.1)
        expected_actions = (mean_action(actor, inputs["next_observations"]) + noise).clamp(-1, 1)
        expected_values = inputs["rewards"] - 0.995 * (~inputs["terminals"]).float() * 2
        torch.manual_seed(101002)
        values, actions = clipped_target(actor, Critics(), inputs, 0.995, 0.2, 0.1)
        torch.testing.assert_close(actions, expected_actions)
        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(values[inputs["terminals"]], inputs["rewards"][inputs["terminals"]])
        self.assertTrue(torch.all(actions.abs() <= 1))
        self.assertFalse(values.requires_grad)

    def test_delayed_actor_updates_every_head_and_polyak_targets_without_updating_critics_via_actor(self):
        torch.set_num_threads(1)
        torch.manual_seed(101003)
        actor = MotionActorCritic("broad-pace-v1", value_critic=False)
        learner = TD3Learner(actor, target_tau=0.1)
        inputs = batch(actor)
        initial_actor = [parameter.clone() for parameter in actor.parameters()]
        initial_target_critic = [parameter.clone() for parameter in learner.target_critics.parameters()]
        first = learner.update(inputs, policy_regularization="public-deadline-residual-v1", pace_weight=40)
        self.assertFalse(first["actorUpdated"])
        for original, current, target in zip(initial_actor, actor.parameters(), learner.target_actor.parameters()):
            torch.testing.assert_close(original, current, rtol=0, atol=0)
            torch.testing.assert_close(original, target, rtol=0, atol=0)
        second = learner.update(inputs, policy_regularization="public-deadline-residual-v1", pace_weight=40)
        self.assertTrue(second["actorUpdated"])
        self.assertEqual((learner.gradient_updates, learner.actor_updates), (2, 1))
        self.assertTrue(torch.all(actor.actor.weight.grad.abs().sum(-1) > 0))
        for original, current, target in zip(initial_actor, actor.parameters(), learner.target_actor.parameters()):
            torch.testing.assert_close(target, original * 0.9 + current * 0.1)
        for original, current, target in zip(initial_target_critic, learner.critics.parameters(), learner.target_critics.parameters()):
            torch.testing.assert_close(target, original * 0.9 + current * 0.1)
        self.assertTrue(all(float(state["step"]) == 2 for state in learner.critic_optimizer.state.values()))
        self.assertTrue(all(parameter.requires_grad for parameter in learner.critics.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in learner.target_critics.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in learner.target_actor.parameters()))

    def test_replay_ring_exact_rng_round_trip_and_atomic_invalid_append(self):
        replay = ReplayBuffer(34, 64)
        rows = [{"modelInput": [index / 100] * 34, "nextModelInput": [(index + 1) / 100] * 34,
                 "action": [0.1, 0, 0, 0], "reward": 0.01, "requiredPace": 0.85, "terminal": index % 4 == 0}
                for index in range(80)]
        replay.append(rows)
        self.assertEqual((replay.count, replay.cursor), (64, 16))
        saved = replay.state()
        restored = ReplayBuffer(34, 64)
        restored.restore(saved)
        torch.manual_seed(101004)
        first = replay.sample(32)
        torch.manual_seed(101004)
        second = restored.sample(32)
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            replay.append([rows[0], {**rows[1], "action": [2, 0, 0, 0]}])
        after = replay.state()
        for name, value in saved.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(after[name], value, rtol=0, atol=0)
            else:
                self.assertEqual(after[name], value)
        for mutate in (lambda state: state["actions"].fill_(2), lambda state: state["rewards"].fill_(math.nan)):
            malformed = copy.deepcopy(saved)
            mutate(malformed)
            with self.assertRaises(ValueError):
                restored.restore(malformed)

    def test_learner_state_restore_reproduces_next_update_and_rejects_target_scale_tampering(self):
        torch.manual_seed(101005)
        actor = MotionActorCritic("broad-pace-v1", value_critic=False)
        learner = TD3Learner(actor)
        inputs = batch(actor)
        learner.update(inputs)
        actor_state = copy.deepcopy(actor.state_dict())
        saved = copy.deepcopy(learner.state())
        random_state = torch.get_rng_state()
        expected = learner.update(inputs)
        clone_actor = MotionActorCritic("broad-pace-v1", value_critic=False)
        clone_actor.load_state_dict(actor_state)
        clone = TD3Learner(clone_actor)
        clone.restore(saved)
        torch.set_rng_state(random_state)
        actual = clone.update(inputs)
        self.assertEqual(actual, expected)
        for original, reproduced in zip(actor.parameters(), clone_actor.parameters()):
            torch.testing.assert_close(original, reproduced, rtol=0, atol=0)
        corrupted = copy.deepcopy(saved)
        corrupted["targetActor"]["scales"][3] *= 2
        with self.assertRaisesRegex(ValueError, "sensor scales"):
            clone.restore(corrupted)


if __name__ == "__main__":
    unittest.main()
