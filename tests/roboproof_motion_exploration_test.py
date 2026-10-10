import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest

import torch
from torch.distributions import Beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import MotionActorCritic, DEADLINE_CONTEXT, exploration_factor, sampling_distribution, update, control_prior_loss
from roboproof.motion_learning.imitation import MeasuredAnchor
from roboproof.motion_learning.train import compatible_config, source_identity, train_seed, run


class MotionExplorationTests(unittest.TestCase):
    def test_scaled_beta_preserves_means_and_support_while_reducing_every_variance(self):
        original = Beta(torch.tensor([[4.5, 60, 60, 60]]), torch.tensor([[1.5, 60, 60, 60]]))
        concentrated = sampling_distribution(original, 8)
        torch.testing.assert_close(concentrated.mean, original.mean)
        self.assertTrue(torch.all(concentrated.variance < original.variance / 6))
        for position in (0.001, 0.01, 0.5, 0.99, 0.999):
            self.assertTrue(torch.isfinite(concentrated.log_prob(torch.full((1, 4), position))).all())
        torch.manual_seed(8901)
        samples = concentrated.sample((4096,))
        self.assertTrue(torch.all(samples > 0) and torch.all(samples < 1))
        self.assertTrue(torch.all(samples.std(0) > 0))
        self.assertIs(sampling_distribution(original, 1), original)
        model = MotionActorCritic("broad-pace-v1")
        raw = [0.0] * 34
        before = model.act(raw, deterministic=True)
        after = model.act(raw, deterministic=True, exploration_concentration=8)
        self.assertEqual(before[0], after[0])
        self.assertEqual(float(before[3]), float(after[3]))
        for invalid in (True, 0, 17, math.nan, math.inf, "8"):
            with self.assertRaises(ValueError):
                exploration_factor(invalid)
            with self.assertRaises(ValueError):
                model.act(raw, exploration_concentration=invalid)
            with self.assertRaises(ValueError):
                run(exploration_concentration=invalid)
        for invalid in (torch.tensor([True]), torch.tensor([math.nan]), torch.tensor([0.9]), torch.tensor([17.0]), torch.tensor([[8.0]])):
            with self.assertRaises(ValueError):
                sampling_distribution(original, invalid)

    def test_collected_scaled_likelihood_and_entropy_replay_exactly_for_mixed_factors(self):
        torch.set_num_threads(1)
        torch.manual_seed(8902)
        model = MotionActorCritic("broad-pace-v1")
        raw = [0.0] * 34
        rows, entropies = [], []
        for index in range(64):
            factor = 1 if index % 2 == 0 else 8
            action, unit, log_prob, value = model.act(raw, exploration_concentration=factor)
            distribution, _ = model(torch.tensor(raw))
            sampled = sampling_distribution(distribution, factor)
            self.assertAlmostEqual(float(log_prob), float(sampled.log_prob(unit).sum().detach()), places=6)
            self.assertTrue(all(-1 < element < 1 for element in action))
            entropies.append(float(sampled.entropy().sum().detach()))
            rows.append({"observation": raw, "action": unit, "logProb": float(log_prob), "value": float(value),
                         "nextValue": 0, "reward": float(index % 2), "terminated": True, "ended": True,
                         "explorationConcentration": factor})
        optimizer = torch.optim.Adam(model.parameters(), lr=0)
        result = update(model, optimizer, rows, epochs=1)
        self.assertAlmostEqual(result["meanApproxKL"], 0, places=7)
        self.assertEqual(result["clipFraction"], 0)
        self.assertAlmostEqual(result["meanEntropy"], sum(entropies) / len(entropies), places=5)
        self.assertEqual(result["meanExplorationConcentration"], 4.5)
        previous = copy.deepcopy(model)
        explicit = copy.deepcopy(model)
        legacy = [{key: value for key, value in row.items() if key != "explorationConcentration"} for row in rows[:1]]
        for row in legacy:
            row["reward"] = 0
        declared = [{**row, "explorationConcentration": 1} for row in legacy]
        torch.manual_seed(8903)
        default_result = update(previous, torch.optim.Adam(previous.parameters()), legacy, epochs=1)
        torch.manual_seed(8903)
        explicit_result = update(explicit, torch.optim.Adam(explicit.parameters()), declared, epochs=1)
        self.assertEqual(default_result, explicit_result)
        self.assertTrue(all(torch.equal(previous.state_dict()[key], tensor) for key, tensor in explicit.state_dict().items()))
        with self.assertRaises(ValueError):
            update(model, optimizer, [{**rows[0], "explorationConcentration": True}])

    def test_heterogeneous_temporal_rows_replay_fractional_and_boundary_factors_across_minibatches(self):
        torch.set_num_threads(1)
        torch.manual_seed(8910)
        model = MotionActorCritic("broad-pace-v1", "deadline-context-v1")
        with torch.no_grad():
            model.actor.weight.uniform_(-0.15, 0.15)
        model.reset_policy_state(3)
        factors = (1, 1.25, 2.5, 8, 15.5, 16)
        rows = []
        for index in range(129):
            raw = (torch.randn(34) * 0.5).tolist()
            raw[18], raw[25] = 10 - index * 0.05, index * 0.01
            factor = factors[index % len(factors)]
            _, unit, probability, value = model.act(raw, exploration_concentration=factor)
            rows.append({"modelInput": list(model.last_input), "action": unit, "logProb": float(probability),
                         "value": float(value), "nextValue": 0, "reward": float(index % 3),
                         "terminated": True, "ended": True, "explorationConcentration": factor})
        with torch.no_grad():
            distribution, _ = model(torch.tensor([row["modelInput"] for row in rows]), encoded=True)
            concentrations = torch.tensor([row["explorationConcentration"] for row in rows])
            alpha = distribution.concentration1.double() * concentrations.double().unsqueeze(-1)
            beta = distribution.concentration0.double() * concentrations.double().unsqueeze(-1)
            actions = torch.stack([row["action"] for row in rows]).double()
            oracle_probability = ((alpha - 1) * actions.log() + (beta - 1) * torch.log1p(-actions)
                                  + torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)).sum(-1)
            replayed = sampling_distribution(distribution, concentrations)
            torch.testing.assert_close(replayed.log_prob(actions.float()).sum(-1).double(), oracle_probability,
                                       rtol=0, atol=0.005)
            torch.testing.assert_close(torch.tensor([row["logProb"] for row in rows]).double(), oracle_probability,
                                       rtol=0, atol=0.005)
            oracle_entropy = Beta(alpha, beta).entropy().sum(-1).mean().item()
        result = update(model, torch.optim.Adam(model.parameters(), lr=0), rows, epochs=1)
        self.assertLess(abs(result["meanApproxKL"]), 0.00001)
        self.assertEqual(result["clipFraction"], 0)
        self.assertAlmostEqual(result["meanEntropy"], oracle_entropy, delta=0.005)

    def test_sampling_factor_does_not_rescale_control_prior_or_precision_anchor(self):
        torch.set_num_threads(1)
        torch.manual_seed(8911)
        model = MotionActorCritic("broad-pace-v1")
        with torch.no_grad():
            model.actor.weight.uniform_(-0.1, 0.1)
            model.actor.bias[1] += 3
        raw = torch.randn(34).tolist()
        features = model.encode_observation(raw).tolist()
        anchor = MeasuredAnchor(len(features))
        anchor.append([{"modelInput": features, "teacherAction": [0.7, 0, 0, 0]}])
        with torch.no_grad():
            distribution, _ = model(torch.tensor([features] * 64), encoded=True)
            expected_pace, expected_prior = control_prior_loss(distribution, torch.full((64,), 0.95))
            expected_anchor = float(anchor.loss(model, "pose-budget-v2"))
            units = distribution.sample()
        results = []
        for factor in (1, 8, 16):
            rows = [{"observation": raw, "action": units[index],
                     "logProb": float(sampling_distribution(distribution, factor).log_prob(units).sum(-1)[index]),
                     "value": 0, "nextValue": 0, "reward": float(index % 3), "terminated": True, "ended": True,
                     "requiredPace": 0.95, "explorationConcentration": factor} for index in range(64)]
            torch.manual_seed(8912)
            result = update(model, torch.optim.Adam(model.parameters(), lr=0), rows, epochs=1,
                            policy_regularization="public-deadline-residual-v1", deadline_pace_loss_weight=40,
                            measured_anchor=anchor, anchor_weight=10, anchor_loss_profile="pose-budget-v2")
            self.assertAlmostEqual(result["meanDeadlinePaceLoss"], float(expected_pace), places=6)
            self.assertAlmostEqual(result["meanResidualPriorKL"], float(expected_prior), places=6)
            self.assertAlmostEqual(result["meanMeasuredAnchorLoss"], expected_anchor, places=6)
            self.assertTrue(torch.isfinite(model.actor.weight.grad).all())
            self.assertTrue(torch.all(model.actor.weight.grad.abs().sum(-1) > 0))
            results.append(result)
        for key in ("meanDeadlinePaceLoss", "meanResidualPriorKL", "meanMeasuredAnchorLoss"):
            self.assertEqual(len({result[key] for result in results}), 1)

    def test_partial_concentrated_rollout_rolls_back_rng_without_refunding_work(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 2, "maxSteps": 2, "rolloutSteps": 16, "seed": 8913, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "deadline-context-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
                  "explorationConcentration": 8, "objective": "finite-total-return-pose-effort-v3"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            interrupted = train_seed(environment, directory / "resumed", 8913, config, sources, time.monotonic() + 90)
            self.assertEqual(interrupted["updates"], 0)
            self.assertEqual(interrupted["discardedUncommittedSteps"], 2)
            exposure = json.loads((directory / "resumed/exposure.json").read_text())
            self.assertEqual(len(exposure["cases"]), 1)
            unchanged = train_seed(environment, directory / "resumed", 8913, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(unchanged["updates"], 0)
            self.assertEqual(unchanged["collectionSteps"], 2)
            extended = {**config, "maxSteps": 2000}
            resumed = train_seed(environment, directory / "resumed", 8913, extended, sources, time.monotonic() + 90, resume=True)
            complete = train_seed(environment, directory / "complete", 8913, extended, sources, time.monotonic() + 90)
            self.assertEqual(resumed["policySha256"], complete["policySha256"])
            self.assertEqual(resumed["history"], complete["history"])
            self.assertEqual(resumed["collectionSteps"], complete["collectionSteps"] + 2)
            self.assertEqual(resumed["discardedCollectionStepsTotal"], 2)

    def test_real_concentrated_ppo_resumes_and_rejects_changed_sampling_definition(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 3, "maxSteps": 2000, "rolloutSteps": 16, "seed": 8904, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "deadline-context-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
                  "policyRegularization": "public-deadline-residual-v1", "deadlinePaceLossWeight": 40,
                  "warmStartUpdates": 1, "demonstrationAnchorWeight": 10, "demonstrationAnchorLossProfile": "pose-budget-v2",
                  "teacherSelection": "pose-deadline-margin-v2", "explorationConcentration": 8,
                  "objective": "finite-total-return-pose-effort-v3"}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 8904, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 8904, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 8904, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertTrue(complete["actorWeightsChanged"])
            self.assertTrue(all(history["meanExplorationConcentration"] == 8 for history in complete["history"][1:]))
            for changed in ({"explorationConcentration": 1}, {"explorationDefinition": {"factor": 2}}):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    train_seed(environment, directory / "resumed", 8904, {**config, **changed}, sources,
                               time.monotonic() + 90, resume=True)
        self.assertFalse(compatible_config(config, {**config, "explorationConcentration": 1}))
        with tempfile.TemporaryDirectory() as temporary:
            declared = run(directory=temporary, seed=8905, seed_count=1, updates=1, max_steps=500,
                           rollout_steps=16, max_seconds=90, exploration_concentration=8)
        self.assertEqual(declared["status"], "completed")
        self.assertEqual(declared["config"]["explorationDefinition"]["factor"], 8)
        self.assertEqual(declared["models"][0]["history"][0]["meanExplorationConcentration"], 8)


if __name__ == "__main__":
    unittest.main()
