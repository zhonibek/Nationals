import copy
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import torch
from torch.distributions import Beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.model import MotionActorCritic, DEADLINE_CONTEXT, analytic_policy_kl, sampling_distribution, update
from roboproof.motion_learning.train import compatible_config, run, source_identity, train_seed


class AnalyticKlTests(unittest.TestCase):
    def test_float64_distribution_kl_matches_independent_density_integral_and_component_sum(self):
        alpha, beta, other_alpha, other_beta = 2.5, 3.5, 4.0, 2.0
        previous = Beta(torch.tensor([[alpha] * 4]), torch.tensor([[beta] * 4]))
        current = Beta(torch.tensor([[other_alpha] * 4]), torch.tensor([[other_beta] * 4]))
        measured = analytic_policy_kl(previous, current)
        self.assertEqual(measured.dtype, torch.float64)
        cells = 20000
        normalization = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
        other_normalization = math.lgamma(other_alpha + other_beta) - math.lgamma(other_alpha) - math.lgamma(other_beta)
        integral = 0.0
        for index in range(cells):
            point = (index + 0.5) / cells
            log_density = normalization + (alpha - 1) * math.log(point) + (beta - 1) * math.log1p(-point)
            other_log_density = other_normalization + (other_alpha - 1) * math.log(point) + (other_beta - 1) * math.log1p(-point)
            integral += math.exp(log_density) * (log_density - other_log_density) / cells
        self.assertAlmostEqual(float(measured[0, 0]), integral, delta=1e-7)
        self.assertAlmostEqual(float(measured.sum()), 4 * integral, delta=4e-7)
        self.assertTrue(torch.equal(analytic_policy_kl(previous, previous), torch.zeros((1, 4), dtype=torch.float64)))
        self.assertFalse(torch.allclose(measured, analytic_policy_kl(current, previous)))
        concentrated = analytic_policy_kl(sampling_distribution(previous, 8), sampling_distribution(current, 8))
        self.assertTrue(torch.all(concentrated > measured))
        with self.assertRaises(ValueError):
            analytic_policy_kl(previous, Beta(torch.ones((2, 4)), torch.ones((2, 4))))
        with self.assertRaises(ValueError):
            analytic_policy_kl(Beta(torch.tensor(2.0), torch.tensor(3.0)), Beta(torch.tensor(2.0), torch.tensor(3.0)))

    def test_diagnostics_do_not_change_rng_weights_or_optimizer_and_use_every_row(self):
        torch.set_num_threads(1)
        torch.manual_seed(95001)
        model = MotionActorCritic("broad-pace-v1")
        with torch.no_grad():
            model.actor.weight.uniform_(-0.1, 0.1)
        rows = []
        for index in range(129):
            raw = (torch.randn(34) * 0.2).tolist()
            factor = 1 if index % 2 == 0 else 8
            _, action, probability, value = model.act(raw, exploration_concentration=factor)
            rows.append({"observation": raw, "action": action, "logProb": float(probability), "value": float(value),
                         "nextValue": 0, "reward": float(index % 3), "terminated": True, "ended": True,
                         "explorationConcentration": factor})
        baseline = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.0003)
        with torch.no_grad():
            before, _ = model(torch.tensor([row["observation"] for row in rows]))
            concentrations = torch.tensor([row["explorationConcentration"] for row in rows])
            collected = sampling_distribution(before, concentrations)
        torch.manual_seed(95002)
        result = update(model, optimizer, rows, epochs=2)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(95002)
        with mock.patch("roboproof.motion_learning.model.analytic_policy_kl",
                        side_effect=lambda previous, current: torch.zeros_like(previous.concentration1, dtype=torch.float64)):
            update(baseline, baseline_optimizer, rows, epochs=2)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, baseline.state_dict()[name]) for name, value in model.state_dict().items()))
        for state in optimizer.state.values():
            self.assertTrue(all(torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor)))
        for state, other in zip(optimizer.state.values(), baseline_optimizer.state.values()):
            self.assertEqual(state.keys(), other.keys())
            for name, value in state.items():
                self.assertTrue(torch.equal(value, other[name]) if isinstance(value, torch.Tensor) else value == other[name])
        with torch.no_grad():
            after, _ = model(torch.tensor([row["observation"] for row in rows]))
            expected = analytic_policy_kl(collected, sampling_distribution(after, concentrations))
        self.assertAlmostEqual(result["finalRolloutMeanAnalyticKL"], float(expected.sum(-1).mean()), places=12)
        self.assertAlmostEqual(result["finalRolloutMaxStateAnalyticKL"], float(expected.sum(-1).max()), places=12)
        self.assertEqual(result["finalRolloutActionMeanAnalyticKL"], expected.mean(0).tolist())
        self.assertGreater(result["maximumRolloutMeanAnalyticKL"], 0)
        unchanged = MotionActorCritic("broad-pace-v1")
        raw = [0.0] * 34
        _, action, probability, value = unchanged.act(raw)
        stable = update(unchanged, torch.optim.Adam(unchanged.parameters(), lr=0),
                        [{"observation": raw, "action": action, "logProb": float(probability), "value": float(value),
                          "nextValue": 0, "reward": 0, "terminated": True, "ended": True}], epochs=1)
        self.assertEqual(stable["finalRolloutMeanAnalyticKL"], 0)

    def test_actual_training_resume_and_official_diagnostic_definition_remain_read_only(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = {"updates": 3, "maxSteps": 3000, "rolloutSteps": 16, "seed": 95003, "seedCount": 1,
                  "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1",
                  "featureTransform": "deadline-context-v1", "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
                  "explorationConcentration": 8, "objective": "finite-total-return-pose-effort-v3", "warmStartUpdates": 1}
        with tempfile.TemporaryDirectory() as temporary, MotionEnvironment() as environment:
            directory = Path(temporary)
            sources = source_identity()
            complete = train_seed(environment, directory / "complete", 95003, config, sources, time.monotonic() + 90)
            class Interrupted(Exception):
                pass
            def interrupt(row):
                if row["updates"] == 2:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_seed(environment, directory / "resumed", 95003, config, sources, time.monotonic() + 90, on_checkpoint=interrupt)
            resumed = train_seed(environment, directory / "resumed", 95003, config, sources, time.monotonic() + 90, resume=True)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(complete["history"], resumed["history"])
            self.assertNotIn("finalRolloutMeanAnalyticKL", complete["history"][0])
            self.assertTrue(all(len(row["finalRolloutActionMeanAnalyticKL"]) == 4 for row in complete["history"][1:]))
        self.assertFalse(compatible_config(config, {**config, "klDiagnosticDefinition": {"method": "different"}}))
        with tempfile.TemporaryDirectory() as temporary:
            saved = run(directory=temporary, seed=95004, seed_count=1, updates=1, max_steps=1000, rollout_steps=16, max_seconds=90)
        self.assertEqual(saved["config"]["klDiagnosticDefinition"]["precision"], "float64 Beta KL")
        self.assertFalse(saved["learnedImprovementVerified"])


if __name__ == "__main__":
    unittest.main()
