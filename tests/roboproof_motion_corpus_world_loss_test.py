from pathlib import Path
import tempfile
import time
import unittest

import torch

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.corpus_warm import WORLD_LOSS_PROFILE, action_loss, load_corpus, loss_definition, world_row_weights
from roboproof.motion_learning.imitation import POSE_BUDGET_SCALES
from roboproof.motion_learning.td3_train import configuration, source_identity, train_seed
from roboproof_motion_corpus_warm_test import arguments, dataset, identical


class CorpusWorldLossTests(unittest.TestCase):
    def test_unequal_trajectories_contribute_equal_loss_and_gradient_without_batch_normalization(self):
        records = [{"rows": [{}]}, {"rows": []}, {"rows": [{}, {}, {}]}]
        weights = torch.tensor(world_row_weights(records))
        self.assertTrue(torch.allclose(weights, torch.tensor([2, 2 / 3, 2 / 3, 2 / 3])))
        predicted = torch.tensor([[0.2, 0.03, -0.04, 0.02], [0.5, 0.02, 0.01, 0.03],
                                  [0.6, -0.01, 0.02, 0.04], [0.7, 0.01, 0.03, -0.02]], requires_grad=True)
        targets = torch.zeros_like(predicted)
        errors = (predicted * torch.tensor(POSE_BUDGET_SCALES)).square().mean(dim=1)
        expected = (errors[0] + errors[1:].mean()) / 2
        loss = action_loss(predicted, targets, WORLD_LOSS_PROFILE, weights)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        expected_gradient = 2 * predicted.detach() * torch.tensor(POSE_BUDGET_SCALES).square() * weights[:, None] / 16
        self.assertTrue(torch.allclose(predicted.grad, expected_gradient))
        pieces = [action_loss(predicted[indices], targets[indices], WORLD_LOSS_PROFILE, weights[indices])
                  for indices in (slice(0, 1), slice(1, 4))]
        self.assertTrue(torch.allclose((pieces[0] + 3 * pieces[1]) / 4, expected))

    def test_weights_and_explicit_declarations_reject_invalid_or_missing_world_data(self):
        predicted = torch.zeros((2, 4))
        for invalid in (None, torch.ones((2, 1)), torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 1.0]),
                        torch.tensor([float("nan"), 1.0]), torch.tensor([203.0, 1.0]), torch.ones(2, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                action_loss(predicted, predicted, WORLD_LOSS_PROFILE, invalid)
        with self.assertRaises(ValueError):
            action_loss(predicted, predicted, "pose-budget-v2", torch.ones(2))
        for invalid in ([], [{"rows": []}], [{"rows": [{}] * 203}]):
            with self.assertRaises(ValueError):
                world_row_weights(invalid)
        with self.assertRaises(ValueError):
            configuration(**{**arguments(), "corpus_warm_loss_profile": WORLD_LOSS_PROFILE})
        self.assertEqual(loss_definition(WORLD_LOSS_PROFILE)["scales"], list(POSE_BUDGET_SCALES))

    def test_actual_world_weighted_corpus_native_training_and_resume_match_all_state(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            corpus = load_corpus(corpus_path)
            self.assertEqual(len(corpus["rowWeights"]), len(corpus["rows"]))
            self.assertAlmostEqual(sum(corpus["rowWeights"]), len(corpus["rows"]))
            config = configuration(**{**arguments(seed=890071, definition=corpus["definition"]), "corpus_warm_loss_profile": WORLD_LOSS_PROFILE})
            sources = source_identity()
            with MotionEnvironment() as environment:
                complete = train_seed(environment, root / "complete", config["seed"], config, sources, time.monotonic() + 90,
                                      expert_dataset=corpus_path)

                class Interrupted(Exception):
                    pass

                def interrupt(row):
                    if row["updates"] == 3:
                        raise Interrupted()

                with self.assertRaises(Interrupted):
                    train_seed(environment, root / "resumed", config["seed"], config, sources, time.monotonic() + 90,
                               on_checkpoint=interrupt, expert_dataset=corpus_path)
                resumed = train_seed(environment, root / "resumed", config["seed"], config, sources, time.monotonic() + 90,
                                     resume=True, expert_dataset=corpus_path)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertTrue(identical(torch.load(root / "complete/checkpoint.pt", weights_only=True),
                                      torch.load(root / "resumed/checkpoint.pt", weights_only=True)))
            self.assertEqual(complete["history"][0]["imitationLossProfile"], WORLD_LOSS_PROFILE)
            self.assertEqual(complete["history"][0]["actualDecisionsThisCollection"], 0)
            self.assertGreater(complete["history"][0]["imitationRowWeightMinimum"], 0)
            self.assertEqual(complete["steps"], sum(row["acceptedRows"] for row in complete["history"]))


if __name__ == "__main__":
    unittest.main()
