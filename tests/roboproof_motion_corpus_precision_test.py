import copy
from pathlib import Path
import tempfile
import time
import unittest

import torch

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.corpus_warm import action_loss, imitation, load_corpus, loss_definition
from roboproof.motion_learning.imitation import imitation_update, POSE_BUDGET_SCALES
from roboproof.motion_learning.model import MotionActorCritic
from roboproof.motion_learning.td3_train import configuration, source_identity, train_seed
from roboproof_motion_corpus_warm_test import arguments, dataset, identical


class CorpusPrecisionTests(unittest.TestCase):
    def test_loss_preserves_four_axis_denominator_and_declared_pose_units(self):
        predicted = torch.tensor([[0.2, 0.03, -0.04, 0.02]], requires_grad=True)
        targets = torch.zeros_like(predicted)
        loss = action_loss(predicted, targets, "pose-budget-v2")
        scales = torch.tensor(POSE_BUDGET_SCALES)
        self.assertTrue(torch.equal(loss, (predicted * scales).square().mean()))
        loss.backward()
        self.assertTrue(torch.allclose(predicted.grad, 2 * predicted.detach() * scales.square() / 4))
        self.assertEqual(loss_definition("pose-budget-v2")["scales"][0], 1)
        self.assertTrue(torch.equal(action_loss(predicted, targets, "uniform-action-v1"), predicted.square().mean()))
        for invalid in (torch.full((1, 4), float("nan")), torch.ones((1, 3))):
            with self.assertRaises(ValueError):
                action_loss(predicted, invalid, "pose-budget-v2")
        with self.assertRaises(ValueError):
            action_loss(predicted, torch.full((1, 4), 1.1), "pose-budget-v2")

    def test_default_update_remains_identical_in_weights_optimizer_metrics_and_rng(self):
        torch.set_num_threads(1)
        torch.manual_seed(180)
        original = MotionActorCritic("broad-pace-v1", "deadline-context-v1", False, 32, 2)
        wrapper = copy.deepcopy(original)
        original_optimizer = torch.optim.Adam([*original.body.parameters(), *original.actor.parameters()], lr=0.001)
        wrapper_optimizer = torch.optim.Adam([*wrapper.body.parameters(), *wrapper.actor.parameters()], lr=0.001)
        rows = [{"modelInput": [index / 10] * len(original.scales), "teacherAction": [0.6, 0.125, -0.125, 0.1]} for index in range(4)]
        before = torch.get_rng_state()
        original_metrics = imitation_update(original, original_optimizer, rows, epochs=2)
        following = torch.get_rng_state()
        torch.set_rng_state(before)
        wrapped_metrics = imitation(wrapper, wrapper_optimizer, rows, epochs=2)
        self.assertEqual(original_metrics, wrapped_metrics)
        self.assertTrue(identical(original.state_dict(), wrapper.state_dict()))
        self.assertTrue(identical(original_optimizer.state_dict(), wrapper_optimizer.state_dict()))
        self.assertTrue(torch.equal(following, torch.get_rng_state()))
        with self.assertRaises(ValueError):
            configuration(**{**arguments(), "corpus_warm_loss_profile": "pose-budget-v2"})

    def test_actual_precision_corpus_and_native_td3_resume_exactly(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            corpus = load_corpus(corpus_path)
            config = configuration(**{**arguments(seed=890031, definition=corpus["definition"]), "corpus_warm_loss_profile": "pose-budget-v2"})
            self.assertEqual(config["corpusWarmLossDefinition"], loss_definition("pose-budget-v2"))
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
                changed = {**config, "corpusWarmLossDefinition": {}}
                with self.assertRaisesRegex(ValueError, "definition mismatch"):
                    train_seed(environment, root / "bad", config["seed"], changed, sources, time.monotonic() + 90,
                               expert_dataset=corpus_path)
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertTrue(identical(torch.load(root / "complete/checkpoint.pt", weights_only=True),
                                      torch.load(root / "resumed/checkpoint.pt", weights_only=True)))
            self.assertEqual(complete["history"][0]["imitationLossProfile"], "pose-budget-v2")
            self.assertEqual(complete["history"][0]["actualDecisionsThisCollection"], 0)


if __name__ == "__main__":
    unittest.main()
