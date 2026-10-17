from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.corpus_warm import FullCorpusAnchor, load_corpus, world_row_weights
from roboproof.motion_learning.directional_anchor import ANCHOR_SOURCE, DirectionalPaceAnchor
from roboproof.motion_learning.imitation import POSE_BUDGET_SCALES
from roboproof.motion_learning.td3_train import configuration, source_identity, train_seed
from roboproof_motion_corpus_warm_test import arguments, dataset, identical
from roboproof_motion_full_corpus_anchor_test import ConstantActor, rows


class DirectionalAnchorTests(unittest.TestCase):
    def test_native_loss_preserves_pose_gradients_and_absolute_world_weights(self):
        records = [{"rows": rows(1)}, {"rows": rows(65)}]
        values = [row for record in records for row in record["rows"]]
        weights = world_row_weights(records)
        model = ConstantActor()
        with torch.no_grad():
            model.action[0] = -0.2
        anchor = DirectionalPaceAnchor(values, weights, 145)
        with patch("torch.randperm", return_value=torch.arange(len(values))):
            loss = anchor.loss(model)
        scales = torch.tensor(POSE_BUDGET_SCALES)
        difference = model.action.detach() * scales
        expected = (difference[1:].square().sum() / 4 * torch.tensor(weights[:32])).mean()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertEqual(float(model.action.grad[0]), 0)
        expected_gradient = 2 * scales[1:].square() * model.action.detach()[1:] * torch.tensor(weights[:32]).mean() / 4
        self.assertTrue(torch.allclose(model.action.grad[1:], expected_gradient))
        with torch.no_grad():
            model.action[0] = 0.2
        model.action.grad = None
        with patch("torch.randperm", return_value=torch.arange(len(values))):
            following = anchor.loss(model)
            baseline = FullCorpusAnchor(values, weights, 145).loss(model)
        self.assertTrue(torch.equal(following, baseline))
        following.backward()
        self.assertGreater(float(model.action.grad[0]), 0)

    def test_same_uniform_native_sampling_and_no_inference_or_warm_mutation(self):
        anchor = DirectionalPaceAnchor(rows(1200), [1.0] * 1200, 145)
        model = ConstantActor()
        permutation = torch.cat((torch.arange(1000, 1032), torch.arange(1000), torch.arange(1032, 1200)))
        with patch("torch.randperm", return_value=permutation) as sample:
            anchor.loss(model)
        sample.assert_called_once_with(1200, device="cpu")
        self.assertEqual(model.seen[:, 0].tolist(), list(range(1000, 1032)))
        with patch("torch.randperm") as sample, self.assertRaises(ValueError):
            anchor.loss(model, "uniform-action-v1")
        sample.assert_not_called()
        default = configuration(**arguments())
        self.assertNotIn("corpusAnchorSource", default)
        self.assertNotIn("roboproof/motion_learning/directional_anchor.py", source_identity())
        self.assertIn("roboproof/motion_learning/directional_anchor.py", source_identity(False, True))
        with self.assertRaises(ValueError):
            configuration(**{**arguments(), "corpus_anchor_source": ANCHOR_SOURCE})

    def test_actual_native_directional_continuation_is_exact_and_warm_policy_is_unchanged(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            corpus = load_corpus(corpus_path)
            actual = {**arguments(seed=890111, definition=corpus["definition"]), "n_steps": 32,
                      "corpus_warm_loss_profile": "pose-budget-v2", "corpus_anchor_source": ANCHOR_SOURCE}
            config = configuration(**actual)
            default = configuration(**{**actual, "corpus_anchor_source": "full-corpus-equal-world-v1"})
            self.assertEqual({key for key in config if config[key] != default.get(key)},
                             {"corpusAnchorSource", "corpusAnchorDefinition"})
            with self.assertRaises(ValueError):
                configuration(**{**actual, "online_anchor_interval": 1})
            sources = source_identity(False, True)
            with MotionEnvironment() as environment:
                baseline = train_seed(environment, root / "baseline", config["seed"], default, source_identity(),
                                      time.monotonic() + 90, expert_dataset=corpus_path)
                complete = train_seed(environment, root / "complete", config["seed"], config, sources,
                                      time.monotonic() + 90, expert_dataset=corpus_path)

                class Interrupted(Exception):
                    pass

                def interrupt(summary):
                    if summary["updates"] == 3:
                        raise Interrupted()

                with self.assertRaises(Interrupted):
                    train_seed(environment, root / "resumed", config["seed"], config, sources,
                               time.monotonic() + 90, on_checkpoint=interrupt, expert_dataset=corpus_path)
                resumed = train_seed(environment, root / "resumed", config["seed"], config, sources,
                                     time.monotonic() + 90, resume=True, expert_dataset=corpus_path)
            self.assertEqual((root / "baseline/policy-0001.json").read_bytes(), (root / "complete/policy-0001.json").read_bytes())
            self.assertEqual(baseline["history"][0], complete["history"][0])
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertTrue(identical(torch.load(root / "complete/checkpoint.pt", weights_only=True),
                                      torch.load(root / "resumed/checkpoint.pt", weights_only=True)))
            self.assertTrue(all(row["nativeAnchorSource"] == ANCHOR_SOURCE for row in complete["history"][1:]))
            self.assertGreater(complete["collectionSteps"], 0)
            self.assertEqual(complete["discardedCollectionStepsTotal"], 0)


if __name__ == "__main__":
    unittest.main()
