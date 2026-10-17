import copy
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.corpus_warm import FullCorpusAnchor, load_corpus, world_row_weights
from roboproof.motion_learning.imitation import POSE_BUDGET_SCALES
from roboproof.motion_learning.td3_train import configuration, source_identity, train_seed
from roboproof_motion_corpus_warm_test import arguments, dataset, identical


class ConstantActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Parameter(torch.tensor([0.2, 0.03, -0.04, 0.02]))
        self.seen = None

    def forward(self, features, encoded=False):
        self.seen = features
        return SimpleNamespace(mean=(self.action.expand(len(features), 4) + 1) / 2), None


def rows(count):
    return [{"modelInput": [float(index)] + [0.0] * 144, "teacherAction": [0.0] * 4} for index in range(count)]


class FullCorpusAnchorTests(unittest.TestCase):
    def test_expected_equal_world_gradient_uses_absolute_weights_without_batch_renormalization(self):
        records = [{"rows": rows(1)}, {"rows": []}, {"rows": rows(65)}]
        data = [row for record in records for row in record["rows"]]
        for index, row in enumerate(data):
            row["teacherAction"] = [index * 0.01, index * 0.0001, 0.0, 0.0]
        weights = world_row_weights(records)
        anchor = FullCorpusAnchor(data, weights, 145)
        model = ConstantActor()
        with patch("torch.randperm", return_value=torch.arange(66)):
            loss = anchor.loss(model)
        scales = torch.tensor(POSE_BUDGET_SCALES)
        targets = torch.tensor([row["teacherAction"] for row in data[:32]])
        absolute_weights = torch.tensor(weights[:32])
        difference = model.action - targets
        expected = ((difference * scales).square().mean(1) * absolute_weights).mean()
        self.assertGreater(float(absolute_weights.mean()), 1.5)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        expected_gradient = 2 * scales.square() * (difference.detach() * absolute_weights[:, None]).sum(0) / (32 * 4)
        self.assertTrue(torch.allclose(model.action.grad, expected_gradient))

    def test_native_batch_can_draw_measured_rows_beyond_the_legacy_bank_and_replays_rng_exactly(self):
        anchor = FullCorpusAnchor(rows(1200), [1.0] * 1200, 145)
        model = ConstantActor()
        permutation = torch.cat((torch.arange(1000, 1032), torch.arange(1000), torch.arange(1032, 1200)))
        with patch("torch.randperm", return_value=permutation) as sample:
            anchor.loss(model)
            sample.assert_called_once_with(1200, device="cpu")
        self.assertEqual(model.seen[:, 0].tolist(), list(range(1000, 1032)))
        torch.manual_seed(890082)
        before = torch.get_rng_state()
        first = anchor.loss(model)
        sampled = model.seen.clone()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        self.assertTrue(torch.equal(first, anchor.loss(model)))
        self.assertTrue(torch.equal(sampled, model.seen))
        self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_malformed_or_undeclared_anchor_inputs_fail_before_rng_or_model_use(self):
        for values, weights, width in (([], [], 145), (rows(1), [0.0], 145), (rows(1), [203.0], 145),
                                       (rows(1), [float("nan")], 145), (rows(1), [], 145), (rows(1), [1.0], 34),
                                       ([{"modelInput": [0.0] * 145, "teacherAction": [2.0] * 4}], [1.0], 145),
                                       ([{"modelInput": [float("nan")] * 145, "teacherAction": [0.0] * 4}], [1.0], 145)):
            with self.assertRaises(ValueError):
                FullCorpusAnchor(values, weights, width)
        with patch("torch.randperm") as sample, self.assertRaises(ValueError):
            FullCorpusAnchor(rows(1), [1.0], 145).loss(ConstantActor(), "uniform-action-v1")
        sample.assert_not_called()
        default = arguments()
        omitted = {name: value for name, value in default.items() if name != "corpus_anchor_source"}
        self.assertEqual(configuration(**default), configuration(**omitted))
        self.assertNotIn("corpusAnchorSource", configuration(**default))
        with self.assertRaises(ValueError):
            configuration(**{**default, "corpus_anchor_source": "full-corpus-equal-world-v1"})
        with self.assertRaises(ValueError):
            configuration(**{**default, "corpus_anchor_source": "invented"})

    def test_actual_complete_corpus_anchor_native_resume_matches_every_checkpoint_state(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            corpus = load_corpus(corpus_path)
            actual = {**arguments(seed=890083, definition=corpus["definition"]), "n_steps": 32,
                      "corpus_warm_loss_profile": "pose-budget-v2", "corpus_anchor_source": "full-corpus-equal-world-v1"}
            config = configuration(**actual)
            baseline = configuration(**{**actual, "corpus_anchor_source": "balanced-bank-v1"})
            self.assertEqual(set(config) - set(baseline), {"corpusAnchorSource", "corpusAnchorDefinition"})
            self.assertEqual(config["corpusAnchorDefinition"]["rows"], len(corpus["rows"]))
            with self.assertRaises(ValueError):
                configuration(**{**actual, "online_anchor_interval": 1})
            with self.assertRaises(ValueError):
                configuration(**{**actual, "demonstration_anchor_weight": 0})
            sources = source_identity()
            with MotionEnvironment() as environment:
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
                malformed = copy.deepcopy(config)
                malformed["corpusAnchorDefinition"]["rows"] += 1
                with patch.object(environment, "reset") as reset, self.assertRaisesRegex(ValueError, "definition mismatch"):
                    train_seed(environment, root / "resumed", config["seed"], malformed, sources,
                               time.monotonic() + 90, resume=True, expert_dataset=corpus_path)
                reset.assert_not_called()
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            first = torch.load(root / "complete/checkpoint.pt", weights_only=True)
            second = torch.load(root / "resumed/checkpoint.pt", weights_only=True)
            self.assertTrue(identical(first, second))
            self.assertTrue(all(row["nativeAnchorRows"] == len(corpus["rows"]) for row in complete["history"][1:]))
            self.assertTrue(all(row["nativeAnchorBatchRows"] == 32 for row in complete["history"][1:]))
            self.assertEqual(complete["history"][0]["actualDecisionsThisCollection"], 0)
            self.assertLess(len(first["measuredAnchor"]["features"]), len(corpus["rows"]))
            self.assertNotIn("fullCorpusAnchor", first)
            self.assertTrue(complete["actorWeightsChanged"])


if __name__ == "__main__":
    unittest.main()
