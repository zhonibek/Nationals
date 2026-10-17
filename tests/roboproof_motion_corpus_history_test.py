import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.corpus_history import extend_records, extend_corpus, history_definition, TARGET_TRANSFORM
from roboproof.motion_learning.corpus_warm import load_corpus
from roboproof.motion_learning.model import MotionActorCritic
from roboproof.motion_learning.td3_train import configuration, source_identity, train_seed
from roboproof_motion_corpus_warm_test import arguments, dataset, identical


def record(length, offset=0):
    frames = []
    rows = []
    context = [0.2, 0.3, 0.4, 0.5]
    for index in range(length):
        frames.append([float(index + offset)] * 35)
        short = frames[-4:]
        values = [item for frame in [frames[0]] * (4 - len(short)) + short for item in frame] + context + [0.25]
        rows.append({"modelInput": values, "teacherAction": [0.2, 0.1, 0.0, -0.1]})
    return {"rows": rows}


class CorpusHistoryTests(unittest.TestCase):
    def test_complete_chronological_view_retains_labels_context_and_world_reset(self):
        records = [record(20), {"rows": []}, record(3, 100)]
        original = copy.deepcopy(records)
        expanded = extend_records(records)
        self.assertEqual(records, original)
        self.assertEqual([len(item["rows"]) for item in expanded], [20, 0, 3])
        self.assertEqual(len(expanded[0]["rows"][0]["modelInput"]), 565)
        self.assertEqual(expanded[0]["rows"][15]["modelInput"][:560], [float(index) for index in range(16) for _ in range(35)])
        self.assertEqual(expanded[0]["rows"][17]["modelInput"][:560], [float(index) for index in range(2, 18) for _ in range(35)])
        self.assertEqual(expanded[2]["rows"][0]["modelInput"][:560], [100.0] * 560)
        for before, after in zip(records, expanded):
            for first, second in zip(before["rows"], after["rows"]):
                self.assertEqual(first["teacherAction"], second["teacherAction"])
                self.assertEqual(first["modelInput"][-5:], second["modelInput"][-5:])

    def test_inconsistent_history_or_context_cannot_be_reinterpreted(self):
        for position in (0, 140):
            values = [record(3)]
            values[0]["rows"][1]["modelInput"][position] += 1
            with self.assertRaises(ValueError):
                extend_records(values)
        self.assertIsNone(history_definition(4))
        for value in (True, 0, 8, 16.0):
            with self.assertRaises(ValueError):
                history_definition(value)

    def test_actual_source_rows_match_native_existing_long_encoder_without_rewriting_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            original = {filename.name: filename.read_bytes() for filename in corpus_path.iterdir() if filename.is_file()}
            corpus = load_corpus(corpus_path)
            view = extend_corpus(corpus_path, corpus)
            self.assertEqual(len(view["rows"]), len(corpus["rows"]))
            self.assertEqual(view["rowWeights"], corpus["rowWeights"])
            self.assertEqual(len(view["anchorRows"]), len(corpus["anchorRows"]))
            with MotionEnvironment() as environment:
                model = MotionActorCritic("broad-pace-v1", TARGET_TRANSFORM, False, 32, 2)
                offset = 0
                for index, world in enumerate(corpus["worlds"]):
                    source = json.loads((corpus_path / f"world-{index:04d}.json").read_text())
                    if not source["rows"]:
                        continue
                    observation, reset = environment.reset(seed=world["seed"], task=world["task"], configuration=world["configuration"])
                    model.reset_policy_state(reset["referenceDurationSeconds"])
                    for row in source["rows"]:
                        self.assertEqual(model.encode_observation(observation).tolist(), view["rows"][offset]["modelInput"])
                        observation, _, _, _, _ = environment.step(row["teacherAction"])
                        offset += 1
                self.assertEqual(offset, len(view["rows"]))
            self.assertEqual(original, {filename.name: filename.read_bytes() for filename in corpus_path.iterdir() if filename.is_file()})

    def test_actual_long_view_native_training_resume_matches_every_state(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            corpus = load_corpus(corpus_path)
            defaults = arguments(seed=890131, definition=corpus["definition"])
            self.assertNotIn("expertHistoryFrames", configuration(**defaults))
            with self.assertRaises(ValueError):
                configuration(**{**defaults, "expert_history_frames": 16})
            actual = {**defaults, "expert_history_frames": 16, "feature_transform": TARGET_TRANSFORM,
                      "corpus_warm_loss_profile": "pose-budget-v2", "corpus_anchor_source": "full-corpus-equal-world-v1"}
            config = configuration(**actual)
            sources = source_identity(False, False, True)
            self.assertEqual(config["expertCorpusHistoryDefinition"]["targetFeatureSize"], 565)
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
            first = torch.load(root / "complete/checkpoint.pt", weights_only=True)
            second = torch.load(root / "resumed/checkpoint.pt", weights_only=True)
            self.assertTrue(identical(first, second))
            self.assertEqual(complete["policySha256"], resumed["policySha256"])
            self.assertEqual(first["model"]["scales"].shape, (565,))
            self.assertEqual(first["measuredAnchor"]["features"].shape[1], 565)
            self.assertEqual(complete["history"][0]["actualDecisionsThisCollection"], 0)
            self.assertEqual(complete["discardedCollectionStepsTotal"], 0)


if __name__ == "__main__":
    unittest.main()
