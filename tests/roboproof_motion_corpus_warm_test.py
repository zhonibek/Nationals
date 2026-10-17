import inspect
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.corpus_warm import anchor_rows, load_corpus
from roboproof.motion_learning.expert_data import collect
from roboproof.motion_learning.td3_train import configuration, run, source_identity, train_seed


def arguments(seed=890021, updates=4, definition=None):
    defaults = {name: parameter.default for name, parameter in inspect.signature(run).parameters.items()}
    result = {name: defaults[name] for name in inspect.signature(configuration).parameters}
    result.update(seed=seed, updates=updates, max_steps=4096, rollout_steps=16, warm_start_updates=1,
                  gradient_steps=2, replay_capacity=64, expert_dataset_definition=definition)
    return result


def dataset(root):
    worlds = {"schemaVersion": 1, "cases": [{"id": f"corpus-warm-{index}", "group": f"corpus-family-{index}", "seed": 890011 + index,
        "task": {"start": {"xIn": 0.125, "yIn": -0.25, "headingDeg": 0.5},
                 "goal": {"xIn": 0.375, "yIn": 12.25 * (index + 1), "headingDeg": 1}, "deadlineSeconds": 10},
        "configuration": {}} for index in range(3)]}
    filename = root / "worlds.json"
    filename.write_text(json.dumps(worlds))
    collect(filename, root / "dataset", max_queries=5000, max_seconds=90)
    return root / "dataset"


def identical(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(identical(value, second[name]) for name, value in first.items())
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(identical(left, right) for left, right in zip(first, second))
    return first == second


class CorpusWarmTests(unittest.TestCase):
    def test_default_configuration_does_not_add_or_change_a_corpus_declaration(self):
        explicit = arguments()
        omitted = {name: value for name, value in explicit.items() if name != "expert_dataset_definition"}
        self.assertEqual(configuration(**explicit), configuration(**omitted))
        self.assertNotIn("expertDatasetDefinition", configuration(**omitted))
        with self.assertRaises(ValueError):
            configuration(**{**explicit, "expert_dataset_definition": {}})

    def test_anchor_covers_every_world_and_both_trajectory_endpoints(self):
        records = [{"rows": [{"modelInput": [world, step], "teacherAction": [0, 0, 0, 0]} for step in range(32)]}
                   for world in range(91)]
        chosen = anchor_rows(records)
        self.assertEqual(len(chosen), 512)
        self.assertEqual({row["modelInput"][0] for row in chosen}, set(range(91)))
        for world in range(91):
            steps = [row["modelInput"][1] for row in chosen if row["modelInput"][0] == world]
            self.assertIn(0, steps)
            self.assertIn(31, steps)
            self.assertLessEqual(len(steps), 16)

    def test_actual_offline_warm_and_native_td3_resume_match_all_state(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            corpus = load_corpus(corpus_path)
            config = configuration(**arguments(definition=corpus["definition"]))
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
            warm = complete["history"][0]
            self.assertTrue(warm["expertDatasetWarm"])
            self.assertEqual(warm["expertDatasetRowsUsed"], corpus["definition"]["rowCount"])
            self.assertEqual(warm["actualDecisionsThisCollection"], 0)
            self.assertEqual(warm["worlds"], [])
            native = [row for row in complete["history"] if row["phase"] == "td3"]
            self.assertEqual(complete["collectionSteps"], sum(row["acceptedRows"] for row in native))
            self.assertEqual(complete["episodes"], sum(len(row["worlds"]) for row in native))
            self.assertEqual(complete["steps"], sum(row["acceptedRows"] for row in complete["history"]))
            exposure = json.loads((root / "complete/exposure.json").read_text())["cases"]
            self.assertEqual([row["seed"] for row in exposure if row["kind"] == "td3-expert-corpus"],
                             [world["seed"] for world in corpus["worlds"]])
            self.assertEqual(config["expertDatasetDefinition"]["sourceActualQueries"], 185)
            self.assertTrue(complete["actorWeightsChanged"])

    def test_missing_or_changed_corpus_rejects_before_initial_policy_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            definition = load_corpus(corpus_path)["definition"]
            config = configuration(**arguments(definition=definition))
            with self.assertRaises(ValueError):
                configuration(**{**arguments(definition=definition), "feature_transform": "identity-v1"})
            with MotionEnvironment() as environment, self.assertRaisesRegex(ValueError, "required together"):
                train_seed(environment, root / "missing", config["seed"], config, source_identity(), time.monotonic() + 90)
            self.assertFalse((root / "missing").exists())
            with self.assertRaisesRegex(ValueError, "cannot replace"):
                run(directory=root / "bad", max_seconds=90, **arguments(definition=definition))
            changed = corpus_path / "world-0002.json"
            changed.write_text(changed.read_text() + " ")
            with MotionEnvironment() as environment, self.assertRaisesRegex(ValueError, "hash/world mismatch"):
                train_seed(environment, root / "changed", config["seed"], config, source_identity(), time.monotonic() + 90,
                           expert_dataset=corpus_path)
            self.assertFalse((root / "changed").exists())


if __name__ == "__main__":
    unittest.main()
