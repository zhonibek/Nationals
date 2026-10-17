import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from roboproof.motion_env import MotionEnvironment
from roboproof.motion_learning.corpus_portability import admit_transfer, load_transfer
from roboproof.motion_learning.corpus_warm import load_corpus
from roboproof.motion_learning.td3_train import configuration, load_expert_corpus, run, source_identity, train_seed
from roboproof_motion_corpus_warm_test import arguments, dataset, identical


def sha(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()


class TD3CorpusTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.corpus_path = dataset(cls.root)
        cls.receipt_file = cls.root / "transfer.json"
        admit_transfer(cls.corpus_path, cls.receipt_file)
        cls.receipt_sha = sha(cls.receipt_file)
        cls.corpus = load_transfer(cls.corpus_path, cls.receipt_file, cls.receipt_sha)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def actual(self, seed=890101, updates=4):
        return {**arguments(seed=seed, updates=updates, definition=self.corpus["definition"]),
                "n_steps": 32, "corpus_warm_loss_profile": "pose-budget-v2",
                "corpus_anchor_source": "full-corpus-equal-world-v1"}

    def transferred(self):
        return {"expert_dataset": self.corpus_path, "expert_dataset_attestation": self.receipt_file,
                "expert_dataset_attestation_sha256": self.receipt_sha}

    def test_transfer_is_explicit_source_bound_and_not_added_to_default_configuration(self):
        default = configuration(**self.actual())
        self.assertNotIn("expertDatasetTransfer", default)
        explicit = configuration(**{**self.actual(), "expert_dataset_transfer": self.corpus["portability"]})
        self.assertEqual({name: value for name, value in explicit.items() if name != "expertDatasetTransfer"}, default)
        self.assertEqual(explicit["expertDatasetTransfer"], self.corpus["portability"])
        self.assertNotIn("roboproof/motion_learning/corpus_portability.py", source_identity())
        self.assertIn("roboproof/motion_learning/corpus_portability.py", source_identity(True))
        for changed in ({"attestationSha256": "0"}, {"consumerTorchVersion": "invented"},
                        {"newNativeQueries": False}, {"engineRuntimeParityVerified": True},
                        {"engineSourceIdentityPreserved": 1}, {"consumerEngineRuntime": {}}, {"extra": True}):
            with self.assertRaises(ValueError):
                configuration(**{**self.actual(), "expert_dataset_transfer": {**self.corpus["portability"], **changed}})
        with self.assertRaises(ValueError):
            configuration(**{**self.actual(), "expert_dataset_definition": None,
                             "expert_dataset_transfer": self.corpus["portability"]})

    def test_missing_or_changed_transfer_fails_before_run_publication_or_device_setup(self):
        for index, supplied in enumerate((
            {"expert_dataset_attestation": self.receipt_file},
            {"expert_dataset_attestation_sha256": self.receipt_sha},
            {"expert_dataset_attestation": self.receipt_file, "expert_dataset_attestation_sha256": "0" * 64},
            {**self.transferred(), "expert_dataset_transfer": {**self.corpus["portability"], "attestationSha256": "0" * 64}},
        )):
            directory = self.root / f"invalid-{index}"
            with patch("roboproof.motion_learning.td3_train.prepare_execution") as execution, self.assertRaises(ValueError):
                run(directory=directory, **{**self.actual(), **supplied})
            execution.assert_not_called()
            self.assertFalse(directory.exists())
        with self.assertRaisesRegex(ValueError, "path required"):
            load_expert_corpus(None, attestation_file=self.receipt_file, attestation_sha256=self.receipt_sha)
        with patch("torch.__version__", "2.8.0+consumer-test"):
            with patch("roboproof.motion_learning.td3_train.prepare_execution") as execution, self.assertRaisesRegex(ValueError, "source/runtime-bound"):
                run(directory=self.root / "default-other-build", expert_dataset=self.corpus_path, **self.actual())
            execution.assert_not_called()
            transferred = load_expert_corpus(self.corpus_path, self.corpus["definition"], self.receipt_file, self.receipt_sha)
            self.assertEqual(transferred["portability"]["consumerTorchVersion"], "2.8.0+consumer-test")
            self.assertEqual(transferred["rows"], self.corpus["rows"])

    def test_active_engine_sources_and_declared_provenance_reject_before_initial_actor(self):
        config = configuration(**{**self.actual(), "expert_dataset_transfer": self.corpus["portability"]})
        with MotionEnvironment() as environment:
            bad_identity = {**environment.contract["identity"], "controllerWasmSha256": "0" * 64}
        invalid = SimpleNamespace(contract={"identity": bad_identity})
        with patch("roboproof.motion_learning.td3_train.prepare_execution") as execution, self.assertRaisesRegex(ValueError, "Active native"):
            train_seed(invalid, self.root / "bad-engine", config["seed"], config, source_identity(True), time.monotonic() + 90,
                       **self.transferred())
        execution.assert_not_called()
        self.assertFalse((self.root / "bad-engine").exists())
        with MotionEnvironment() as environment:
            with self.assertRaisesRegex(ValueError, "sources required"):
                train_seed(environment, self.root / "bad-sources", config["seed"], config, source_identity(), time.monotonic() + 90,
                           **self.transferred())
            with self.assertRaisesRegex(ValueError, "attestation required together"):
                train_seed(environment, self.root / "missing-attestation", config["seed"], config,
                           source_identity(True), time.monotonic() + 90, expert_dataset=self.corpus_path)
        self.assertFalse((self.root / "bad-sources").exists())

    def test_actual_native_transferred_training_and_resume_preserve_all_learning_state(self):
        original_manifest = (self.corpus_path / "dataset.json").read_bytes()
        default = run(directory=self.root / "default", max_seconds=90, expert_dataset=self.corpus_path, **self.actual())
        complete = run(directory=self.root / "complete", max_seconds=90, **self.actual(), **self.transferred())
        prefix = run(directory=self.root / "resumed", max_seconds=90, **self.actual(updates=3), **self.transferred())
        manifest_path = self.root / "resumed" / prefix["runId"] / "run.json"
        before = manifest_path.read_bytes()
        alternate = self.root / "alternate.json"
        alternate.write_text(json.dumps(json.loads(self.receipt_file.read_text()), separators=(",", ":")))
        self.assertNotEqual(sha(alternate), self.receipt_sha)
        with self.assertRaisesRegex(ValueError, "source/configuration mismatch"):
            run(directory=self.root / "resumed", max_seconds=90, resume=prefix["runId"], **self.actual(),
                **{**self.transferred(), "expert_dataset_attestation": alternate, "expert_dataset_attestation_sha256": sha(alternate)})
        self.assertEqual(before, manifest_path.read_bytes())
        resumed = run(directory=self.root / "resumed", max_seconds=90, resume=prefix["runId"], **self.actual(), **self.transferred())
        self.assertEqual([item["status"] for item in (default, complete, resumed)], ["completed"] * 3)
        saved = []
        for label, manifest in (("default", default), ("complete", complete), ("resumed", resumed)):
            path = self.root / label / manifest["runId"] / f"seed-{manifest['config']['seed']}"
            saved.append(torch.load(path / "checkpoint.pt", weights_only=True, map_location="cpu"))
        self.assertTrue(identical(saved[1], saved[2]))
        baseline, transferred = copy.deepcopy(saved[0]), copy.deepcopy(saved[1])
        transferred["config"].pop("expertDatasetTransfer")
        baseline.pop("sources")
        transferred.pop("sources")
        self.assertTrue(identical(baseline, transferred))
        self.assertEqual([item["models"][0]["policySha256"] for item in (default, complete, resumed)],
                         [default["models"][0]["policySha256"]] * 3)
        self.assertTrue(all(row["nativeAnchorRows"] == len(self.corpus["rows"]) for row in saved[1]["history"][1:]))
        self.assertGreater(saved[1]["collectionSteps"], 0)
        self.assertEqual(saved[1]["collectionSteps"], sum(row["actualDecisionsThisCollection"] for row in saved[1]["history"]))
        self.assertEqual(saved[1]["discardedCollectionStepsTotal"], 0)
        self.assertFalse(complete["learnedImprovementVerified"])
        self.assertEqual(original_manifest, (self.corpus_path / "dataset.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
