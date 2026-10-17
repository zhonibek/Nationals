import hashlib
import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from roboproof.motion_learning.corpus_portability import admit_transfer, load_transfer
from roboproof.motion_learning.corpus_warm import imitation, load_corpus
from roboproof.motion_learning.model import MotionActorCritic
from roboproof_motion_corpus_warm_test import dataset, identical


def sha(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()


class CorpusPortabilityTests(unittest.TestCase):
    def test_actual_strict_admission_preserves_data_and_default_runtime_guard(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            expected = load_corpus(corpus_path)
            original_manifest = (corpus_path / "dataset.json").read_bytes()
            receipt_file = root / "transfer.json"
            receipt = admit_transfer(corpus_path, receipt_file)
            self.assertEqual(receipt["newNativeQueries"], 0)
            self.assertEqual(receipt["admissionTorchVersion"], str(torch.__version__))
            with patch("torch.__version__", "different-build+cuda"):
                with self.assertRaisesRegex(ValueError, "source/runtime-bound"):
                    load_corpus(corpus_path)
                transferred = load_transfer(corpus_path, receipt_file, sha(receipt_file))
            self.assertEqual({name: transferred[name] for name in expected}, expected)
            self.assertEqual(transferred["portability"]["consumerTorchVersion"], "different-build+cuda")
            self.assertEqual(original_manifest, (corpus_path / "dataset.json").read_bytes())
            with self.assertRaises(FileExistsError):
                admit_transfer(corpus_path, receipt_file)

    def test_changed_late_record_rejects_before_consumer_engine_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            receipt_file = root / "transfer.json"
            admit_transfer(corpus_path, receipt_file)
            changed = corpus_path / "world-0002.json"
            changed.write_text(changed.read_text() + " ")
            with patch("roboproof.motion_learning.corpus_portability.MotionEnvironment") as environment:
                with self.assertRaisesRegex(ValueError, "hash/world mismatch"):
                    load_transfer(corpus_path, receipt_file, sha(receipt_file))
                environment.assert_not_called()

    def test_bad_receipts_cannot_relabel_producer_runtime_sources_or_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            receipt_file = root / "transfer.json"
            original = admit_transfer(corpus_path, receipt_file)
            with patch("roboproof.motion_learning.corpus_portability.MotionEnvironment") as environment:
                with self.assertRaises(ValueError):
                    load_transfer(corpus_path, receipt_file, "0" * 64)
                for index, changes in enumerate(({"producerTorchVersion": "invented", "admissionTorchVersion": "invented"},
                                                  {"sources": {}}, {"consumerSources": {}}, {"contentsSha256": "0" * 64},
                                                  {"newNativeQueries": 1}, {"newNativeQueries": False},
                                                  {"schemaVersion": True}, {"admissionPlatform": []},
                                                  {"consumerEngineRuntime": {"platform": "invented"}}, {"extra": True})):
                    changed = root / f"bad-{index}.json"
                    changed.write_text(json.dumps({**original, **changes}))
                    with self.assertRaises(ValueError):
                        load_transfer(corpus_path, changed, sha(changed))
                environment.assert_not_called()

    def test_explicit_host_runtime_transfer_preserves_every_engine_source_and_wasm_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            producer_runtime = json.loads((corpus_path / "dataset.json").read_text())["engineIdentity"]["runtime"]
            runtime = {"node": "v20.19.0", "v8": "11.3.244.8-node.26",
                       "platform": "linux" if producer_runtime["platform"] != "linux" else "win32", "arch": "x64"}
            receipt_file = root / "transfer.json"
            receipt = admit_transfer(corpus_path, receipt_file, runtime)
            consumer_identity = {**receipt["engineIdentity"], "runtime": runtime}
            with patch("roboproof.motion_learning.corpus_portability.MotionEnvironment") as environment:
                environment.return_value.__enter__.return_value.contract = {"identity": consumer_identity}
                transferred = load_transfer(corpus_path, receipt_file, sha(receipt_file))
                self.assertEqual(transferred["portability"]["consumerEngineRuntime"], runtime)
                self.assertTrue(transferred["portability"]["engineSourceIdentityPreserved"])
                self.assertFalse(transferred["portability"]["engineRuntimeParityVerified"])
                for changed in ({"controllerWasmSha256": "0" * 64}, {"canonicalSimulator": False},
                                {"sources": {}}, {"runtime": {**runtime, "node": "v99.0.0"}}, {"unexpected": True}):
                    environment.return_value.__enter__.return_value.contract = {"identity": {**consumer_identity, **changed}}
                    with self.assertRaises(ValueError):
                        load_transfer(corpus_path, receipt_file, sha(receipt_file))
            strict_file = root / "strict.json"
            admit_transfer(corpus_path, strict_file)
            with patch("roboproof.motion_learning.corpus_portability.MotionEnvironment") as environment:
                environment.return_value.__enter__.return_value.contract = {"identity": consumer_identity}
                with self.assertRaisesRegex(ValueError, "without explicit runtime transfer"):
                    load_transfer(corpus_path, strict_file, sha(strict_file))
            for malformed in ({}, {**runtime, "node": "invented"}, {**runtime, "arch": []},
                              {**runtime, "v8": "invalid\nvalue"}):
                with self.assertRaises(ValueError):
                    admit_transfer(corpus_path, root / "invalid.json", copy.deepcopy(malformed))

    def test_actual_existing_imitation_update_has_identical_state_after_transfer(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = dataset(root)
            receipt_file = root / "transfer.json"
            original = load_corpus(corpus_path)
            admit_transfer(corpus_path, receipt_file)
            transferred = load_transfer(corpus_path, receipt_file, sha(receipt_file))
            results = []
            for corpus in (original, transferred):
                torch.manual_seed(890093)
                model = MotionActorCritic("broad-pace-v1", "deadline-context-v1", False, 32, 2)
                optimizer = torch.optim.Adam([*model.body.parameters(), *model.actor.parameters()], lr=0.0003)
                before = model.actor.weight.detach().clone()
                metrics = imitation(model, optimizer, corpus["rows"], "pose-budget-v2", epochs=2)
                self.assertFalse(torch.equal(before, model.actor.weight))
                results.append({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": metrics,
                                "rng": torch.get_rng_state()})
            self.assertTrue(identical(results[0], results[1]))


if __name__ == "__main__":
    unittest.main()
