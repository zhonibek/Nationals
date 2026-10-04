"""Local ML preparation tests; fake lifecycle tests never count as AMD execution."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from roboproof.ml.acquire import acquire
from roboproof.ml.data import CONTRACT, digest, features, group_keys, load_export, load_package, normalization, prepare, read_json, split_groups, write_json
from roboproof.ml.jobs import FakeProvider, build_bundle, validate_config, verify_bundle
from roboproof.ml.model import classification_metrics
from roboproof.ml.train import evaluate_final, load_model, tensors, train_model


class MlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary_root = ROOT / "roboproof/runs/ml-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        cls.workspace = tempfile.TemporaryDirectory(dir=temporary_root)
        cls.directory = Path(cls.workspace.name)
        fixture = subprocess.run([shutil.which("node"), str(ROOT / "roboproof/ml/wasm_dataset_fixture.cjs")], capture_output=True, text=True, check=True)
        cls.dataset = cls.directory / "dataset.json"
        cls.dataset.write_text(fixture.stdout, encoding="utf-8")
        cls.package_path = cls.directory / "package.json"
        cls.package = prepare(cls.dataset, cls.package_path, allow_legacy=True)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=self.directory)
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def train(self, output="training", **options):
        return train_model(self.package_path, self.output / output, allow_legacy=True,
                           epochs=4, width=16, batch_size=8, max_seconds=30, **options)

    def test_pre_run_features_exclude_labels_ids_seed_and_inactive_imu(self):
        scenario = copy.deepcopy(self.package["rows"][0]["scenario"])
        names, values = features(scenario)
        scenario.update(scenario_id="renamed", random_seed=999, final_pose=[99, 99, 99], passed=True)
        scenario["environment"]["imu_noise"] = 0.009
        self.assertEqual(features(scenario), (names, values))
        self.assertFalse(any("seed" in name or "error" in name and not name.startswith("environment.initial") for name in names))
        scenario["task"]["localization"] = "encoders-imu"
        self.assertNotEqual(features(scenario)[1], values)
        scenario["environment"]["mass"] = float("nan")
        with self.assertRaises(ValueError):
            features(scenario)

    def test_export_integrity_opt_in_and_nullable_targets(self):
        with self.assertRaises(ValueError):
            load_export(self.dataset)
        data = read_json(self.dataset)
        data["rows"][0]["scenario_sha256"] = "corrupt"
        write_json(self.output / "bad.json", data)
        with self.assertRaisesRegex(ValueError, "Scenario hash"):
            load_export(self.output / "bad.json", True)
        data = read_json(self.dataset)
        data["rows"][0]["categories"] = ["NUMERICAL_FAILURE", "TIMEOUT"]
        data["rows"][0]["metrics"].update(endpoint_error=None, final_heading_error=None)
        write_json(self.output / "masked.json", data)
        _, rows, _ = load_export(self.output / "masked.json", True)
        self.assertEqual(rows[0]["regression_mask"][0], 0)
        self.assertEqual(rows[0]["regression"][0], 0)
        self.assertEqual(rows[0]["labels"][CONTRACT["classification_targets"].index("NUMERICAL_FAILURE")], 1)

    def test_group_split_is_transitive_deterministic_and_complete(self):
        rows = [dict(row_id=str(index), group_keys=[f"seed:{index}"]) for index in range(9)]
        rows[0]["group_keys"].append("lineage:root")
        rows[1]["group_keys"].extend(["lineage:root", "world:paired"])
        rows[2]["group_keys"].append("world:paired")
        assignments = split_groups(rows, 7)
        self.assertEqual(assignments, split_groups(rows, 7))
        self.assertEqual(sorted(index for indices in assignments.values() for index in indices), list(range(9)))
        owner = {index: name for name, indices in assignments.items() for index in indices}
        self.assertEqual(owner[0], owner[1])
        self.assertEqual(owner[1], owner[2])
        with self.assertRaises(ValueError):
            split_groups(rows[:3], 7)

    def test_package_detects_corruption_leakage_and_train_only_preprocessing(self):
        package = load_package(self.package_path, True)
        self.assertEqual(package["preprocessing"], normalization(package["rows"], package["assignments"]["train"]))
        scenario = copy.deepcopy(package["rows"][0]["scenario"])
        paired = copy.deepcopy(scenario)
        paired["controller"]["reference_time_scale"] = 2
        self.assertEqual(group_keys(scenario, None), group_keys(paired, None))
        package["rows"][0]["features"][0] = 0
        write_json(self.output / "corrupt.json", package)
        with self.assertRaisesRegex(ValueError, "integrity"):
            load_package(self.output / "corrupt.json", True)
        package["package_sha256"] = digest({name: value for name, value in package.items() if name != "package_sha256"})
        write_json(self.output / "forged.json", package)
        with self.assertRaisesRegex(ValueError, "Feature"):
            load_package(self.output / "forged.json", True)
        package = copy.deepcopy(self.package)
        package["assignments"]["validation"].append(package["assignments"]["train"][0])
        package["package_sha256"] = digest({name: value for name, value in package.items() if name != "package_sha256"})
        write_json(self.output / "leaky.json", package)
        with self.assertRaisesRegex(ValueError, "split"):
            load_package(self.output / "leaky.json", True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(self.dataset, self.package_path, allow_legacy=True)

    def test_real_gradient_training_is_repeatable_and_never_evaluates_final(self):
        accessed = []

        def recording_tensors(package, split, device):
            accessed.append(split)
            return tensors(package, split, device)

        with patch("roboproof.ml.train.tensors", side_effect=recording_tensors):
            card = self.train()
        self.assertNotIn("final", accessed)
        self.assertTrue(card["weights_changed"])
        self.assertTrue(card["actual_gradient_training"])
        self.assertFalse(card["amd_training"])
        self.assertEqual(card["status"], "EXPERIMENTAL_LEGACY_MODEL")
        self.assertIn("logistic_baseline", card["validation"])
        self.assertEqual(card["final_evaluation"], "NOT RUN; separate one-shot command required")
        self.train("second")
        first = torch.load(self.output / "training/model.pt", weights_only=True)
        second = torch.load(self.output / "second/model.pt", weights_only=True)
        self.assertTrue(all(torch.equal(first["model"][name], second["model"][name]) for name in first["model"]))
        artifact, model, device = load_model(self.output / "training/model.pt")
        with torch.no_grad():
            logits, regression = model(tensors(self.package, "validation", device)[0])
        self.assertEqual(logits.shape[1], len(CONTRACT["classification_targets"]))
        self.assertEqual(regression.shape[1], len(CONTRACT["regression_targets"]))
        self.assertEqual(artifact["package_sha256"], self.package["package_sha256"])

    def test_interrupted_training_resumes_identically_and_rejects_changed_inputs(self):
        calls = 0

        def interrupt(deadline):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise TimeoutError("test interruption after first epoch checkpoint")

        with patch("roboproof.ml.train.check_time", side_effect=interrupt):
            with self.assertRaises(TimeoutError):
                self.train("resumed")
        checkpoint = self.output / "resumed/checkpoint.pt"
        self.assertTrue(checkpoint.exists())
        with self.assertRaisesRegex(ValueError, "identical"):
            self.train("resumed", resume=checkpoint, seed=777)
        self.train("resumed", resume=checkpoint)
        self.train("fresh")
        resumed = torch.load(self.output / "resumed/model.pt", weights_only=True)
        fresh = torch.load(self.output / "fresh/model.pt", weights_only=True)
        self.assertTrue(all(torch.equal(resumed["model"][name], fresh["model"][name]) for name in fresh["model"]))
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.train("resumed", resume=checkpoint)

    def test_training_limits_and_no_false_rocm_fallback(self):
        for options in [dict(epochs=501), dict(max_seconds=0), dict(learning_rate=2)]:
            with self.assertRaises(ValueError):
                train_model(self.package_path, self.output / "bad", allow_legacy=True, **options)
        with patch("torch.version.hip", None), patch("torch.cuda.is_available", return_value=True):
            with self.assertRaises(RuntimeError):
                self.train(device_name="rocm")

    def test_final_evaluation_requires_explicit_unlock_and_consumes_holdout_once(self):
        package_path = self.output / "evaluation-package.json"
        write_json(package_path, self.package)
        self.train()
        model_path = self.output / "training/model.pt"
        with self.assertRaisesRegex(ValueError, "unlock"):
            evaluate_final(package_path, model_path, self.output / "final.json", True)
        report = evaluate_final(package_path, model_path, self.output / "final.json", True, True)
        self.assertEqual(report["split"], "final")
        with self.assertRaises(FileExistsError):
            evaluate_final(package_path, model_path, self.output / "again.json", True, True)

    def test_metrics_handle_tied_predictions_absent_classes_and_false_negatives(self):
        metrics = classification_metrics(torch.tensor([[0.1, 0.8], [0.1, 0.8], [0.1, 0.8]]),
                                         torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), ["rare", "absent"])
        self.assertEqual(metrics["rare"]["false_negatives"], 2)
        self.assertAlmostEqual(metrics["rare"]["average_precision"], 2 / 3)
        self.assertIsNone(metrics["absent"]["recall"])
        self.assertIsNone(metrics["absent"]["average_precision"])

    def test_acquisition_excludes_all_split_worlds_and_keeps_random_exploration(self):
        self.train()
        candidates = [copy.deepcopy(row["scenario"]) for row in self.package["rows"]]
        for index in range(16):
            scenario = copy.deepcopy(candidates[0])
            scenario["scenario_id"] = f"new-{index}"
            scenario["random_seed"] = 99000 + index
            scenario["environment"]["friction"] = 0.1 + index / 30
            scenario["environment"]["inertia"] = 0.19
            candidates.append(scenario)
        write_json(self.output / "pool.json", candidates)
        selection = acquire(self.output / "training/model.pt", self.package_path, self.output / "pool.json",
                            self.output / "selection.json", budget=8, allow_legacy=True)
        self.assertEqual(selection["excluded_count"], len(self.package["rows"]))
        self.assertEqual(selection["exploration_count"], 2)
        self.assertEqual(len(selection["scenarios"]), 8)
        self.assertTrue(all(scenario["scenario_id"].startswith("new-") for scenario in selection["scenarios"]))
        self.assertTrue(selection["simulation_status"].startswith("NOT RUN"))
        again = acquire(self.output / "training/model.pt", self.package_path, self.output / "pool.json",
                        self.output / "again.json", budget=8, allow_legacy=True)
        self.assertEqual(selection, again)

    def test_bundle_integrity_allowlist_and_fail_closed_remote_configuration(self):
        config_path = ROOT / "roboproof/ml/amd-job.example.json"
        manifest = build_bundle(self.package_path, config_path, self.output / "bundle", True)
        archive = self.output / "bundle/job.tar.gz"
        self.assertTrue(verify_bundle(archive, manifest))
        self.assertFalse(manifest["remote_enabled"])
        self.assertTrue(all(".env" not in name and "Pedro" not in name for name in manifest["files"]))
        corrupted = copy.deepcopy(manifest)
        corrupted["archive_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "hash"):
            verify_bundle(archive, corrupted)
        for update in [dict(remote_enabled=True), dict(provider="guessed-endpoint")]:
            config = read_json(config_path)
            config.update(update)
            with self.assertRaises(ValueError):
                validate_config(config)
        config = read_json(config_path)
        config["limits"]["max_spend_usd"] = 1
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_fake_lifecycle_budget_cancel_timeout_logs_and_verified_download(self):
        manifest = build_bundle(self.package_path, ROOT / "roboproof/ml/amd-job.example.json", self.output / "bundle", True)
        archive = self.output / "bundle/job.tar.gz"
        provider = FakeProvider(manifest["limits"])
        job = provider.submit(manifest, archive)
        self.assertTrue(provider.status(job)["fake"])
        self.assertIn("no GPU", provider.logs(job)[0])
        with self.assertRaises(ValueError):
            provider.submit(manifest, archive)
        with self.assertRaises(ValueError):
            provider.download(job, "model.pt", "none")
        provider.complete(job, {"model.pt": b"fixture"})
        self.assertEqual(provider.download(job, "model.pt", hashlib.sha256(b"fixture").hexdigest()), b"fixture")
        with self.assertRaises(ValueError):
            provider.download(job, "model.pt", "bad")
        cancelled = FakeProvider(manifest["limits"])
        self.assertEqual(cancelled.cancel(cancelled.submit(manifest, archive))["state"], "CANCELLED")
        timed = FakeProvider(manifest["limits"])
        job = timed.submit(manifest, archive)
        timed.advance(job, manifest["limits"]["max_wall_seconds"])
        self.assertEqual(timed.status(job)["state"], "TIMED_OUT")


if __name__ == "__main__":
    unittest.main()
