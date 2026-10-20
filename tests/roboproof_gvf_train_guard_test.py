import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from roboproof.motion_learning import gvf_train


class GvfTrainGuardTests(unittest.TestCase):
    def test_training_module_import_does_not_import_torch(self):
        self.assertNotIn("torch", sys.modules)

    def test_remote_guard_rejects_local_and_unapproved_execution(self):
        with mock.patch.object(gvf_train.sys, "platform", "win32"), \
                mock.patch.object(gvf_train.platform, "system", return_value="Windows"), \
                mock.patch.dict(os.environ, {"ROBOPROOF_APPROVED_REMOTE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "no laptop fallback"):
                gvf_train.require_approved_remote(True)
        with mock.patch.object(gvf_train.sys, "platform", "linux"), \
                mock.patch.object(gvf_train.platform, "system", return_value="Linux"), \
                mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "approved Linux remote"):
                gvf_train.require_approved_remote(True)
            with mock.patch.dict(os.environ, {"ROBOPROOF_APPROVED_REMOTE": "1"}):
                with self.assertRaisesRegex(RuntimeError, "approved Linux remote"):
                    gvf_train.require_approved_remote(False)

    def test_remote_guard_accepts_only_explicit_linux_approval(self):
        with mock.patch.object(gvf_train.sys, "platform", "linux"), \
                mock.patch.object(gvf_train.platform, "system", return_value="Linux"), \
                mock.patch.dict(os.environ, {"ROBOPROOF_APPROVED_REMOTE": "1"}):
            self.assertIsNone(gvf_train.require_approved_remote(True))

    def test_mi300x_probe_requires_rocm_real_device_and_memory_floor(self):
        class FakeCuda:
            def is_available(self):
                return True

            def get_device_properties(self, device):
                self.device = device
                return types.SimpleNamespace(name="AMD Instinct MI300X", total_memory=192 * 1024 ** 3)

        torch = types.SimpleNamespace(version=types.SimpleNamespace(hip="10.1", cuda=None),
            cuda=FakeCuda(), device=lambda value: value, __version__="2.14")
        device, info = gvf_train.verify_mi300x(torch)
        self.assertEqual(device, "cuda:0")
        self.assertEqual(info["name"], "AMD Instinct MI300X")
        torch.version.hip = None
        with self.assertRaisesRegex(RuntimeError, "ROCm PyTorch"):
            gvf_train.verify_mi300x(torch)
        torch.version.hip = "10.1"
        torch.cuda.get_device_properties = lambda device: types.SimpleNamespace(name="Tesla T4", total_memory=192 * 1024 ** 3)
        with self.assertRaisesRegex(RuntimeError, "MI300X"):
            gvf_train.verify_mi300x(torch)
        torch.cuda.get_device_properties = lambda device: types.SimpleNamespace(name="AMD MI300X", total_memory=16 * 1024 ** 3)
        with self.assertRaisesRegex(RuntimeError, "180 GiB"):
            gvf_train.verify_mi300x(torch)

    def test_t4_probe_requires_real_cuda_t4_capability_and_memory(self):
        class FakeCuda:
            def is_available(self):
                return True

            def get_device_properties(self, device):
                return types.SimpleNamespace(name="Tesla T4", total_memory=15 * 1024 ** 3)

            def get_device_capability(self, device):
                return (7, 5)

        torch = types.SimpleNamespace(version=types.SimpleNamespace(hip=None, cuda="13.0"),
            cuda=FakeCuda(), device=lambda value: value, __version__="2.11.0+cu130")
        device, info = gvf_train.verify_t4(torch)
        self.assertEqual(device, "cuda:0")
        self.assertEqual(info["name"], "Tesla T4")
        self.assertEqual(info["capability"], [7, 5])
        torch.version.hip = "rocm"
        with self.assertRaisesRegex(RuntimeError, "CUDA PyTorch"):
            gvf_train.verify_t4(torch)
        torch.version.hip = None
        torch.cuda.get_device_capability = lambda device: (8, 0)
        with self.assertRaisesRegex(RuntimeError, "capability 7.5"):
            gvf_train.verify_t4(torch)
        torch.cuda.get_device_capability = lambda device: (7, 5)
        torch.cuda.get_device_properties = lambda device: types.SimpleNamespace(name="Tesla T4", total_memory=13 * 1024 ** 3)
        with self.assertRaisesRegex(RuntimeError, "14-16 GiB"):
            gvf_train.verify_t4(torch)

    def test_training_limits_are_explicit_and_bounded(self):
        config = gvf_train.validate_limits(77, 128, 1, 60)
        self.assertEqual(config["seed"], 77)
        self.assertEqual(config["networkWidth"], 64)
        self.assertEqual(config["hiddenLayers"], 2)
        self.assertEqual(config["decisionBudgetPolicy"], "finish-current-episode-v1")
        self.assertEqual(config["maxDecisionOvershoot"], gvf_train.MAX_EPISODE_DECISIONS - 1)
        expanded = gvf_train.validate_limits(77, 128, 1, 60, network_width=512, hidden_layers=10)
        self.assertEqual(expanded["networkWidth"], 512)
        self.assertEqual(expanded["hiddenLayers"], 10)
        for values in ((-1, 128, 1, 60), (1, 127, 1, 60), (1, 100001, 1, 60),
                       (1, 128, 0, 60), (1, 128, 1, 59), (1, 128, 1, 3601)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                gvf_train.validate_limits(*values)

    def test_training_rejects_unsupported_network_architectures(self):
        for width, depth in ((128, 3), (1024, 10), (256, 2.0), (True, 10)):
            with self.subTest(width=width, depth=depth), self.assertRaises(ValueError):
                gvf_train.validate_limits(77, 128, 1, 60, width, depth)

    def test_cli_passes_explicit_10_by_512_architecture_to_remote_trainer(self):
        result = {"status": "completed-budget", "decisions": 128, "episodes": 1, "updates": 1}
        arguments = ("--approved-amd-mi300x", "--source-manifest", "source-manifest.json",
            "--source-manifest-sha256", "a" * 64, "--approved-max-credits", "100",
            "--output", "roboproof/runs/motion/gvf-td3/gvf-td3-seed50512-w512-d10-pilot",
            "--seed", "50512", "--max-decisions", "100000", "--max-updates", "100000",
            "--max-seconds", "1800", "--network-width", "512", "--hidden-layers", "10")
        with mock.patch.object(gvf_train, "run", return_value=result) as trainer, \
                contextlib.redirect_stdout(io.StringIO()):
            exit_code = gvf_train.main(arguments)
        self.assertEqual(exit_code, 0)
        self.assertEqual(trainer.call_args.kwargs["network_width"], 512)
        self.assertEqual(trainer.call_args.kwargs["hidden_layers"], 10)
        self.assertEqual(trainer.call_args.kwargs["seed"], 50512)

    def test_cli_accepts_only_explicit_free_t4_without_credit_cap(self):
        result = {"status": "completed-budget", "decisions": 128, "episodes": 1, "updates": 1}
        arguments = ("--approved-free-t4", "--source-manifest", "source-manifest.json",
            "--source-manifest-sha256", "a" * 64,
            "--output", "roboproof/runs/motion/gvf-td3/gvf-td3-seed50515-t4-w512-d10",
            "--seed", "50515", "--max-decisions", "100000", "--max-updates", "100000",
            "--max-seconds", "3600", "--network-width", "512", "--hidden-layers", "10")
        with mock.patch.object(gvf_train, "run", return_value=result) as trainer, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gvf_train.main(arguments), 0)
        self.assertIsNone(trainer.call_args.kwargs["approved_max_credits"])
        self.assertTrue(trainer.call_args.kwargs["approved_free_t4"])
        self.assertFalse(trainer.call_args.kwargs["approved_amd_mi300x"])

    def test_free_t4_credit_override_is_rejected(self):
        with mock.patch.object(gvf_train, "require_approved_remote"), \
                self.assertRaisesRegex(ValueError, "cannot declare or consume paid credits"):
            gvf_train.run(False, "manifest", "a" * 64, Path("unused"), approved_max_credits=0.01,
                approved_free_t4=True)

    def test_t4_snapshot_loader_recomputes_the_frozen_exposure_identity(self):
        filename = Path(gvf_train.ROOT, ".cache/roboproof-t4-study-20261019-v2/training-exposure-ledger.json")
        if not filename.exists():
            self.skipTest("T4 exposure ledger snapshot is not present in local .cache")
        data = filename.read_bytes()
        row = {"path": filename.relative_to(gvf_train.ROOT).as_posix(), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}
        manifest = json.dumps({"schemaVersion": 1, "purpose": gvf_train.MANIFEST_PURPOSE,
            "files": [row]}, separators=(",", ":")).encode()
        result = gvf_train.load_t4_exposure_exclusion(filename, manifest, gvf_train.T4_EXPOSURE_LEDGER_SHA256)
        self.assertEqual(result["sha256"], gvf_train.T4_EXPOSURE_LEDGER_SHA256)
        self.assertEqual(result["knownWorldSeedCount"], 63051)
        self.assertEqual(result["knownBodyGeometryCount"], 61489)
        self.assertEqual(result["reservedTrainingSeeds"], [136001, 136002, 136003])
        json.dumps(result["receipt"], allow_nan=False)
        with self.assertRaisesRegex(ValueError, "independently audited"):
            gvf_train.load_t4_exposure_exclusion(filename, manifest, "0" * 64)

    def test_t4_training_worlds_exclude_known_seeds_geometries_and_prior_fit_duplicates(self):
        class Generator:
            def __init__(self, worlds):
                self.worlds = iter(worlds)

        task = {"start": {"xIn": 0, "yIn": 0, "headingDeg": 0},
                "goal": {"xIn": 20, "yIn": 30, "headingDeg": 90}}
        changed_task = {"start": {"xIn": 0, "yIn": 0, "headingDeg": 0},
                "goal": {"xIn": 40, "yIn": 30, "headingDeg": 90}}
        worlds = [{"seed": 9, "task": task, "configuration": {}},
            {"seed": 10, "task": task, "configuration": {}},
            {"seed": 10, "task": changed_task, "configuration": {}}]
        exposure = {"worldSeeds": {9}, "bodyGeometries": gvf_train._world_geometry_keys(task)}
        seen_seeds, seen_geometry = set(), set()
        fake_train = types.ModuleType("roboproof.motion_learning.train")
        fake_train.training_world = lambda generator, index, curriculum: next(generator.worlds)
        with mock.patch.dict(sys.modules, {"roboproof.motion_learning.train": fake_train}):
            selected, rejected = gvf_train.select_unexposed_training_world(Generator(worlds), 4,
                "mixed-full-reach-v5", exposure, seen_seeds, seen_geometry)
        self.assertEqual(selected["seed"], 10)
        self.assertEqual(rejected, 2)

    def test_operator_credit_cap_is_finite_positive_and_bounded(self):
        self.assertEqual(gvf_train.validate_approved_credits(100), 100.0)
        self.assertEqual(gvf_train.validate_approved_credits(98.47), 98.47)
        for value in (0, -1, 100.01, float("nan"), float("inf"), True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gvf_train.validate_approved_credits(value)

    def test_source_manifest_requires_exact_current_file_hashes(self):
        path = Path(gvf_train.ROOT, "roboproof/motion_learning/gvf_contract.py")
        payload = path.read_bytes()
        row = {"path": "roboproof/motion_learning/gvf_contract.py",
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        manifest = {"schemaVersion": 1, "purpose": gvf_train.MANIFEST_PURPOSE, "files": [row]}
        content = json.dumps(manifest, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "sources.json"
            manifest_path.write_bytes(content)
            self.assertEqual(gvf_train.verify_sources(manifest_path,
                hashlib.sha256(content).hexdigest()), content)
            with self.assertRaisesRegex(ValueError, "manifest SHA256"):
                gvf_train.verify_sources(manifest_path, "0" * 64)
            for rows in ([None], [{**row, "unexpected": True}], [{**row, "bytes": True}],
                         [{**row, "sha256": row["sha256"].upper()}], [row, row]):
                malformed = {"schemaVersion": 1, "purpose": gvf_train.MANIFEST_PURPOSE, "files": rows}
                bad = json.dumps(malformed, separators=(",", ":")).encode()
                manifest_path.write_bytes(bad)
                with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "source inventory"):
                    gvf_train.verify_sources(manifest_path, hashlib.sha256(bad).hexdigest())
            manifest["files"][0]["path"] = "../outside.py"
            bad = json.dumps(manifest, separators=(",", ":")).encode()
            manifest_path.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "Canonical contained source path"):
                gvf_train.verify_sources(manifest_path, hashlib.sha256(bad).hexdigest())


if __name__ == "__main__":
    unittest.main()
