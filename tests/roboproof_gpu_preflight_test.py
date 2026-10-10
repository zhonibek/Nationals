"""Metadata-only preflight tests: no PyTorch import, hardware queries or workloads."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.gpu import preflight


def runtime(hip="6.4", cuda=None, available=True, count=1):
    properties = SimpleNamespace(name="AMD test fixture", total_memory=8 * 1024 ** 3,
                                 gcnArchName="gfx-fixture")
    return SimpleNamespace(
        __version__="fixture-torch", version=SimpleNamespace(hip=hip, cuda=cuda),
        get_num_threads=Mock(return_value=1),
        cuda=Mock(spec=["is_available", "device_count", "get_device_properties"],
                  is_available=Mock(return_value=available), device_count=Mock(return_value=count),
                  get_device_properties=Mock(return_value=properties)),
    )


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.host_patches = [
            patch.object(preflight.platform, "platform", return_value="fixture-platform"),
            patch.object(preflight.platform, "machine", return_value="fixture-machine"),
            patch.object(preflight.platform, "processor", return_value="fixture-cpu"),
            patch.object(preflight.os, "cpu_count", return_value=2),
        ]
        for host_patch in self.host_patches:
            host_patch.start()
            self.addCleanup(host_patch.stop)

    def collect(self, torch=None, max_devices=preflight.DEFAULT_MAX_DEVICES, error=None):
        with patch.object(preflight.importlib, "import_module", return_value=torch, side_effect=error) as loader:
            report = preflight._collect(max_devices)
        loader.assert_called_once_with("torch")
        json.dumps(report, allow_nan=False)
        return report

    def ready_report(self, max_devices=preflight.DEFAULT_MAX_DEVICES):
        return self.collect(runtime(), max_devices)

    def test_hip_metadata_and_legacy_keys_without_workload_apis(self):
        torch = runtime()
        report = self.collect(torch)
        self.assertEqual(report["status"], "ready")
        self.assertTrue(report["rocm_ready"])
        self.assertEqual(report["metadata"]["device"], "rocm")
        for key in ("torch", "hip", "python", "platform", "gpu", "cpu_threads"):
            self.assertIn(key, report["metadata"])
        self.assertEqual(report["metadata"]["gpu"], "AMD test fixture")
        self.assertEqual(report["metadata"]["gpu_memory_bytes"], 8 * 1024 ** 3)
        self.assertEqual(report["metadata"]["gpu_architecture"], "gfx-fixture")
        self.assertEqual(report["cpu"]["status"], "torch_imported")
        self.assertFalse(report["cpu"]["operations_tested"])
        self.assertEqual(report["rocm"]["devices"][0]["index"], 0)
        self.assertEqual(report["errors"], [])
        self.assertEqual([call[0] for call in torch.cuda.mock_calls],
                         ["is_available", "device_count", "get_device_properties"])
        torch.get_num_threads.assert_called_once_with()

    def test_cpu_only_and_nvidia_builds_never_query_gpu(self):
        for cuda, reason in ((None, "cpu_only_build"), ("12.8", "non_hip_build")):
            with self.subTest(cuda=cuda):
                torch = runtime(hip=None, cuda=cuda, available=True)
                report = self.collect(torch)
                self.assertEqual(report["status"], "blocked")
                self.assertEqual(report["reason"], reason)
                self.assertFalse(report["rocm_ready"])
                self.assertEqual(report["metadata"]["device"], "cpu")
                self.assertFalse(report["rocm"]["hip_build"])
                self.assertIsNone(report["rocm"]["device_available"])
                self.assertEqual(torch.cuda.mock_calls, [])

    def test_empty_hip_and_ambiguous_builds_are_blocked(self):
        for hip, cuda, reason in ((" ", None, "cpu_only_build"), ("6.4", "12.8", "ambiguous_hip_cuda_build")):
            with self.subTest(hip=hip, cuda=cuda):
                torch = runtime(hip=hip, cuda=cuda)
                report = self.collect(torch)
                self.assertEqual(report["reason"], reason)
                self.assertFalse(report["rocm_ready"])
                self.assertEqual(torch.cuda.mock_calls, [])

    def test_invalid_version_types_are_not_hip_evidence(self):
        for hip, cuda in ((True, None), (False, None), (0, None), ("6.4", True)):
            with self.subTest(hip=hip, cuda=cuda):
                torch = runtime(hip=hip, cuda=cuda)
                report = self.collect(torch)
                self.assertEqual(report["status"], "error")
                self.assertFalse(report["rocm_ready"])
                self.assertEqual(torch.cuda.mock_calls, [])

    def test_hip_build_without_available_device_is_blocked(self):
        torch = runtime(available=False)
        report = self.collect(torch)
        self.assertEqual(report["reason"], "hip_device_unavailable")
        self.assertTrue(report["rocm"]["hip_build"])
        self.assertFalse(report["rocm"]["device_available"])
        self.assertFalse(report["rocm_ready"])
        torch.cuda.device_count.assert_not_called()
        torch.cuda.get_device_properties.assert_not_called()

    def test_available_with_zero_devices_is_not_ready(self):
        torch = runtime(count=0)
        report = self.collect(torch)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["reason"], "no_visible_hip_devices")
        torch.cuda.get_device_properties.assert_not_called()

    def test_device_enumeration_is_capped_and_disclosed(self):
        torch = runtime(count=100)
        report = self.collect(torch, max_devices=2)
        self.assertTrue(report["rocm_ready"])
        self.assertEqual(report["rocm"]["device_count"], 100)
        self.assertTrue(report["rocm"]["devices_truncated"])
        self.assertEqual([device["index"] for device in report["rocm"]["devices"]], [0, 1])
        self.assertEqual(torch.cuda.get_device_properties.call_count, 2)

    def test_optional_architecture_can_be_absent(self):
        torch = runtime()
        del torch.cuda.get_device_properties.return_value.gcnArchName
        report = self.collect(torch)
        self.assertTrue(report["rocm_ready"])
        self.assertIsNone(report["metadata"]["gpu_architecture"])

    def test_missing_torch_is_blocked_but_missing_dependency_is_error(self):
        for name, status, cpu_status in (("torch", "blocked", "pytorch_missing"),
                                         ("torch_dependency", "error", "import_error")):
            with self.subTest(name=name):
                report = self.collect(error=ModuleNotFoundError("fixture missing import", name=name))
                self.assertEqual(report["status"], status)
                self.assertEqual(report["cpu"]["status"], cpu_status)
                self.assertFalse(report["rocm_ready"])
                self.assertIsNone(report["metadata"]["device"])
                self.assertEqual(report["errors"][0]["stage"], "torch_import")

    def test_import_failure_reports_error_without_claiming_cpu_operation(self):
        report = self.collect(error=OSError("fixture DLL load failed"))
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["cpu"]["status"], "import_error")
        self.assertFalse(report["cpu"]["operations_tested"])

    def test_query_errors_preserve_evidence_and_fail_closed(self):
        for method, stage in (("is_available", "device_availability"),
                              ("device_count", "device_count"),
                              ("get_device_properties", "device_properties:0")):
            with self.subTest(method=method):
                torch = runtime()
                getattr(torch.cuda, method).side_effect = RuntimeError("fixture driver error")
                report = self.collect(torch)
                self.assertEqual(report["status"], "error")
                self.assertFalse(report["rocm_ready"])
                self.assertEqual(report["metadata"]["hip"], "6.4")
                self.assertIsNone(report["metadata"]["gpu"])
                self.assertEqual(report["errors"][0]["stage"], stage)

    def test_partial_device_metadata_does_not_pass(self):
        torch = runtime(count=2)
        first = torch.cuda.get_device_properties.return_value
        torch.cuda.get_device_properties.side_effect = [first, RuntimeError("fixture device failure")]
        report = self.collect(torch)
        self.assertFalse(report["rocm_ready"])
        self.assertEqual(len(report["rocm"]["devices"]), 1)
        self.assertEqual(report["errors"][0]["stage"], "device_properties:1")

    def test_invalid_query_values_do_not_pass(self):
        cases = [("available", "yes"), ("count", -1), ("count", True),
                 ("memory", 0), ("memory", True), ("name", " "), ("name", None),
                 ("name", 1), ("threads", 0)]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                torch = runtime()
                if field == "available":
                    torch.cuda.is_available.return_value = value
                elif field == "count":
                    torch.cuda.device_count.return_value = value
                elif field == "threads":
                    torch.get_num_threads.return_value = value
                else:
                    setattr(torch.cuda.get_device_properties.return_value,
                            "total_memory" if field == "memory" else "name", value)
                report = self.collect(torch)
                self.assertEqual(report["status"], "error")
                self.assertFalse(report["rocm_ready"])

    def test_text_and_error_details_are_bounded(self):
        torch = runtime()
        torch.cuda.get_device_properties.return_value.name = "A" * 10000
        report = self.collect(torch)
        self.assertEqual(len(report["metadata"]["gpu"]), preflight.MAX_TEXT_LENGTH)
        report = self.collect(error=RuntimeError("E" * 10000))
        self.assertEqual(len(report["errors"][0]["message"]), preflight.MAX_TEXT_LENGTH)

    def test_public_probe_uses_current_interpreter_and_finite_timeout(self):
        expected = self.ready_report()
        completed = subprocess.CompletedProcess([], 0, json.dumps(expected).encode("utf-8"))
        with patch.object(preflight.subprocess, "run", return_value=completed) as runner:
            report = preflight.probe(timeout_seconds=3, max_devices=2)
        self.assertEqual(report, expected)
        arguments, options = runner.call_args
        self.assertEqual(arguments[0], [sys.executable, "-B", str(Path(preflight.__file__).resolve()), "--_worker", "2"])
        self.assertEqual(options["timeout"], 3)
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertEqual(options["stderr"], subprocess.DEVNULL)
        self.assertNotIn("shell", options)

    def test_timeout_is_structured_and_never_ready(self):
        error = subprocess.TimeoutExpired("fixture probe", 0.1)
        with patch.object(preflight.subprocess, "run", side_effect=error):
            report = preflight.probe(timeout_seconds=0.1)
        self.assertEqual(report["status"], "timeout")
        self.assertFalse(report["rocm_ready"])
        self.assertEqual(report["cpu"]["status"], "unverified")
        self.assertIsNone(report["rocm"]["device_available"])

    def test_invalid_limits_fail_before_launch(self):
        for timeout in (0, -1, 61, float("nan"), float("inf"), True, "15", None):
            with self.subTest(timeout=timeout), patch.object(preflight.subprocess, "run") as runner:
                with self.assertRaises(ValueError):
                    preflight.probe(timeout_seconds=timeout)
                runner.assert_not_called()
        for limit in (0, -1, 17, True, 1.5, "2"):
            with self.subTest(limit=limit), patch.object(preflight.subprocess, "run") as runner:
                with self.assertRaises(ValueError):
                    preflight.probe(max_devices=limit)
                runner.assert_not_called()

    def test_failed_launch_invalid_json_and_oversized_output_are_errors(self):
        results = [subprocess.CompletedProcess([], 9, b""),
                   subprocess.CompletedProcess([], 0, b"not json"),
                   subprocess.CompletedProcess([], 0, b"[]"),
                   subprocess.CompletedProcess([], 0, b"x" * (preflight.MAX_REPORT_BYTES + 1))]
        for result in results:
            with self.subTest(returncode=result.returncode, size=len(result.stdout)):
                with patch.object(preflight.subprocess, "run", return_value=result):
                    report = preflight.probe()
                self.assertEqual(report["status"], "error")
                self.assertFalse(report["rocm_ready"])
        with patch.object(preflight.subprocess, "run", side_effect=OSError("fixture launch failed")):
            self.assertEqual(preflight.probe()["status"], "error")

    def test_ready_report_requires_hip_availability_and_complete_memory_metadata(self):
        for field in ("hip", "cuda", "device_available", "memory", "devices", "rocm_ready"):
            with self.subTest(field=field):
                report = self.ready_report()
                if field == "hip":
                    report["metadata"]["hip"] = None
                elif field == "cuda":
                    report["metadata"]["cuda"] = "12.8"
                elif field == "device_available":
                    report["rocm"]["device_available"] = False
                elif field == "memory":
                    report["rocm"]["devices"][0]["total_memory_bytes"] = 0
                elif field == "devices":
                    report["rocm"]["devices"] = []
                else:
                    report["rocm_ready"] = False
                result = subprocess.CompletedProcess([], 0, json.dumps(report).encode("utf-8"))
                with patch.object(preflight.subprocess, "run", return_value=result):
                    self.assertEqual(preflight.probe()["status"], "error")

    def test_cli_emits_json_with_non_success_for_blocked_error_and_timeout(self):
        for status, code in (("ready", 0), ("blocked", 2), ("error", 1), ("timeout", 1)):
            with self.subTest(status=status):
                report = self.ready_report() if status == "ready" else preflight._failure(preflight._report(), status, "fixture")
                output = io.StringIO()
                with patch.object(preflight, "probe", return_value=report), contextlib.redirect_stdout(output):
                    self.assertEqual(preflight.main([]), code)
                self.assertEqual(json.loads(output.getvalue()), report)


if __name__ == "__main__":
    unittest.main()
