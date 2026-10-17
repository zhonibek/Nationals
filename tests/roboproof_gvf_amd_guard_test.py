import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_gvf_amd", ROOT / "tools/verify-gvf-amd.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def probe():
    return {"system": "Linux", "python": "3.12.0", "torch": "2.6.0+rocm6.3", "hip": "6.3.0", "cudaBuild": None,
            "cudaAvailable": True, "deviceCount": 1, "deviceIndex": 0, "device": "cuda:0", "name": "AMD Instinct MI300X",
            "totalMemoryBytes": 192 * 1024 ** 3, "inputDevice": "cuda:0", "outputDevice": "cuda:0", "lossDevice": "cuda:0",
            "parameterDevices": ["cuda:0"] * 4, "gradientDevices": ["cuda:0"] * 5, "inputShape": [2, 4], "outputShape": [2, 2],
            "executionCompleted": True, "synchronized": True, "forwardFinite": True, "backwardFinite": True,
            "optimizerSteps": 0, "loss": 0.5, "gradientL1": [0.25] * 5}


def cpu_summary():
    return {"testDevice": "cpu", "testsRun": 16, "successful": True, "gpuInitialized": False,
            "failures": 0, "errors": 0, "skipped": 0, "expectedFailures": 0, "unexpectedSuccesses": 0}


class AmdGuardTests(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for name in sorted(VERIFY.REQUIRED_SOURCES):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = ("frozen mock source: " + name).encode()
            target.write_bytes(content)
            rows.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        filename = root / "source-manifest.json"
        content = json.dumps({"files": rows}).encode()
        filename.write_bytes(content)
        return filename, hashlib.sha256(content).hexdigest()

    def arguments(self, root, filename, digest):
        return ["--approved-amd-mi300x", "--source-manifest", str(filename), "--source-manifest-sha256", digest,
                "--output", str(root / "roboproof/runs/motion/research/gvf-amd-neural-core-mock")]

    def approved(self):
        return patch.multiple(VERIFY, foreground_bound=Mock(return_value=nullcontext()))

    def fake_stage(self, stage, values, output, environment, terminal, deadline, timeout):
        terminal["commands"].append({"stage": stage, "state": "passed", "mockOnly": True})
        content = b""
        if stage == "node-version":
            content = b"v22.0.0\n"
        elif stage == "mi300x-actual-execution":
            VERIFY.write_json(output / "gpu-probe.json", probe())
        elif stage == "cpu-synthetic-model-td3":
            self.assertEqual(environment["HIP_VISIBLE_DEVICES"], "")
            self.assertEqual(environment["ROCR_VISIBLE_DEVICES"], "")
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
            VERIFY.write_json(output / "cpu-checks.json", cpu_summary())
        elif stage == "node-synthetic-parity":
            self.assertIn("--test-name-pattern=" + VERIFY.NODE_TEST_PATTERN, values)
            content = b"# tests 9\n# pass 7\n# fail 0\n# cancelled 0\n# skipped 2\n# todo 0\n"
        else:
            self.fail("Unexpected workload")
        (output / (stage + ".stdout.log")).write_bytes(content)
        (output / (stage + ".stderr.log")).write_bytes(b"mock stderr preserved\n")

    def test_import_and_embedded_program_syntax_need_no_torch_or_project_import(self):
        tree = ast.parse((ROOT / "tools/verify-gvf-amd.py").read_text())
        imports = [name.name for statement in tree.body if isinstance(statement, ast.Import) for name in statement.names]
        self.assertNotIn("torch", imports)
        isolated_import = (
            "import importlib.util,sys; "
            f"spec=importlib.util.spec_from_file_location('verify_gvf_amd',{str(ROOT / 'tools/verify-gvf-amd.py')!r}); "
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "assert 'torch' not in sys.modules; "
            "assert not any(name.startswith('roboproof.motion_learning') for name in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", isolated_import], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        ast.parse(VERIFY.GPU_PROBE)
        ast.parse(VERIFY.CPU_CHECKS)

    def test_missing_cli_approval_refused_before_sources_output_and_process(self):
        with patch.object(VERIFY, "execute") as execute, patch.object(VERIFY.subprocess, "Popen") as child:
            with self.assertRaises(SystemExit) as result, patch.object(sys, "stderr"):
                VERIFY.main(["--source-manifest", "unused", "--source-manifest-sha256", "0" * 64, "--output", "unused"])
            self.assertEqual(result.exception.code, 2)
            execute.assert_not_called()
            child.assert_not_called()

    def test_remote_guard_refuses_laptop_or_missing_environment_before_any_work(self):
        for system, sys_platform, approval in (("Windows", "win32", "1"), ("Darwin", "darwin", "1"),
                                               ("Linux", "linux", "0"), ("Linux", "linux", ""), ("Linux", "win32", "1")):
            with self.subTest(system=system, flag=approval), patch.object(VERIFY.platform, "system", return_value=system), \
                    patch.object(VERIFY.sys, "platform", sys_platform), patch.dict(os.environ, ROBOPROOF_APPROVED_REMOTE=approval), \
                    patch.object(VERIFY, "execute") as execute, patch.object(VERIFY, "verify_sources") as sources, \
                    patch.object(VERIFY.subprocess, "Popen") as child:
                with self.assertRaisesRegex(ValueError, "no laptop fallback"):
                    VERIFY.main(self.arguments(Path("unused"), Path("unused.json"), "0" * 64))
                execute.assert_not_called()
                sources.assert_not_called()
                child.assert_not_called()

    def test_exact_complete_manifest_and_observed_before_after_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            filename, digest = self.fixture(root)
            before, after = {}, {}
            with patch.object(VERIFY, "ROOT", root):
                frozen = VERIFY.verify_sources(filename, digest, before)
                self.assertEqual(VERIFY.verify_sources(filename, digest, after, frozen=frozen), frozen)
            self.assertTrue(before["verified"])
            self.assertTrue(after["verified"])
            self.assertEqual(before["files"], after["files"])
            self.assertEqual(before["observedManifestSha256"], digest)

    def test_missing_sources_duplicate_paths_bad_sizes_hashes_and_escapes_refused(self):
        mutations = [lambda rows: rows.pop(), lambda rows: rows.append(rows[0]),
                     lambda rows: rows[0].update(bytes=True), lambda rows: rows[0].update(bytes=-1),
                     lambda rows: rows[0].update(bytes=VERIFY.MAX_SOURCE_BYTES + 1),
                     lambda rows: rows[0].update(sha256="0" * 64), lambda rows: rows[0].update(extra=1)]
        for name in ("../escape", "/escape", "dir//source", "dir/./source", "dir/../source", "C:/escape", "dir\\source", ""):
            mutations.append(lambda rows, name=name: rows[0].update(path=name))
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                filename, _ = self.fixture(root)
                manifest = json.loads(filename.read_bytes())
                mutate(manifest["files"])
                content = json.dumps(manifest).encode()
                filename.write_bytes(content)
                with patch.object(VERIFY, "ROOT", root), self.assertRaises(ValueError):
                    VERIFY.verify_sources(filename, VERIFY.digest(content))

    def test_manifest_hash_and_duplicate_json_keys_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            filename, _ = self.fixture(root)
            with patch.object(VERIFY, "ROOT", root), self.assertRaisesRegex(ValueError, "SHA256"):
                VERIFY.verify_sources(filename, "0" * 64)
            content = b'{"files":[],"files":[]}'
            filename.write_bytes(content)
            with patch.object(VERIFY, "ROOT", root), self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                VERIFY.verify_sources(filename, VERIFY.digest(content))

    def test_source_and_manifest_mutations_have_failed_observed_after_snapshot(self):
        for change_manifest in (False, True):
            with self.subTest(manifest=change_manifest), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                filename, digest = self.fixture(root)
                after = {}
                with patch.object(VERIFY, "ROOT", root):
                    frozen = VERIFY.verify_sources(filename, digest)
                    if change_manifest:
                        filename.write_bytes(b"{}")
                    else:
                        (root / "simulator/gvf-policy.js").write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "byte mismatch"):
                        VERIFY.verify_sources(filename, digest, after, frozen=frozen)
                self.assertFalse(after["verified"])
                self.assertEqual(len(after["files"]), len(VERIFY.REQUIRED_SOURCES))
                if not change_manifest:
                    self.assertFalse(next(row for row in after["files"] if row["path"] == "simulator/gvf-policy.js")["matches"])

    def test_probe_validator_accepts_exact_device_zero_finite_execution_and_boundary_memory(self):
        evidence = probe()
        evidence["totalMemoryBytes"] = VERIFY.MIN_MEMORY_BYTES
        VERIFY.validate_probe(evidence)

    def test_probe_refuses_labels_without_actual_rocm_memory_or_execution(self):
        patches = ({"system": "Windows"}, {"torch": ""}, {"hip": None}, {"hip": ""}, {"cudaBuild": "12.6"},
                   {"cudaAvailable": False}, {"deviceCount": 0}, {"deviceCount": True}, {"device": "cpu"},
                   {"deviceIndex": 1}, {"deviceIndex": False}, {"name": "AMD Instinct MI250X"}, {"name": "NVIDIA MI300X"},
                   {"name": "AMD MI300XYZ"}, {"totalMemoryBytes": VERIFY.MIN_MEMORY_BYTES - 1}, {"totalMemoryBytes": True},
                   {"executionCompleted": False}, {"synchronized": False}, {"forwardFinite": False}, {"backwardFinite": False},
                   {"inputDevice": "cpu"}, {"outputDevice": "cuda:1"}, {"lossDevice": "cpu"},
                   {"gradientDevices": ["cpu"] * 5}, {"parameterDevices": ["cuda:0"] * 3},
                   {"inputShape": [4, 2]}, {"outputShape": [2, 3]}, {"optimizerSteps": 1}, {"optimizerSteps": False},
                   {"loss": float("nan")}, {"loss": float("inf")}, {"loss": True}, {"loss": None},
                   {"gradientL1": [0] * 5}, {"gradientL1": [float("inf")] * 5}, {"gradientL1": [1] * 4})
        for change in patches:
            with self.subTest(change=change), self.assertRaises(ValueError):
                VERIFY.validate_probe({**probe(), **change})

    def test_cpu_requires_sixteen_passes_without_any_skip_or_gpu_use(self):
        VERIFY.validate_cpu(cpu_summary())
        for change in ({"testsRun": 15}, {"skipped": 1}, {"successful": False}, {"gpuInitialized": True},
                       {"testDevice": "cuda:0"}, {"failures": 1}, {"errors": 1}, {"failures": False}, {"expectedFailures": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                VERIFY.validate_cpu({**cpu_summary(), **change})

    def test_node_counts_allow_only_seven_selected_non_native_checks(self):
        for skipped in (0, 2):
            VERIFY.validate_node(f"# tests {7 + skipped}\n# pass 7\n# fail 0\n# cancelled 0\n# skipped {skipped}\n# todo 0\n".encode())
        for counts in (b"", b"# tests 9\n# pass 9\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n",
                       b"# tests 7\n# pass 6\n# fail 1\n# cancelled 0\n# skipped 0\n# todo 0\n"):
            with self.assertRaises(ValueError):
                VERIFY.validate_node(counts)

    def test_output_rejects_overwrites_archive_collision_and_wrong_parent(self):
        for collision in ("directory", "archive", "parent", "name"):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                output = root / "roboproof/runs/motion/research/gvf-amd-neural-core-mock"
                if collision == "directory":
                    output.mkdir(parents=True)
                elif collision == "archive":
                    output.parent.mkdir(parents=True)
                    output.with_name(output.name + ".zip").write_bytes(b"do not replace")
                elif collision == "parent":
                    output = root / output.name
                else:
                    output = output.with_name("other")
                with patch.object(VERIFY, "ROOT", root), self.assertRaises(ValueError):
                    VERIFY.prepare_output(output)
                if collision == "archive":
                    self.assertEqual(output.with_name(output.name + ".zip").read_bytes(), b"do not replace")

    def test_output_and_source_symlink_ancestors_rejected_without_privileged_link_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            filename, digest = self.fixture(root)
            output = root / "roboproof/runs/motion/research/gvf-amd-neural-core-mock"
            for rejected in (output, output.parent, root, output.with_name(output.name + ".zip")):
                with self.subTest(rejected=rejected), patch.object(VERIFY, "ROOT", root), \
                        patch.object(Path, "is_symlink", autospec=True, side_effect=lambda path: path == rejected):
                    with self.assertRaisesRegex(ValueError, "Symlink"):
                        VERIFY.prepare_output(output)
            with patch.object(VERIFY, "ROOT", root), patch.object(Path, "is_symlink", autospec=True,
                    side_effect=lambda path: path == root / "simulator"), self.assertRaisesRegex(ValueError, "byte mismatch"):
                VERIFY.verify_sources(filename, digest)

    def test_successful_mock_run_preserves_logs_source_and_verified_zip_not_billing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            filename, digest = self.fixture(root)
            arguments = self.arguments(root, filename, digest)
            with patch.object(VERIFY, "ROOT", root), patch.object(VERIFY.platform, "system", return_value="Linux"), \
                    patch.object(VERIFY.sys, "platform", "linux"), patch.dict(os.environ, ROBOPROOF_APPROVED_REMOTE="1"), \
                    self.approved(), patch.object(VERIFY, "run_stage", side_effect=self.fake_stage) as stages, \
                    patch.object(VERIFY.subprocess, "Popen") as child, patch("builtins.print") as printed:
                self.assertEqual(VERIFY.main(arguments), 0)
                child.assert_not_called()
            output = Path(arguments[-1])
            receipt = json.loads(printed.call_args.args[0])
            self.assertEqual(stages.call_count, 4)
            self.assertTrue(receipt["archiveVerified"])
            self.assertFalse(receipt["externalCustodyVerified"])
            self.assertTrue(receipt["terminal"]["completed"])
            for name in ("providerBillingVerifiedByScript", "movementTrainingPerformed", "trainingFitPerformed",
                         "teacherCollectionPerformed", "fullRegressionPerformed", "learnedImprovementVerified"):
                self.assertFalse(receipt["terminal"][name])
            manifest = json.loads((output / "preservation-manifest.json").read_bytes())
            with zipfile.ZipFile(receipt["archive"]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(json.loads(archive.read("preservation-manifest.json")), manifest)
                for row in manifest["files"]:
                    content = archive.read(row["path"])
                    self.assertEqual(len(content), row["bytes"])
                    self.assertEqual(VERIFY.digest(content), row["sha256"])
                    self.assertEqual(content, (output / row["path"]).read_bytes())
            self.assertEqual(VERIFY.digest(Path(receipt["archive"]).read_bytes()), receipt["sha256"])

    def test_failed_probe_timeout_or_source_mutation_stops_and_preserves_failure(self):
        for mode in ("wrong_gpu", "timeout", "source_change", "manifest_hash"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                filename, digest = self.fixture(root)
                arguments = self.arguments(root, filename, "0" * 64 if mode == "manifest_hash" else digest)

                def stage(*values):
                    self.fake_stage(*values)
                    if values[0] == "mi300x-actual-execution":
                        if mode == "wrong_gpu":
                            evidence = probe()
                            evidence["name"] = "AMD MI250X"
                            (values[2] / "gpu-probe.json").write_text(json.dumps(evidence))
                        elif mode == "timeout":
                            raise subprocess.TimeoutExpired("mock probe", 1)
                    if values[0] == "node-synthetic-parity" and mode == "source_change":
                        (root / "simulator/gvf-policy.js").write_bytes(b"changed")

                with patch.object(VERIFY, "ROOT", root), patch.object(VERIFY.platform, "system", return_value="Linux"), \
                        patch.object(VERIFY.sys, "platform", "linux"), patch.dict(os.environ, ROBOPROOF_APPROVED_REMOTE="1"), \
                        self.approved(), patch.object(VERIFY, "run_stage", side_effect=stage) as stages, patch("builtins.print") as printed:
                    self.assertEqual(VERIFY.main(arguments), 1)
                terminal = json.loads(printed.call_args.args[0])["terminal"]
                self.assertFalse(terminal["completed"])
                self.assertEqual(stages.call_count, 0 if mode == "manifest_hash" else 4 if mode == "source_change" else 2)
                self.assertTrue((Path(arguments[-1]) / "source-after.json").is_file())
                self.assertTrue(Path(json.loads(printed.call_args.args[0])["archive"]).is_file())

    def test_mock_child_timeout_kills_group_reaps_and_never_claims_model_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            process = Mock(pid=12345, returncode=-9)
            process.wait.side_effect = [subprocess.TimeoutExpired("mock", 1), None]
            process.poll.return_value = -9
            terminal = {"commands": []}
            with patch.object(VERIFY.subprocess, "Popen", return_value=process) as popen, \
                    patch.object(VERIFY.time, "monotonic", side_effect=[0, 0, 0.1, 2]), \
                    patch.object(VERIFY.os, "killpg", create=True) as killpg, patch.object(VERIFY, "group_missing", return_value=True):
                with self.assertRaises(subprocess.TimeoutExpired):
                    VERIFY.run_stage("mock-timeout", ["not-executed"], output, {}, terminal, 1, 1)
            record = terminal["commands"][0]
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            killpg.assert_called_with(12345, VERIFY.signal.SIGTERM)
            self.assertTrue(record["directChildReaped"])
            self.assertTrue(record["processGroupMissing"])
            self.assertEqual(record["state"], "timed_out")
            self.assertEqual(record["modelOutcome"], "unresolved after observation timeout")
            self.assertTrue((output / "mock-timeout.stdout.log").is_file())
            self.assertTrue((output / "mock-timeout.stderr.log").is_file())

    def test_mock_child_escalation_does_not_claim_missing_process_without_evidence(self):
        process = Mock(pid=12345, returncode=None)
        process.wait.side_effect = subprocess.TimeoutExpired("mock", 2)
        process.poll.return_value = None
        record = {}
        with patch.object(VERIFY.os, "killpg", create=True) as killpg, patch.object(VERIFY, "group_missing", return_value=False):
            VERIFY.terminate_child(process, record)
        self.assertEqual(killpg.call_args_list[-1].args, (12345, 9))
        self.assertFalse(record["directChildReaped"])
        self.assertFalse(record["processGroupMissing"])

    def test_workload_deadline_prevents_spawn_and_limits_fit_free_stage(self):
        with patch.object(VERIFY.time, "monotonic", return_value=601), patch.object(VERIFY.subprocess, "Popen") as child:
            with self.assertRaisesRegex(ValueError, "workload budget"):
                VERIFY.run_stage("unused", ["not-executed"], Path("unused"), {}, {"commands": []}, 600, 240)
            child.assert_not_called()
        self.assertLessEqual(VERIFY.WORK_SECONDS + 4 * 4, VERIFY.TOTAL_SECONDS)
        self.assertLessEqual(VERIFY.TOTAL_SECONDS, 12 * 60)

    def test_archive_rejects_symlink_or_oversized_results(self):
        for mode in ("symlink", "size"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                with patch.object(VERIFY, "ROOT", root):
                    output, archive, stream = VERIFY.prepare_output(root / "roboproof/runs/motion/research/gvf-amd-neural-core-mock")
                target = output / "result.log"
                target.write_bytes(b"x")
                try:
                    if mode == "symlink":
                        with patch.object(Path, "is_symlink", autospec=True, side_effect=lambda path: path == target), \
                                self.assertRaisesRegex(ValueError, "Symlink"):
                            VERIFY.preserve(output, archive, stream)
                    else:
                        with patch.object(VERIFY, "MAX_LOG_BYTES", 0), self.assertRaisesRegex(ValueError, "Bounded"):
                            VERIFY.preserve(output, archive, stream)
                finally:
                    stream.close()


if __name__ == "__main__":
    unittest.main()
