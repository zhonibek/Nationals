import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_gvf_learning", ROOT / "tools/verify-gvf-learning.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class VerificationGuardTests(unittest.TestCase):
    def fixture(self, root, name="tiny.txt", sha256=None):
        target = root / "tiny.txt"
        target.write_bytes(b"tiny source fixture")
        manifest = root / "manifest.json"
        content = json.dumps({"files": [{"path": name, "bytes": target.stat().st_size,
                              "sha256": sha256 or hashlib.sha256(target.read_bytes()).hexdigest()}]}).encode()
        manifest.write_bytes(content)
        return manifest, hashlib.sha256(content).hexdigest()

    def test_laptop_refusal_happens_before_subprocess_model_or_source_checks(self):
        with patch.object(VERIFY.platform, "system", return_value="Windows"), patch.object(VERIFY, "verify_sources") as sources, \
                patch.object(VERIFY.subprocess, "check_output") as subprocess_call:
            with self.assertRaisesRegex(ValueError, "no laptop fallback"):
                VERIFY.main(["--approved-free-colab", "--source-manifest", "unused.json", "--source-manifest-sha256", "0" * 64,
                             "--output", "unused-output"])
            sources.assert_not_called()
            subprocess_call.assert_not_called()

    def test_tiny_exact_source_manifest_and_hash_verify_without_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, digest = self.fixture(root)
            with patch.object(VERIFY, "ROOT", root):
                self.assertEqual(VERIFY.verify_sources(manifest, digest), manifest.read_bytes())

    def test_wrong_source_or_manifest_hash_and_escape_paths_refused(self):
        for name, wrong_hash in (("tiny.txt", True), ("../tiny.txt", False), ("/tiny.txt", False), ("dir//tiny.txt", False)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                manifest, digest = self.fixture(root, name=name, sha256="0" * 64 if wrong_hash else None)
                with patch.object(VERIFY, "ROOT", root), self.assertRaises(ValueError):
                    VERIFY.verify_sources(manifest, digest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, _ = self.fixture(root)
            with patch.object(VERIFY, "ROOT", root), self.assertRaisesRegex(ValueError, "manifest SHA256"):
                VERIFY.verify_sources(manifest, "0" * 64)

    def test_duplicate_source_entries_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, _ = self.fixture(root)
            value = json.loads(manifest.read_text())
            value["files"] *= 2
            content = json.dumps(value).encode()
            manifest.write_bytes(content)
            with patch.object(VERIFY, "ROOT", root), self.assertRaisesRegex(ValueError, "unique bounded"):
                VERIFY.verify_sources(manifest, hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()
