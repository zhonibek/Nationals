import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "tools/nebius-tactical-job.sh"
RUN_NAME = "tactical-nebius-l40s-seed731201"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
# Windows has no reliable bash on PATH (it may be WSL); point ROBOPROOF_TEST_BASH at Git Bash to run these there.
BASH = os.environ.get("ROBOPROOF_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)


@unittest.skipUnless(BASH, "bash is required for the Nebius job script tests")
class NebiusTacticalJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        shims = self.root / "shims"
        shims.mkdir()
        # The job calls `python` and `nvidia-smi`; the failing nvidia-smi stands in for a missing GPU.
        (shims / "python").write_text(f'#!/bin/sh\nexec "{Path(sys.executable).as_posix()}" "$@"\n', newline="\n")
        (shims / "nvidia-smi").write_text("#!/bin/sh\necho 'no GPU in this test' >&2\nexit 9\n", newline="\n")
        for shim in shims.iterdir():
            shim.chmod(0o755)
        self.shims = shims

    def run_job(self, **environment):
        variables = {"ROBOPROOF_EXPECTED_COMMIT": COMMIT, "ROBOPROOF_APPROVED_MAX_COST_USD": "2.00",
                     "ROBOPROOF_OUTPUT_ROOT": self.output.as_posix(), "ROBOPROOF_WORK_ROOT": (self.root / "work").as_posix()}
        variables.update(environment)
        # `cd && pwd` yields a PATH-safe directory on Git Bash too (C:/... would split at the colon).
        script = (f'export PATH="$(cd "{self.shims.as_posix()}" && pwd):$PATH"; '
                  + " ".join(f"export {key}='{value}';" for key, value in variables.items())
                  + f' bash "{JOB.as_posix()}"')
        return subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=120)

    def test_preflight_fails_closed_before_claiming_the_seed(self):
        for label, environment in {"no commit": {"ROBOPROOF_EXPECTED_COMMIT": "main"},
                                   "no cost": {"ROBOPROOF_APPROVED_MAX_COST_USD": ""},
                                   "no bucket": {"ROBOPROOF_OUTPUT_ROOT": (self.root / "missing").as_posix()}}.items():
            with self.subTest(label):
                result = self.run_job(**environment)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(list(self.output.iterdir()), [])
                self.assertFalse((self.root / "work").exists())

    def test_existing_run_folder_is_never_reused(self):
        previous = self.output / RUN_NAME
        previous.mkdir()
        (previous / "claimed.json").write_text("{}\n")
        result = self.run_job()
        self.assertEqual(result.returncode, 2)
        self.assertIn("already claimed", result.stderr)
        self.assertEqual([path.name for path in previous.iterdir()], ["claimed.json"])

    def test_failed_job_is_still_packaged_hashed_and_copied(self):
        result = self.run_job()
        self.assertNotEqual(result.returncode, 0)
        destination = self.output / RUN_NAME
        self.assertEqual(sorted(path.name for path in destination.iterdir()),
                         ["SHA256SUMS", "claimed.json", "job.json", RUN_NAME + ".tar.gz", RUN_NAME + ".tar.gz.sha256"])
        archive = destination / (RUN_NAME + ".tar.gz")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.assertEqual((destination / (RUN_NAME + ".tar.gz.sha256")).read_text(), f"{digest}  {RUN_NAME}.tar.gz\n")
        self.assertIn(digest, result.stdout)
        job = json.loads((destination / "job.json").read_text())
        self.assertEqual((job["seed"], job["commit"], job["jobExitStatus"], job["trainerExitStatus"]),
                         (731201, COMMIT, result.returncode, None))
        self.assertIs(job["learnedImprovementVerified"], False)
        self.assertIs(job["providerBillingVerified"], False)
        with tarfile.open(archive) as bundle:
            members = {member.name: bundle.extractfile(member).read() for member in bundle.getmembers() if member.isfile()}
        # sha256sum separates with two spaces, or " *" where it reads in binary mode (Git Bash).
        listed = {name: digest for digest, name in
                  re.findall(r"^([0-9a-f]{64}) [ *](.+)$", members[f"{RUN_NAME}/SHA256SUMS"].decode(), re.M)}
        self.assertEqual(set(listed), set(members) - {f"{RUN_NAME}/SHA256SUMS"})
        for name, expected in listed.items():
            self.assertEqual(hashlib.sha256(members[name]).hexdigest(), expected, name)
        self.assertIn(f"{RUN_NAME}/nvidia-smi.txt", listed)


if __name__ == "__main__":
    unittest.main()
