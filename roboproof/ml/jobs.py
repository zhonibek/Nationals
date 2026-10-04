"""Portable allowlisted job bundle and fake lifecycle; no remote transport exists yet."""

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

from .data import ROOT, file_hash, load_package, read_json, write_json

SOURCES = [
    "roboproof/scenario-spec.json", "roboproof/gpu/__init__.py", "roboproof/gpu/controller.py",
    "roboproof/ml/__init__.py", "roboproof/ml/contract.json", "roboproof/ml/data.py",
    "roboproof/ml/model.py", "roboproof/ml/train.py"
]


def validate_config(config):
    if config.get("schema_version") != 1 or config.get("provider") != "unconfigured" or config.get("remote_enabled") is not False:
        raise ValueError("Remote provider is not implemented; remote execution remains disabled")
    limits = config["limits"]
    for name, upper in [("max_jobs", 8), ("max_concurrent", 2), ("max_wall_seconds", 3600), ("max_artifact_bytes", 268435456)]:
        value = limits[name]
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= upper:
            raise ValueError("Invalid job limit: " + name)
    if limits["max_concurrent"] > limits["max_jobs"] or limits["max_spend_usd"] != 0 or limits["automatic_retries"] != 0:
        raise ValueError("Current preparation authorizes zero spend and no retries")
    return limits


def build_bundle(package_path, config_path, output, allow_legacy=False):
    config = read_json(config_path)
    limits = validate_config(config)
    package = load_package(package_path, allow_legacy)
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Bundle output must be empty")
    files = {name: (ROOT / name).read_bytes() for name in SOURCES}
    files["roboproof/runs/ml-job/package.json"] = Path(package_path).read_bytes()
    if sum(len(content) for content in files.values()) > limits["max_artifact_bytes"]:
        raise ValueError("Bundle exceeds artifact quota")
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / "job.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, content in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            entry.mode = 0o644
            entry.mtime = 0
            archive.addfile(entry, io.BytesIO(content))
    import hashlib
    manifest = dict(schema_version=1, provider="unconfigured", remote_enabled=False,
                    package_sha256=package["package_sha256"], canonical_ready=False,
                    archive_sha256=file_hash(archive_path), limits=limits,
                    files={name: dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest()) for name, content in files.items()},
                    command=["python", "-m", "roboproof.ml.train", "train", "--package", "roboproof/runs/ml-job/package.json",
                             "--out", "roboproof/runs/ml-job/training", "--device", "rocm", "--max-seconds", str(limits["max_wall_seconds"]), "--allow-legacy"],
                    runtime="Install a compatible HIP PyTorch build only after supplied AMD host capabilities are verified",
                    lifecycle=["submit", "status", "logs", "cancel", "download-and-verify"],
                    status="PREPARED_ONLY; no job submitted or charged")
    write_json(directory / "job.json", manifest)
    return manifest


def verify_bundle(archive_path, manifest):
    if file_hash(archive_path) != manifest["archive_sha256"]:
        raise ValueError("Archive hash mismatch")
    limits = validate_config(dict(schema_version=1, provider=manifest["provider"], remote_enabled=manifest["remote_enabled"], limits=manifest["limits"]))
    expected = set(SOURCES) | {"roboproof/runs/ml-job/package.json"}
    if set(manifest["files"]) != expected:
        raise ValueError("Bundle file allowlist mismatch")
    import hashlib
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != expected or any(not member.isfile() for member in members):
            raise ValueError("Unexpected, linked or duplicate archive member")
        if sum(member.size for member in members) > limits["max_artifact_bytes"]:
            raise ValueError("Archive exceeds artifact quota")
        for member in members:
            content = archive.extractfile(member).read()
            recorded = manifest["files"][member.name]
            if len(content) != recorded["bytes"] or hashlib.sha256(content).hexdigest() != recorded["sha256"]:
                raise ValueError("Bundle member hash mismatch")
    return True


class FakeProvider:
    def __init__(self, limits):
        validate_config(dict(schema_version=1, provider="unconfigured", remote_enabled=False, limits=limits))
        self.limits = limits
        self.jobs = {}

    def submit(self, manifest, archive_path):
        if len(self.jobs) >= self.limits["max_jobs"]:
            raise ValueError("Job count budget exhausted")
        if sum(job["state"] == "RUNNING" for job in self.jobs.values()) >= self.limits["max_concurrent"]:
            raise ValueError("Concurrency budget exhausted")
        if manifest["limits"] != self.limits:
            raise ValueError("Provider/job limits mismatch")
        verify_bundle(archive_path, manifest)
        job_id = f"fake-{len(self.jobs) + 1}"
        self.jobs[job_id] = dict(state="RUNNING", elapsed=0, artifacts={}, logs=["FAKE provider: no GPU, process, network or billing used"])
        return job_id

    def status(self, job_id):
        job = self.jobs[job_id]
        return dict(job_id=job_id, state=job["state"], elapsed=job["elapsed"], fake=True, spend_usd=0)

    def logs(self, job_id):
        return list(self.jobs[job_id]["logs"])

    def advance(self, job_id, seconds):
        if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 0:
            raise ValueError("Invalid simulated elapsed time")
        job = self.jobs[job_id]
        if job["state"] != "RUNNING":
            raise ValueError("Job is not running")
        job["elapsed"] += seconds
        if job["elapsed"] >= self.limits["max_wall_seconds"]:
            job["state"] = "TIMED_OUT"
            job["logs"].append("Fake wall-time budget enforced")

    def cancel(self, job_id):
        job = self.jobs[job_id]
        if job["state"] == "RUNNING":
            job["state"] = "CANCELLED"
            job["logs"].append("Fake cancellation acknowledged")
        return self.status(job_id)

    def complete(self, job_id, artifacts):
        job = self.jobs[job_id]
        if job["state"] != "RUNNING":
            raise ValueError("Job is not running")
        if set(artifacts) - {"model.pt", "checkpoint.pt", "model-card.json"} or any(not isinstance(content, bytes) for content in artifacts.values()):
            raise ValueError("Invalid artifact payload")
        if sum(len(content) for content in artifacts.values()) > self.limits["max_artifact_bytes"]:
            raise ValueError("Artifact quota exceeded")
        job["artifacts"] = dict(artifacts)
        job["state"] = "SUCCEEDED"

    def download(self, job_id, name, expected_sha256):
        import hashlib
        job = self.jobs[job_id]
        if job["state"] != "SUCCEEDED":
            raise ValueError("Artifacts unavailable before success")
        content = job["artifacts"][name]
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError("Downloaded artifact hash mismatch")
        return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--config", default="roboproof/ml/amd-job.example.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-legacy", action="store_true")
    options = parser.parse_args()
    manifest = build_bundle(options.package, options.config, options.out, options.allow_legacy)
    verify_bundle(Path(options.out) / "job.tar.gz", manifest)
    print(json.dumps(dict(status=manifest["status"], archive_sha256=manifest["archive_sha256"])))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, tarfile.TarError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
