import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def verify_sources(filename, expected):
    require(filename.is_file() and not filename.is_symlink() and filename.stat().st_size <= 4 * 1024 * 1024,
            "Bounded regular frozen source manifest required")
    content = filename.read_bytes()
    require(digest(content) == expected, "Exact frozen source-manifest SHA256 required")
    manifest = json.loads(content)
    rows = manifest["files"]
    require(type(rows) is list and 1 <= len(rows) <= 512 and len({row["path"] for row in rows}) == len(rows),
            "Complete unique bounded source inventory required")
    for row in rows:
        name = PurePosixPath(row["path"])
        require(not name.is_absolute() and name.as_posix() == row["path"] and "\\" not in row["path"] and
                bool(name.parts) and all(part not in (".", "..") for part in name.parts), "Canonical contained source path required")
        target = ROOT.joinpath(*name.parts)
        for ancestor in [target, *target.parents]:
            if ancestor == ROOT.parent:
                break
            require(not ancestor.is_symlink(), "Source symlink/ancestor rejected")
        require(target.is_file() and target.resolve().is_relative_to(ROOT) and target.stat().st_size == row["bytes"] and
                row["bytes"] <= 8 * 1024 * 1024 and digest(target.read_bytes()) == row["sha256"],
                "Frozen source byte mismatch: " + row["path"])
    return content


def main(argv=None):
    parser = argparse.ArgumentParser(description="Remote-only bounded GVF neural-core checks; not movement training or qualification")
    parser.add_argument("--approved-free-colab", action="store_true", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    require(options.approved_free_colab and platform.system() == "Linux" and Path("/content").is_dir() and
            os.environ.get("ROBOPROOF_APPROVED_REMOTE") == "1", "Approved free remote Colab only; no laptop fallback")
    source_manifest = verify_sources(options.source_manifest, options.source_manifest_sha256)
    parent = ROOT / "roboproof/runs/motion/research"
    for ancestor in [parent, *parent.parents]:
        if ancestor == ROOT.parent:
            break
        require(not ancestor.is_symlink(), "Verification output ancestor must not be a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    output = options.output.absolute()
    require(output.parent == parent and output.name.startswith("gvf-neural-core-") and not output.exists() and
            not output.is_symlink() and not output.with_suffix(".zip").exists(), "New uniquely owned verification output required")
    output.mkdir()
    terminal = {"schemaVersion": 1, "completed": False, "commands": [], "syntheticOptimizerChecksAttempted": False,
                "movementTrainingPerformed": False, "newPhysicsQueries": 0, "independentFinalPrepared": False,
                "learnedImprovementVerified": False, "automaticTrainingOrPromotionEnabled": False}
    environment = dict(os.environ, PYTHONPATH=str(ROOT), ROBOPROOF_PYTHON=sys.executable, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def command(stage, values, timeout=240):
        terminal["commands"].append({"stage": stage, "startedAtUtc": datetime.now(timezone.utc).isoformat()})
        with (output / (stage + ".stdout.log")).open("xb") as stdout, (output / (stage + ".stderr.log")).open("xb") as stderr:
            try:
                result = subprocess.run(values, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                        stdout=stdout, stderr=stderr, timeout=timeout)
            except subprocess.TimeoutExpired:
                terminal["commands"][-1].update(timedOut=True, directChildTerminatedAndWaited=True)
                raise
        terminal["commands"][-1].update(exitCode=result.returncode, terminalAtUtc=datetime.now(timezone.utc).isoformat())
        require(result.returncode == 0, "Preserve failed verification stage: " + stage)

    try:
        host = {"checkedAtUtc": datetime.now(timezone.utc).isoformat(), "system": platform.system(), "python": platform.python_version(),
                "node": subprocess.check_output(["node", "--version"], text=True).strip(), "cpus": os.cpu_count(),
                "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip(),
                "testDevice": "cpu", "gpuUsedByChecks": False, "freeResourceOperatorApproval": True, "providerBillingVerifiedByScript": False}
        require(host["gpu"] == "Tesla T4, 15360 MiB", "Actual approved free T4 allocation must be checked before tests")
        (output / "host.json").write_text(json.dumps(host, indent=2))
        (output / "frozen-source-manifest.json").write_bytes(source_manifest)
        command("actual-torch-runtime", [sys.executable, "-c", "import json,torch; print(json.dumps({'torch':str(torch.__version__),'cudaBuild':torch.version.cuda,'cudaAvailable':torch.cuda.is_available(),'testDevice':'cpu'}))"])
        command("node-synthetic-model-and-stdlib-contract", ["node", "--test", "tests/gvf-model-contract.test.js"], 120)
        terminal["syntheticOptimizerChecksAttempted"] = True
        command("python-neural-core-contract-and-gradient-checks", [sys.executable, "-m", "unittest", "-v",
            "tests.roboproof_gvf_model_contract_test", "tests.roboproof_gvf_model_remote_test", "tests.roboproof_gvf_td3_remote_test"])
        verify_sources(options.source_manifest, options.source_manifest_sha256)
        terminal["completed"] = True
    except BaseException as error:
        terminal["error"] = str(error)
        (output / "exception.log").write_text(traceback.format_exc())
    finally:
        terminal["terminalAtUtc"] = datetime.now(timezone.utc).isoformat()
        (output / "terminal.json").write_text(json.dumps(terminal, indent=2))
        files = sorted(filename for filename in output.rglob("*") if filename.is_file())
        preservation = {"schemaVersion": 1, "scope": "Actual remote synthetic neural-core checks, not movement training or performance",
                        "files": [{"path": filename.relative_to(output).as_posix(), "bytes": filename.stat().st_size,
                                   "sha256": digest(filename.read_bytes())} for filename in files]}
        destination = output.with_suffix(".zip")
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for filename in files:
                archive.write(filename, filename.relative_to(output).as_posix())
            archive.writestr("preservation-manifest.json", json.dumps(preservation, indent=2))
        with zipfile.ZipFile(destination) as archive:
            require(archive.testzip() is None and len(archive.namelist()) == len(files) + 1, "Every actual result CRC/inventory required")
            for row in preservation["files"]:
                content = archive.read(row["path"])
                require(len(content) == row["bytes"] and digest(content) == row["sha256"], "Every actual result byte must verify")
        print(json.dumps({"archive": str(destination), "bytes": destination.stat().st_size, "sha256": digest(destination.read_bytes()),
                          "terminal": terminal, "externalCustodyVerified": False}), flush=True)
    return 0 if terminal["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
