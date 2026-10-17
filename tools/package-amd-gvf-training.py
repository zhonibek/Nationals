"""Package only the hash-bound original-simulator GVF training source for AMD."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_BUNDLE_BYTES = 160 * 1024 * 1024
MAX_FILES = 2048
SIMULATOR_RUNTIME = (
    "headless.js", "engine.js", "motion.js", "policy.js", "gvf-policy.js",
    "control-runtime.js", "robot-config.js", "override.js", "override-geometry.js",
    "override-dynamics.js", "override-autonomy.js",
)
EVALUATION_FIXTURES = (
    "tests/fixtures/motion-evaluation-worlds.json",
    "tests/fixtures/motion-evaluation-worlds-v2.json",
)
EVALUATION_RUNTIME = (
    "roboproof/core.js", "roboproof/scenario-spec.json", "roboproof/gvf-motion-experiment.js",
    "roboproof/motion.js", "roboproof/motion-evaluation.js", "roboproof/motion-development.js",
    "roboproof/motion-gvf-learner.js", "roboproof/motion-learner.js",
    "roboproof/motion-exposures.js", "roboproof/motion-exposure-codec.js", "roboproof/motion-holdout.js",
    "roboproof/motion-exposure-recovery.js",
)
EXPOSURE_DIRECTORIES = ("learning", "research", "development", "td3", "gvf-td3")
EXPOSURE_FILENAMES = frozenset((
    "exposure-loss.json", "exposures.jsonl", "run.json", "summary.json",
    "trial-protocol.json", "worlds.json", "exposure.json",
))
RECOVERY_BASE = "roboproof/runs/motion/recovery"
RECOVERY_ENVELOPE_SHA256 = "9e600ecea59aa2b9e42e0cf3ad9d0e65f038bb35ff52d9e219b016fc9e33b3a6"
RECOVERY_EVIDENCE = (
    "colab-expiry-20261016/retained-history-inspection.json",
    "colab-expiry-20261016/source.zip",
    "colab-expiry-20261016/auditor.py",
    "continuation-audit-20261017/exposure-audit.json",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _relative_path(value):
    name = PurePosixPath(value)
    if (not value or name.is_absolute() or name.as_posix() != value or "\\" in value or
            any(part in ("", ".", "..") for part in name.parts)):
        raise ValueError("Canonical contained bundle paths required")
    return name


def _read_regular(root, relative):
    name = _relative_path(relative)
    target = root.joinpath(*name.parts)
    for parent in (target, *target.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError("Symlinked source rejected: " + relative)
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
        raise ValueError("Bounded regular source file required: " + relative)
    data = target.read_bytes()
    if len(data) != info.st_size:
        raise ValueError("Source changed while being packaged: " + relative)
    return data


def collect_sources(root=ROOT):
    root = Path(root).resolve(strict=True)
    sources = {}
    total_bytes = 0

    def add(relative):
        nonlocal total_bytes
        if relative in sources:
            return
        data = _read_regular(root, relative)
        sources[relative] = data
        total_bytes += len(data)
        if len(sources) > MAX_FILES or total_bytes > MAX_BUNDLE_BYTES:
            raise ValueError("AMD training bundle exceeds the bounded source inventory")

    add("config/robot.json")
    add("roboproof/motion_env.py")
    add("roboproof/motion-bridge.js")
    add("roboproof/motion_learning/__init__.py")
    for filename in sorted((root / "roboproof/motion_learning").glob("*.py")):
        if filename.is_symlink():
            raise ValueError("Symlinked Python module rejected")
        add(filename.relative_to(root).as_posix())
    for relative in EVALUATION_FIXTURES:
        add(relative)
    for relative in EVALUATION_RUNTIME:
        add(relative)

    build_manifest = json.loads(add_and_return(sources, root, "simulator/control-build.json"))
    if (not isinstance(build_manifest, dict) or set(build_manifest) != {"wasm", "sources", "compiler"} or
            not isinstance(build_manifest["sources"], dict) or len(build_manifest["sources"]) > MAX_FILES):
        raise ValueError("Exact original simulator build identity required")
    add("simulator/control.wasm")
    if sha256(sources["simulator/control.wasm"]) != build_manifest["wasm"]:
        raise ValueError("Original simulator WASM does not match its build identity")
    for relative, expected in build_manifest["sources"].items():
        data = _read_regular(root, relative)
        normalized = data.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")
        if sha256(normalized) != expected:
            raise ValueError("Stale original simulator controller source: " + relative)
        add(relative)
    add("simulator/control-build.json")
    for filename in SIMULATOR_RUNTIME:
        add("simulator/" + filename)

    history_root = root / "roboproof/runs/motion"
    history_files = 0
    for directory_name in EXPOSURE_DIRECTORIES:
        directory = history_root / directory_name
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Regular retained exposure directory required: " + directory_name)
        for current, directories, filenames in os.walk(directory, followlinks=False,
                                                        onerror=lambda error: (_ for _ in ()).throw(error)):
            current_path = Path(current)
            depth = len(current_path.relative_to(directory).parts)
            if depth > 7:
                raise ValueError("Retained exposure directory depth exceeded")
            for name in directories:
                if (current_path / name).is_symlink():
                    raise ValueError("Symlinked retained exposure directory rejected")
            for name in filenames:
                target = current_path / name
                if target.is_symlink():
                    raise ValueError("Symlinked retained exposure file rejected")
                if name not in EXPOSURE_FILENAMES:
                    continue
                history_files += 1
                if history_files > 2048:
                    raise ValueError("Retained exposure file count exceeds its bounded inventory")
                add(target.relative_to(root).as_posix())
    registry = root / "roboproof/motion-exposure-losses.json"
    if registry.exists():
        add(registry.relative_to(root).as_posix())
        recovery_root = root / RECOVERY_BASE / "colab-expiry-20261016"
        manifest_path = recovery_root / "envelope/envelope-manifest.json"
        manifest_bytes = _read_regular(root, manifest_path.relative_to(root).as_posix())
        if sha256(manifest_bytes) != RECOVERY_ENVELOPE_SHA256:
            raise ValueError("Pinned reviewed retained-exposure recovery manifest required")
        manifest = json.loads(manifest_bytes)
        if (not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1 or
                manifest.get("specification") != "td3-full-corpus-prefix-envelope-v1" or
                not isinstance(manifest.get("shards"), list) or len(manifest["shards"]) != 45 or
                not isinstance(manifest.get("offlineCorpus"), dict)):
            raise ValueError("Exact reviewed retained-exposure recovery envelope required")
        for relative in RECOVERY_EVIDENCE:
            add(f"{RECOVERY_BASE}/{relative}")
        add(manifest_path.relative_to(root).as_posix())
        for row in manifest["shards"]:
            if not isinstance(row, dict) or set(row) != {"path", "trainingSeed", "firstOrdinal", "count", "sha256", "bytes"}:
                raise ValueError("Malformed pinned recovery shard descriptor")
            add(f"{RECOVERY_BASE}/colab-expiry-20261016/envelope/{_relative_path(row['path']).as_posix()}")
        offline = manifest["offlineCorpus"]
        if set(offline) != {"path", "count", "sha256", "bytes"}:
            raise ValueError("Malformed pinned offline exposure descriptor")
        add(f"{RECOVERY_BASE}/colab-expiry-20261016/envelope/{_relative_path(offline['path']).as_posix()}")

    holdout = root / "roboproof/runs/motion/holdout"
    if holdout.exists():
        if holdout.is_symlink() or not holdout.is_dir():
            raise ValueError("Regular original holdout audit directory required")
        folders = sorted(holdout.iterdir())
        if len(folders) > 2048:
            raise ValueError("Bounded original holdout audit inventory required")
        for folder in folders:
            if folder.is_symlink():
                raise ValueError("Symlinked holdout audit folder rejected")
            if not folder.is_dir() or not (folder / "consumed.json").exists():
                continue
            for filename in ("commitment.json", "candidates.json", "consumed.json"):
                add((folder / filename).relative_to(root).as_posix())

    return dict(sorted(sources.items()))


def add_and_return(sources, root, relative):
    name = _relative_path(relative).as_posix()
    if name not in sources:
        sources[name] = _read_regular(root, name)
    return sources[name]


def manifest_for(sources):
    return {"schemaVersion": 1, "purpose": "remote-only experimental GVF residual TD3; original Simulator/iraLIB",
            "files": [{"path": name, "bytes": len(data), "sha256": sha256(data)}
                      for name, data in sources.items()]}


def write_bundle(destination, sources, manifest_bytes):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha256(manifest_bytes)
    if destination.exists():
        with zipfile.ZipFile(destination, "r") as archive:
            if archive.read("source-manifest.json") != manifest_bytes:
                raise FileExistsError("Refusing to overwrite a different AMD training bundle")
            for name, data in sources.items():
                if archive.read(name) != data:
                    raise ValueError("Existing bundle content differs: " + name)
            if set(archive.namelist()) != set(sources) | {"source-manifest.json"}:
                raise ValueError("Existing bundle has unexpected entries")
        return destination, sha256(destination.read_bytes()), manifest_sha

    descriptor, temporary = tempfile.mkstemp(prefix=".amd-gvf-", suffix=".zip.tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in sources.items():
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                archive.writestr(info, data)
            info = zipfile.ZipInfo("source-manifest.json")
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, manifest_bytes)
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination, sha256(destination.read_bytes()), manifest_sha


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a private, hash-bound AMD GVF training source bundle")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(argv)
    sources = collect_sources()
    manifest_bytes = (json.dumps(manifest_for(sources), allow_nan=False, sort_keys=True,
                                 separators=(",", ":")) + "\n").encode("ascii")
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = options.output or CACHE / f"amd-gvf-training-{sha256(manifest_bytes)[:16]}.zip"
    if destination.resolve().parent != CACHE.resolve():
        raise ValueError("AMD source bundle must remain inside the project .cache directory")
    bundle, bundle_sha, manifest_sha = write_bundle(destination, sources, manifest_bytes)
    summary = {"bundle": str(bundle), "sha256": bundle_sha, "sourceManifestSha256": manifest_sha,
               "sourceFiles": len(sources), "sourceBytes": sum(map(len, sources.values())),
               "trainingStarted": False, "secretsIncluded": False, "remoteTrainingPerformed": False}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
