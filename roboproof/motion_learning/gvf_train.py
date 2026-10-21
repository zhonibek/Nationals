"""Remote-only online TD3 fitting for the separate GVF residual contract."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import signal
import sys
import tempfile
import threading
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "roboproof/runs/motion/gvf-td3"
REPLAY_CAPACITY = 25000
BATCH_SIZE = 128
WARMUP_DECISIONS = 256
UPDATES_PER_EPISODE = 8
EXPLORATION_STD = 0.15
MANIFEST_PURPOSE = "remote-only experimental GVF residual TD3; original Simulator/iraLIB"
MAX_APPROVED_CREDITS = 100.0
MAX_EPISODE_DECISIONS = 200
RUN_DISK_RESERVE_BYTES = 4 * 1024 ** 3
EPISODE_OUTPUT_RESERVE_BYTES = 64 * 1024 ** 2
T4_EXPOSURE_LEDGER_SHA256 = "5ccc92f1200c3ee42f9ddacda009a4a5b6e35401a83038e756a49b19049f1eae"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require_approved_remote(approved):
    if (approved is not True or sys.platform != "linux" or platform.system() != "Linux" or
            os.environ.get("ROBOPROOF_APPROVED_REMOTE") != "1"):
        raise RuntimeError("Explicitly approved Linux remote required; no laptop fallback")


def validate_approved_credits(amount):
    require(type(amount) in (int, float) and math.isfinite(amount) and 0 < amount <= MAX_APPROVED_CREDITS,
            "Approved AMD credit cap must be finite, positive, and no greater than $100")
    return round(float(amount), 2)


def verify_sources(filename, expected):
    filename = Path(filename)
    require(filename.is_file() and not filename.is_symlink() and filename.stat().st_size <= 4 * 1024 * 1024,
            "Bounded regular frozen source manifest required")
    content = filename.read_bytes()
    require(digest(content) == expected, "Exact frozen source-manifest SHA256 required")
    manifest = json.loads(content)
    require(type(manifest) is dict and set(manifest) == {"schemaVersion", "purpose", "files"} and
            manifest["schemaVersion"] == 1 and manifest["purpose"] == MANIFEST_PURPOSE,
            "Exact versioned remote training source manifest required")
    rows = manifest["files"]
    require(type(rows) is list and 1 <= len(rows) <= 2048 and
            all(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and
                type(row["path"]) is str and type(row["bytes"]) is int and 1 <= row["bytes"] <= 8 * 1024 * 1024 and
                type(row["sha256"]) is str and len(row["sha256"]) == 64 and
                all(character in "0123456789abcdef" for character in row["sha256"]) for row in rows) and
            len({row["path"] for row in rows}) == len(rows) and
            sum(row["bytes"] for row in rows) <= 160 * 1024 * 1024,
            "Complete unique bounded source inventory required")
    for row in rows:
        name = PurePosixPath(row["path"])
        require(not name.is_absolute() and name.as_posix() == row["path"] and "\\" not in row["path"] and
                bool(name.parts) and all(part not in (".", "..") for part in name.parts),
                "Canonical contained source path required")
        target = ROOT.joinpath(*name.parts)
        for ancestor in [target, *target.parents]:
            if ancestor == ROOT.parent:
                break
            require(not ancestor.is_symlink(), "Source symlink/ancestor rejected")
        require(target.is_file() and target.resolve().is_relative_to(ROOT) and
                target.stat().st_size == row["bytes"] <= 8 * 1024 * 1024 and
                digest(target.read_bytes()) == row["sha256"], "Frozen source byte mismatch: " + row["path"])
    return content


def verify_mi300x(torch):
    if torch.version.hip is None or torch.version.cuda is not None or not torch.cuda.is_available():
        raise RuntimeError("Actual ROCm PyTorch and an available HIP device are required; no CPU fallback")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if "MI300X" not in properties.name.upper() or properties.total_memory < 180 * 1024 ** 3:
        raise RuntimeError("Actual AMD MI300X with at least 180 GiB is required")
    return device, {"name": properties.name, "totalMemoryBytes": properties.total_memory,
                    "hip": str(torch.version.hip), "torch": str(torch.__version__),
                    "cudaBuild": torch.version.cuda, "device": str(device)}


def verify_t4(torch):
    if torch.version.hip is not None or torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA PyTorch and an available NVIDIA device are required; no CPU fallback")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    capability = list(torch.cuda.get_device_capability(device))
    if ("T4" not in properties.name.upper() or properties.total_memory < 14 * 1024 ** 3 or
            properties.total_memory > 16 * 1024 ** 3 or capability != [7, 5]):
        raise RuntimeError("Actual Tesla T4 with capability 7.5 and 14-16 GiB is required")
    return device, {"name": properties.name, "totalMemoryBytes": properties.total_memory,
                    "hip": None, "torch": str(torch.__version__), "cudaBuild": str(torch.version.cuda),
                    "device": str(device), "deviceIndex": 0, "capability": capability,
                    "precision": "float32", "tf32": False,
                    "cublasWorkspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "simulation": "Original Node/WASM Simulator and iraLIB on CPU",
                    "scope": "Device visibility/runtime binding; successful network operations require actual training evidence"}


def load_t4_exposure_exclusion(filename, source_manifest_bytes, expected_sha256):
    require(expected_sha256 == T4_EXPOSURE_LEDGER_SHA256,
            "Only the independently audited frozen T4 exposure snapshot is admissible")
    filename = Path(filename)
    try:
        relative = filename.resolve(strict=True).relative_to(ROOT).as_posix()
    except (OSError, ValueError) as error:
        raise ValueError("T4 exposure snapshot must be an existing project-contained file") from error
    require(filename.is_file() and not filename.is_symlink() and filename.stat().st_size <= 4 * 1024 * 1024,
            "Bounded regular T4 exposure snapshot required")
    manifest = json.loads(source_manifest_bytes)
    rows = [row for row in manifest["files"] if row["path"] == relative]
    require(len(rows) == 1 and rows[0]["bytes"] == filename.stat().st_size and
            digest(filename.read_bytes()) == rows[0]["sha256"],
            "T4 exposure snapshot must be bound by the frozen source manifest")
    node = __import__("shutil").which("node")
    require(node is not None, "Original Simulator Node.js runtime required for exposure snapshot verification")
    script = ("const fs=require('node:fs');const c=require('./roboproof/motion-exposure-codec');"
        "const p=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));const d=c.decodeExposureSnapshot(p);"
        "const l=d.ledger;const x=l.unavailableExposureExclusions;"
        "const r=Array.isArray(x)?[...new Set(x.flatMap(v=>v.trainingSeeds||[]))].sort((a,b)=>a-b):[];"
        "process.stdout.write(JSON.stringify({sha256:d.ledgerSha256,worldSeeds:l.worldSeeds,"
        "bodyGeometries:l.bodyGeometries,reservedTrainingSeeds:r,legacyGeometryFullyKnown:l.legacyGeometryFullyKnown}));")
    output = __import__("subprocess").check_output([node, "-e", script, str(filename)], cwd=ROOT,
        text=True, timeout=60)
    value = json.loads(output)
    seeds, geometries, reserved = value.get("worldSeeds"), value.get("bodyGeometries"), value.get("reservedTrainingSeeds")
    require(value.get("sha256") == expected_sha256 and type(seeds) is list and type(geometries) is list and
            type(reserved) is list and seeds == sorted(set(seeds)) and geometries == sorted(set(geometries)) and
            reserved == [136001, 136002, 136003] and all(type(seed) is int and 0 <= seed <= 0xffffffff for seed in seeds) and
            all(type(geometry) is str and len(geometry) == 64 and
                all(character in "0123456789abcdef" for character in geometry) for geometry in geometries) and
            value.get("legacyGeometryFullyKnown") is False,
            "T4 exposure snapshot contents/reserved seeds do not match its frozen audited identity")
    receipt = {"sha256": expected_sha256, "reservedTrainingSeeds": reserved,
            "knownWorldSeedCount": len(seeds), "knownBodyGeometryCount": len(geometries),
            "legacyGeometryFullyKnown": False,
            "scope": "Known saved seed and body-relative geometry exclusions only; 833 seed-only records remain and arbitrary OOD separation is not claimed"}
    return {**receipt, "worldSeeds": set(seeds) | set(reserved), "bodyGeometries": set(geometries),
            "receipt": receipt}


def _world_geometry_keys(task):
    from .evaluation_exclusion import geometry_keys
    return geometry_keys(task)


def select_unexposed_training_world(generator, episode_index, curriculum, exposure, seen_seeds, seen_geometries):
    from .train import training_world
    for rejected in range(10000):
        try:
            world = training_world(generator, episode_index, curriculum)
        except RuntimeError as error:
            if str(error) not in ("Frozen evaluation world is excluded from training",
                                  "Consumed final world or body-geometry alias is excluded from training"):
                raise
            continue
        geometries = _world_geometry_keys(world["task"])
        if (world["seed"] in exposure["worldSeeds"] or world["seed"] in seen_seeds or
                geometries.intersection(exposure["bodyGeometries"]) or geometries.intersection(seen_geometries)):
            continue
        seen_seeds.add(world["seed"])
        seen_geometries.update(geometries)
        return world, rejected
    raise RuntimeError("T4 training world generator exhausted its bounded known-exposure rejection budget")


def validate_limits(seed, max_decisions, max_updates, max_seconds,
                    network_width=64, hidden_layers=2):
    require(type(seed) is int and 0 <= seed <= 0xffffffff, "Unsigned 32-bit training seed required")
    require(type(max_decisions) is int and 128 <= max_decisions <= 100000,
            "Decision budget must be between 128 and 100000")
    require(type(max_updates) is int and 1 <= max_updates <= 100000,
            "Update budget must be between 1 and 100000")
    require(type(max_seconds) is int and 60 <= max_seconds <= 3600,
            "Foreground wall-clock budget must be between 60 and 3600 seconds")
    from . import gvf_contract
    network_width, hidden_layers = gvf_contract.validate_architecture(network_width, hidden_layers)
    return {"seed": seed, "maxDecisions": max_decisions, "maxUpdates": max_updates,
            "maxSeconds": max_seconds, "algorithm": "td3-tanh-gvf-residual-v1",
            "decisionBudgetPolicy": "finish-current-episode-v1",
            "maxDecisionOvershoot": MAX_EPISODE_DECISIONS - 1,
            "networkWidth": network_width, "hiddenLayers": hidden_layers, "replayCapacity": REPLAY_CAPACITY,
            "batchSize": BATCH_SIZE, "warmupDecisions": WARMUP_DECISIONS,
            "updatesPerEpisode": UPDATES_PER_EPISODE, "exploration": "clipped-gaussian-additive-v1",
            "explorationStd": EXPLORATION_STD, "curriculum": "mixed-full-reach-v5",
            "reward": "unchanged-original-simulator-canonical-reward",
            "physics": "original-Simulator-iraLIB-only"}


def _json_bytes(value):
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")


def write_exclusive(path, value):
    path = Path(path)
    data = _json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(data)


def atomic_json(path, value):
    path = Path(path)
    data = _json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest(data)


def append_jsonl(path, value):
    data = _json_bytes(value)
    with Path(path).open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _random_state_to_json(state):
    return [state[0], list(state[1]), state[2]]


def _random_state_from_json(state):
    require(type(state) is list and len(state) == 3 and type(state[0]) is int and
            type(state[1]) is list and all(type(value) is int for value in state[1]) and
            (state[2] is None or type(state[2]) is float), "Exact serialized Python RNG state required")
    return state[0], tuple(state[1]), state[2]


class ReplayBuffer:
    def __init__(self, torch, capacity=REPLAY_CAPACITY):
        self.capacity = capacity
        self.count = self.cursor = 0
        self.observations = torch.zeros((capacity, 36), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 3), dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_observations = torch.zeros_like(self.observations)
        self.done = torch.zeros(capacity, dtype=torch.bool)

    def append(self, row):
        values = (row["features"], row["action"], row["reward"], row["nextFeatures"])
        if (len(values[0]) != 36 or len(values[1]) != 3 or len(values[3]) != 36 or
                any(not math.isfinite(value) for group in values[:2] + values[3:] for value in group) or
                not math.isfinite(values[2]) or any(abs(value) > 1 for value in values[1]) or
                type(row["done"]) is not bool):
            raise ValueError("Finite exact GVF transition required")
        index = self.cursor
        self.observations[index] = self.observations.new_tensor(values[0])
        self.actions[index] = self.actions.new_tensor(values[1])
        self.rewards[index] = values[2]
        self.next_observations[index] = self.next_observations.new_tensor(values[3])
        self.done[index] = row["done"]
        self.cursor = (self.cursor + 1) % self.capacity
        self.count = min(self.capacity, self.count + 1)

    def sample(self, torch, batch_size, device):
        if self.count < batch_size:
            raise ValueError("Insufficient complete episode replay rows")
        indices = torch.randint(self.count, (batch_size,), dtype=torch.int64)
        return (self.observations[indices].to(device), self.actions[indices].to(device),
                self.rewards[indices].to(device), self.next_observations[indices].to(device),
                self.done[indices].to(device))

    def state_dict(self):
        return {"capacity": self.capacity, "count": self.count, "cursor": self.cursor,
                "observations": self.observations.clone(), "actions": self.actions.clone(),
                "rewards": self.rewards.clone(), "nextObservations": self.next_observations.clone(),
                "done": self.done.clone()}

    def load_state_dict(self, torch, state):
        require(type(state) is dict and set(state) == {"capacity", "count", "cursor", "observations",
                "actions", "rewards", "nextObservations", "done"}, "Exact GVF replay state required")
        count, cursor = state["count"], state["cursor"]
        require(type(state["capacity"]) is int and state["capacity"] == self.capacity and
                type(count) is int and 0 <= count <= self.capacity and type(cursor) is int and
                0 <= cursor < self.capacity and (count == self.capacity or cursor == count),
                "GVF replay capacity/cursor/count mismatch")
        expected = {"observations": self.observations, "actions": self.actions, "rewards": self.rewards,
            "nextObservations": self.next_observations, "done": self.done}
        for name, target in expected.items():
            value = state[name]
            require(torch.is_tensor(value) and tuple(value.shape) == tuple(target.shape) and
                    value.dtype == target.dtype and value.device == target.device and value.layout == target.layout,
                    "GVF replay tensor shape/dtype/device mismatch: " + name)
            if name != "done":
                require(torch.isfinite(value).all().item(), "Nonfinite GVF replay state: " + name)
        require((state["actions"][:count].abs() <= 1).all().item(), "GVF replay actions outside [-1,1]")
        if count < self.capacity:
            for name, value in state.items():
                if name in expected and value.ndim > 1 and not torch.equal(value[count:], torch.zeros_like(value[count:])):
                    raise ValueError("Unused GVF replay rows must remain zero: " + name)
                if name in ("rewards", "done") and not torch.equal(value[count:], torch.zeros_like(value[count:])):
                    raise ValueError("Unused GVF replay rows must remain zero: " + name)
        for name, target in expected.items():
            target.copy_(state[name])
        self.count = count
        self.cursor = cursor


def _safe_output(path, resume=False):
    path = Path(path)
    parent = OUTPUT_ROOT
    if path.parent.resolve() != parent.resolve() or not path.name.startswith("gvf-td3-"):
        raise ValueError("Bounded run directory under roboproof/runs/motion/gvf-td3 required")
    for ancestor in [parent, *parent.parents]:
        if ancestor == ROOT.parent:
            break
        if ancestor.is_symlink():
            raise ValueError("Training output path/ancestor must not be a symlink")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if resume:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Explicit resume requires an existing regular run directory")
    else:
        if path.exists() or path.is_symlink():
            raise ValueError("Never overwrite or reuse a training run directory")
        path.mkdir(mode=0o700)
    return path


def _save_checkpoint(torch, learner, replay, rng, directory, identity, source_sha256,
                     config, episode_count, decisions, run_id, seen_world_seeds,
                     seen_body_geometries, rejected_training_worlds):
    payload = {"schemaVersion": 2, "runId": run_id, "seed": config["seed"], "config": config,
        "sourceManifestSha256": source_sha256, "engineIdentity": identity,
        "episodeCount": episode_count, "decisions": decisions,
        "pythonRandomState": _random_state_to_json(rng.getstate()),
        "seenWorldSeeds": sorted(seen_world_seeds), "seenBodyGeometries": sorted(seen_body_geometries),
        "trainingWorldRejections": rejected_training_worlds,
        "learner": learner.state_dict(), "replay": replay.state_dict()}
    target = directory / "checkpoints" / f"episode-{episode_count:06d}.pt"
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=target.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            raise FileExistsError("Checkpoint path already exists")
        os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    checkpoint_sha = digest(target.read_bytes())
    append_jsonl(directory / "checkpoints.jsonl", {"path": target.relative_to(directory).as_posix(),
        "sha256": checkpoint_sha, "bytes": target.stat().st_size, "episodeCount": episode_count,
        "decisions": decisions, "createdAtUtc": utc_now()})
    return target, checkpoint_sha


def _checkpoint_size_upper_bound(learner, replay):
    model_bytes = sum(tensor.numel() * tensor.element_size()
        for model in (learner.actor, learner.critics, learner.actor_target, learner.critics_target)
        for tensor in model.state_dict().values())
    trainable = (learner.actor, learner.critics)
    trainable_bytes = sum(parameter.numel() * parameter.element_size()
        for model in trainable for parameter in model.parameters())
    optimizer_steps = 1024 * sum(1 for model in trainable for _ in model.parameters())
    replay_bytes = sum(tensor.numel() * tensor.element_size()
        for tensor in (replay.observations, replay.actions, replay.rewards,
                       replay.next_observations, replay.done))
    return math.ceil((model_bytes + 2 * trainable_bytes + optimizer_steps + replay_bytes) * 1.25) + 16 * 1024 ** 2


def _read_run_json(path, maximum=4 * 1024 * 1024):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum,
            "Bounded regular resume evidence required: " + path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def _preserve_terminal_record(path, record):
    path = Path(path)
    if path.exists():
        require(_read_run_json(path) == record,
                "Existing pre-resume terminal receipt differs from the validated stop")
        return
    write_exclusive(path, record)


def _validate_resume_history(directory, record, source_sha256):
    events_path = directory / "exposures.jsonl"
    require(events_path.is_file() and not events_path.is_symlink() and events_path.stat().st_size <= 64 * 1024 * 1024,
            "Bounded complete training exposure history required")
    lines = events_path.read_text(encoding="utf-8").splitlines()
    episodes = record["episodes"]
    require(len(lines) == episodes * 2, "Only fully completed episode exposure history is resumable")
    worlds, total_decisions = [], 0
    for episode in range(1, episodes + 1):
        started = json.loads(lines[(episode - 1) * 2])
        completed = json.loads(lines[(episode - 1) * 2 + 1])
        require(started.get("event") == "started" and completed.get("event") == "completed" and
                started.get("episode") == episode and completed.get("episode") == episode and
                started.get("world") == completed.get("world") and
                type(completed.get("decisionCount")) is int and
                1 <= completed["decisionCount"] <= MAX_EPISODE_DECISIONS and
                started.get("sourceManifestSha256") == source_sha256 == completed.get("sourceManifestSha256"),
                "Training exposure history must be consecutive, paired and source-bound")
        episode_path = directory / "episodes" / f"episode-{episode:06d}.json"
        require(episode_path.is_file() and not episode_path.is_symlink() and episode_path.stat().st_size <= 8 * 1024 * 1024 and
                digest(episode_path.read_bytes()) == completed.get("episodeFileSha256"),
                "Completed episode artifact differs from its exposure record")
        saved_episode = _read_run_json(episode_path, 8 * 1024 * 1024)
        require(saved_episode.get("episode") == episode and saved_episode.get("complete") is True and
                saved_episode.get("world") == started["world"] and
                saved_episode.get("decisionCount") == completed.get("decisionCount") and
                len(saved_episode.get("transitions", [])) == completed.get("decisionCount"),
                "Completed episode artifact/counter mismatch")
        worlds.append(started["world"])
        total_decisions += completed["decisionCount"]
    require(total_decisions == record["decisions"], "Completed episode decisions do not match terminal run")
    return worlds


def _load_resume_state(torch, learner, replay, rng, directory, run_id, config,
                       source_bytes, source_sha256, gpu, approval, exposure_exclusion, identity):
    record = _read_run_json(directory / "run.json")
    terminal = _read_run_json(directory / "terminal.json")
    require(record == terminal and record.get("status") in
            ("time-budget-stopped", "storage-budget-stopped", "signal-stopped") and
            record.get("resumeSupported") is True and record.get("unresolvedExposure") is None,
            "Only a clean, explicitly resumable stop may continue")
    require(record.get("runId") == run_id and record.get("config") == config and
            record.get("sourceManifestSha256") == source_sha256 and
            record.get("trainingDevice") == gpu and record.get("operatorDeclaredApproval") == approval and
            record.get("knownExposureExclusion") == (exposure_exclusion["receipt"] if exposure_exclusion else None) and
            record.get("engineIdentity") == identity,
            "Resume source/configuration/provider/exposure/engine identity mismatch")
    source_path = directory / "source-manifest.json"
    require(source_path.is_file() and not source_path.is_symlink() and source_path.stat().st_size <= 4 * 1024 * 1024 and
            source_path.read_bytes() == source_bytes,
            "Resume source manifest bytes changed")
    host = _read_run_json(directory / "host.json")
    current_host = {"system": platform.platform(), "python": platform.python_version(),
        "node": __import__("subprocess").check_output(["node", "--version"], text=True).strip(),
        "gpu": gpu, "gpuUsedForOptimizer": True}
    require(all(host.get(name) == value for name, value in current_host.items()),
            "Resume host/runtime/GPU identity mismatch")
    require(type(record.get("episodes")) is int and record["episodes"] > 0 and
            type(record.get("decisions")) is int and record["decisions"] > 0 and
            type(record.get("updates")) is int and 0 <= record["updates"] <= config["maxUpdates"] and
            type(record.get("resumeCount")) is int and record["resumeCount"] >= 0,
            "Resumable run counters required")
    elapsed = record.get("cumulativeElapsedSeconds", record.get("elapsedSeconds"))
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
            "Finite cumulative elapsed-time receipt required")
    exposed_worlds = _validate_resume_history(directory, record, source_sha256)
    initial = record.get("initialActor")
    actor = record.get("actor")
    checkpoint = record.get("checkpoint")
    last_episode = record.get("lastEpisode")
    for pointer, name in ((initial, "initial actor"), (actor, "current actor"),
                          (checkpoint, "checkpoint"), (last_episode, "last episode")):
        require(type(pointer) is dict and set(pointer) == {"path", "sha256"} and
                type(pointer["path"]) is str and type(pointer["sha256"]) is str and
                len(pointer["sha256"]) == 64 and all(character in "0123456789abcdef" for character in pointer["sha256"]),
                "Exact saved " + name + " receipt required")
    require(initial["path"] == "actors/actor-episode-000000.json" and
            actor["path"] == f"actors/actor-episode-{record['episodes']:06d}.json" and
            checkpoint["path"] == f"checkpoints/episode-{record['episodes']:06d}.pt" and
            last_episode["path"] == f"episodes/episode-{record['episodes']:06d}.json",
            "Resume artifact pointers/counters must match the committed episode")
    initial_path = directory / initial["path"]
    actor_path = directory / actor["path"]
    last_episode_path = directory / last_episode["path"]
    for path, pointer, label in ((initial_path, initial, "initial actor"), (actor_path, actor, "current actor"),
                                 (last_episode_path, last_episode, "last episode")):
        require(path.resolve().is_relative_to(directory.resolve()) and path.is_file() and not path.is_symlink() and
                path.stat().st_size <= 8 * 1024 * 1024 and
                digest(path.read_bytes()) == pointer["sha256"], "Saved " + label + " hash mismatch")
    checkpoint_path = directory / checkpoint["path"]
    require(checkpoint_path.resolve().is_relative_to(directory.resolve()) and checkpoint_path.is_file() and
            not checkpoint_path.is_symlink() and checkpoint_path.stat().st_size <= 1024 * 1024 * 1024 and
            digest(checkpoint_path.read_bytes()) == checkpoint["sha256"], "Saved checkpoint hash mismatch")
    history_path = directory / "checkpoints.jsonl"
    require(history_path.is_file() and not history_path.is_symlink() and history_path.stat().st_size <= 64 * 1024 * 1024,
            "Bounded checkpoint receipt history required")
    checkpoint_rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    require(checkpoint_rows and checkpoint_rows[-1].get("path") == checkpoint["path"] and
            checkpoint_rows[-1].get("sha256") == checkpoint["sha256"] and
            checkpoint_rows[-1].get("bytes") == checkpoint_path.stat().st_size and
            checkpoint_rows[-1].get("episodeCount") == record["episodes"] and
            checkpoint_rows[-1].get("decisions") == record["decisions"],
            "Latest checkpoint receipt does not match terminal run")
    payload = torch.load(checkpoint_path, weights_only=True)
    expected_keys = {"schemaVersion", "runId", "seed", "config", "sourceManifestSha256", "engineIdentity",
        "episodeCount", "decisions", "pythonRandomState", "seenWorldSeeds", "seenBodyGeometries",
        "trainingWorldRejections", "learner", "replay"}
    require(type(payload) is dict and set(payload) == expected_keys and payload["schemaVersion"] == 2 and
            payload["runId"] == run_id and payload["seed"] == config["seed"] and
            payload["config"] == config and payload["sourceManifestSha256"] == source_sha256 and
            payload["engineIdentity"] == identity and payload["episodeCount"] == record["episodes"] and
            payload["decisions"] == record["decisions"] and
            payload["trainingWorldRejections"] == record["trainingExposureRejections"],
            "Checkpoint run/source/engine/counter mismatch")
    seen_seeds, seen_geometry = payload["seenWorldSeeds"], payload["seenBodyGeometries"]
    require(type(seen_seeds) is list and seen_seeds == sorted(set(seen_seeds)) and
            all(type(seed) is int and 0 <= seed <= 0xffffffff for seed in seen_seeds) and
            type(seen_geometry) is list and seen_geometry == sorted(set(seen_geometry)) and
            all(type(value) is str and len(value) == 64 and
                all(character in "0123456789abcdef" for character in value) for value in seen_geometry) and
            type(payload["trainingWorldRejections"]) is int and payload["trainingWorldRejections"] >= 0,
            "Checkpoint exposure/RNG continuation state malformed")
    if exposure_exclusion is None:
        require(not seen_seeds and not seen_geometry,
                "Non-T4 resume must not consume T4-only duplicate exposure state")
    else:
        expected_seeds = {world["seed"] for world in exposed_worlds}
        expected_geometry = set().union(*(_world_geometry_keys(world["task"]) for world in exposed_worlds))
        require(set(seen_seeds) == expected_seeds and set(seen_geometry) == expected_geometry,
                "Checkpoint duplicate-exposure state differs from completed T4 world history")
    require(type(payload["learner"]) is dict and payload["learner"].get("updates") == record["updates"] and
            type(record.get("trainingExposureRejections")) is int and record["trainingExposureRejections"] ==
            payload["trainingWorldRejections"], "Checkpoint learner/rejection counters differ from terminal run")
    replay.load_state_dict(torch, payload["replay"])
    require(replay.count == min(record["decisions"], REPLAY_CAPACITY) and
            replay.cursor == record["decisions"] % REPLAY_CAPACITY,
            "Restored replay count/cursor does not match completed decisions")
    learner.load_state_dict(payload["learner"])
    rng.setstate(_random_state_from_json(payload["pythonRandomState"]))
    return {"record": record, "initialSha256": initial["sha256"], "actorSha256": actor["sha256"],
        "checkpoint": checkpoint, "seenWorldSeeds": set(seen_seeds), "seenBodyGeometries": set(seen_geometry),
        "trainingWorldRejections": payload["trainingWorldRejections"],
        "cumulativeElapsedSeconds": record.get("cumulativeElapsedSeconds", record.get("elapsedSeconds", 0.0))}


def run(approved_amd_mi300x, source_manifest, source_manifest_sha256, output, seed=50101,
        max_decisions=10000, max_updates=10000, max_seconds=1800, approved_max_credits=None,
        network_width=64, hidden_layers=2, approved_free_t4=False,
        exposure_ledger=None, exposure_ledger_sha256=None, resume=False, stop_requested=None):
    require(type(resume) is bool, "Explicit boolean resume option required")
    if stop_requested is None:
        stop_requested = lambda: False
    require(callable(stop_requested), "A callable graceful-stop check is required")
    if type(approved_amd_mi300x) is not bool or type(approved_free_t4) is not bool or approved_amd_mi300x == approved_free_t4:
        raise ValueError("Exactly one explicitly approved GPU provider is required")
    require_approved_remote(True)
    approved_credits = validate_approved_credits(approved_max_credits) if approved_amd_mi300x else None
    if approved_free_t4 and approved_max_credits is not None:
        raise ValueError("Free T4 execution cannot declare or consume paid credits")
    config = validate_limits(seed, max_decisions, max_updates, max_seconds,
                             network_width, hidden_layers)
    source_bytes = verify_sources(source_manifest, source_manifest_sha256)
    exposure_exclusion = None
    if approved_free_t4:
        require(exposure_ledger is not None, "Free T4 training requires a frozen known-exposure snapshot")
        exposure_exclusion = load_t4_exposure_exclusion(exposure_ledger, source_bytes, exposure_ledger_sha256)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif exposure_ledger is not None or exposure_ledger_sha256 is not None:
        raise ValueError("AMD study cannot silently consume the T4-specific exposure declaration")
    import torch
    device, gpu = verify_mi300x(torch) if approved_amd_mi300x else verify_t4(torch)
    if approved_free_t4:
        torch.backends.cuda.matmul.allow_tf32 = False
    from roboproof.motion_env import MotionEnvironment
    from . import gvf_contract
    from .gvf_model import atomic_actor_json, export
    from .gvf_td3 import GvfTD3
    from .train import training_world

    run_id = Path(output).name.removeprefix("gvf-td3-")
    if not run_id or len(run_id) > 64 or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
        raise ValueError("Bounded explicit run ID required in output directory name")
    directory = _safe_output(output, resume=resume)
    previous = _read_run_json(directory / "run.json") if resume else None
    if not resume:
        (directory / "episodes").mkdir(mode=0o700)
        (directory / "actors").mkdir(mode=0o700)
        (directory / "checkpoints").mkdir(mode=0o700)
        (directory / "source-manifest.json").write_bytes(source_bytes)
        os.chmod(directory / "source-manifest.json", 0o600)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    learner = GvfTD3(width=network_width, hidden_layers=hidden_layers, device=device)
    replay = ReplayBuffer(torch)
    checkpoint_upper_bound = _checkpoint_size_upper_bound(learner, replay)
    decisions = episodes = 0
    status = "failed"
    unresolved_exposure = None
    started = time.monotonic()
    deadline = started + max_seconds
    events_path = directory / "exposures.jsonl"
    transitions_path = directory / "transitions.jsonl"
    seen_world_seeds, seen_body_geometries = set(), set()
    rejected_training_worlds = 0
    approval = ({"provider": "AMD Developer Cloud", "device": "MI300X",
            "maxCredits": approved_credits, "billingBasis": "existing AMD account credits only",
            "paymentMethodChargeAuthorized": False, "autoReloadAuthorized": False}
        if approved_amd_mi300x else
        {"provider": "Google Colab", "device": "free Tesla T4", "maxCredits": 0,
            "billingBasis": "free runtime only", "paymentMethodChargeAuthorized": False,
            "autoReloadAuthorized": False})
    resume_count = (previous.get("resumeCount", 0) + 1) if resume else 0
    protocol = {"schemaVersion": 1, "runId": run_id, "config": config,
        "sourceManifestSha256": source_manifest_sha256,
        "createdAtUtc": previous.get("createdAtUtc", utc_now()) if resume else utc_now(),
        "resumeCount": resume_count,
        "operatorDeclaredApproval": approval,
        "knownExposureExclusion": exposure_exclusion["receipt"] if exposure_exclusion is not None else None,
        "trainingDevice": gpu, "physicsDevice": "Original Simulator Node.js process on the same approved remote runtime",
        "trainingDriver": "roboproof.motion_learning.gvf_train", "resumeSupported": True,
        "learnedImprovementVerified": False, "independentQualificationPerformed": False,
        "scope": "Bounded experimental GVF residual TD3 fit; no promotion or physical/game-score claim"}
    host_record = {"checkedAtUtc": utc_now(), "system": platform.platform(),
        "python": platform.python_version(), "node": __import__("subprocess").check_output(["node", "--version"], text=True).strip(),
        "gpu": gpu, "gpuUsedForOptimizer": True}
    if not resume:
        atomic_json(directory / "run.json", {**protocol, "status": "running", "decisions": 0, "episodes": 0, "updates": 0})
        write_exclusive(directory / "host.json", host_record)
        events_path.touch(mode=0o600, exist_ok=False)
        transitions_path.touch(mode=0o600, exist_ok=False)
        (directory / "segments.jsonl").touch(mode=0o600, exist_ok=False)
    else:
        require((directory / "segments.jsonl").is_file() and not (directory / "segments.jsonl").is_symlink(),
                "Append-only training segment history required for resume")
    try:
        with MotionEnvironment(control_mode="gvf-residual") as environment:
            identity = environment.contract["identity"]
            if resume:
                resumed = _load_resume_state(torch, learner, replay, rng, directory, run_id, config,
                    source_bytes, source_manifest_sha256, gpu, approval,
                    exposure_exclusion, identity)
                previous = resumed["record"]
                decisions, episodes = previous["decisions"], previous["episodes"]
                seen_world_seeds = resumed["seenWorldSeeds"]
                seen_body_geometries = resumed["seenBodyGeometries"]
                rejected_training_worlds = resumed["trainingWorldRejections"]
                prior_elapsed = resumed["cumulativeElapsedSeconds"]
                initial_sha, current_actor_sha = resumed["initialSha256"], resumed["actorSha256"]
                checkpoint_receipt = resumed["checkpoint"]
                initial_actor_receipt = previous["initialActor"]
                actor_receipt = previous["actor"]
                last_episode_receipt = previous["lastEpisode"]
                history_dir = directory / "terminal-history"
                history_dir.mkdir(mode=0o700, exist_ok=True)
                history_target = history_dir / f"terminal-before-resume-{resume_count:06d}.json"
                _preserve_terminal_record(history_target, previous)
                append_jsonl(directory / "segments.jsonl", {"event": "resumed", "resumeCount": resume_count,
                    "atUtc": utc_now(), "priorStatus": previous["status"], "checkpointSha256": checkpoint_receipt["sha256"]})
            else:
                prior_elapsed = 0.0
                initial_actor = export(learner.actor, identity, seed, 0, 0)
                initial_path = directory / "actors/actor-episode-000000.json"
                atomic_actor_json(initial_path, initial_actor)
                initial_sha = digest(initial_path.read_bytes())
                initial_actor_receipt = {"path": initial_path.relative_to(directory).as_posix(), "sha256": initial_sha}
                current_actor_sha = None
                checkpoint_receipt = None
                actor_receipt = None
                last_episode_receipt = None
            append_jsonl(directory / "segments.jsonl", {"event": "started", "resumeCount": resume_count,
                "atUtc": utc_now(), "seed": seed, "maxSeconds": max_seconds})
            atomic_json(directory / "run.json", {**protocol, "status": "running", "engineIdentity": identity,
                "initialActor": initial_actor_receipt, "actor": actor_receipt, "checkpoint": checkpoint_receipt,
                "lastEpisode": last_episode_receipt, "decisions": decisions, "episodes": episodes,
                "updates": learner.updates, "cumulativeElapsedSeconds": prior_elapsed})
            while decisions < max_decisions and learner.updates < max_updates and time.monotonic() < deadline:
                if stop_requested() and episodes > 0:
                    status = "signal-stopped"
                    break
                free_disk_bytes = __import__("shutil").disk_usage(directory).free
                minimum_free_bytes = checkpoint_upper_bound + EPISODE_OUTPUT_RESERVE_BYTES + RUN_DISK_RESERVE_BYTES
                if free_disk_bytes < minimum_free_bytes:
                    status = "storage-budget-stopped"
                    break
                if exposure_exclusion is None:
                    world = training_world(rng, episodes, "mixed-full-reach-v5")
                else:
                    world, rejected = select_unexposed_training_world(rng, episodes,
                        "mixed-full-reach-v5", exposure_exclusion, seen_world_seeds, seen_body_geometries)
                    rejected_training_worlds += rejected
                episode_exposure = {"event": "started", "episode": episodes + 1, "world": world,
                    "actorSha256": initial_sha if learner.updates == 0 else current_actor_sha,
                    "sourceManifestSha256": source_manifest_sha256, "startedAtUtc": utc_now()}
                append_jsonl(events_path, episode_exposure)
                unresolved_exposure = episode_exposure
                actor_sha = initial_sha if learner.updates == 0 else current_actor_sha
                policy_identity = {"kind": "untrained" if learner.updates == 0 else "learned",
                    "id": f"gvf-td3-seed-{seed}-episode-{episodes + 1}", "sha256": actor_sha}
                observation, _ = environment.reset(seed=world["seed"], task=world["task"],
                    configuration=world["configuration"], options={"recordTransitions": False,
                    "policyIdentity": policy_identity})
                episode_rows = []
                episode_decisions = 0
                episode_start = decisions
                partial = False
                while True:
                    if episode_decisions >= MAX_EPISODE_DECISIONS:
                        partial = True
                        break
                    structured = gvf_contract.observation_from_vector(list(observation),
                        contract_name=gvf_contract.CONTRACT_NAME,
                        contract_version=gvf_contract.CONTRACT_VERSION,
                        observation_schema=gvf_contract.OBSERVATION_SCHEMA)
                    if learner.updates == 0:
                        base_action = [0.0, 0.0, 0.0]
                    else:
                        base_action = learner.actor.act(structured)
                    noise_std = EXPLORATION_STD
                    if decisions < WARMUP_DECISIONS:
                        base_action = [0.0, 0.0, 0.0]
                    action = [max(-1.0, min(1.0, value + rng.gauss(0.0, noise_std)))
                              for value in base_action]
                    features = gvf_contract.encode_observation(structured)
                    following, reward, terminated, truncated, info = environment.step(action)
                    next_structured = gvf_contract.observation_from_vector(list(following),
                        contract_name=gvf_contract.CONTRACT_NAME,
                        contract_version=gvf_contract.CONTRACT_VERSION,
                        observation_schema=gvf_contract.OBSERVATION_SCHEMA)
                    episode_rows.append({"observation": list(observation), "features": features,
                        "action": action, "reward": float(reward), "nextObservation": list(following),
                        "nextFeatures": gvf_contract.encode_observation(next_structured),
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "done": bool(terminated or truncated), "info": info})
                    decisions += 1
                    episode_decisions += 1
                    observation = following
                    if terminated or truncated:
                        break
                report = environment.report()
                episode_record = {"schemaVersion": 1, "episode": episodes + 1, "world": world,
                    "actorSha256": actor_sha, "policyIdentity": policy_identity,
                    "complete": not partial, "stopReason": "budget-interrupted-exposure" if partial else None,
                    "decisionCount": episode_decisions, "decisionsBefore": episode_start,
                    "transitions": episode_rows, "originalSimulatorReport": report,
                    "createdAtUtc": utc_now()}
                episode_path = directory / "episodes" / f"episode-{episodes + 1:06d}.json"
                episode_sha = write_exclusive(episode_path, episode_record)
                if partial:
                    append_jsonl(events_path, {"event": "partial-unresolved", "episode": episodes + 1,
                        "world": world, "decisions": episode_decisions, "episodeFileSha256": episode_sha,
                        "sourceManifestSha256": source_manifest_sha256,
                        "atUtc": utc_now()})
                    unresolved_exposure = {**episode_exposure, "status": "partial-unresolved", "episodeFileSha256": episode_sha}
                    status = "budget-stopped-with-unresolved-exposure"
                    break
                for row in episode_rows:
                    replay.append(row)
                    with transitions_path.open("ab") as stream:
                        stream.write(_json_bytes({"episode": episodes + 1, **row}))
                episodes += 1
                update_results = []
                if replay.count >= BATCH_SIZE:
                    remaining_updates = min(UPDATES_PER_EPISODE, max_updates - learner.updates)
                    for _ in range(remaining_updates):
                        batch = replay.sample(torch, BATCH_SIZE, device)
                        update_results.append(learner.update(*batch))
                actor = export(learner.actor, identity, seed, decisions, learner.updates)
                actor_path = directory / "actors" / f"actor-episode-{episodes:06d}.json"
                atomic_actor_json(actor_path, actor)
                current_actor_sha = digest(actor_path.read_bytes())
                checkpoint_path, checkpoint_sha = _save_checkpoint(torch, learner, replay, rng,
                    directory, identity, source_manifest_sha256, config, episodes, decisions, run_id,
                    seen_world_seeds, seen_body_geometries, rejected_training_worlds)
                append_jsonl(events_path, {"event": "completed", "episode": episodes, "world": world,
                    "reason": report["reason"], "decisionCount": episode_decisions,
                    "reward": report["totalReward"], "metrics": report["metrics"],
                    "actorSha256": current_actor_sha, "episodeFileSha256": episode_sha,
                    "sourceManifestSha256": source_manifest_sha256,
                    "checkpointSha256": checkpoint_sha, "completedAtUtc": utc_now()})
                unresolved_exposure = None
                actor_receipt = {"path": actor_path.relative_to(directory).as_posix(), "sha256": current_actor_sha}
                checkpoint_receipt = {"path": checkpoint_path.relative_to(directory).as_posix(), "sha256": checkpoint_sha}
                last_episode_receipt = {"path": episode_path.relative_to(directory).as_posix(), "sha256": episode_sha}
                current = {**protocol, "status": "running", "engineIdentity": identity,
                    "decisions": decisions, "episodes": episodes, "updates": learner.updates,
                    "initialActor": initial_actor_receipt, "actor": actor_receipt,
                    "checkpoint": checkpoint_receipt, "lastEpisode": last_episode_receipt,
                    "cumulativeElapsedSeconds": prior_elapsed + time.monotonic() - started}
                atomic_json(directory / "run.json", current)
                print(json.dumps({"episode": episodes, "decisions": decisions, "updates": learner.updates,
                    "reason": report["reason"], "elapsedSeconds": round(time.monotonic() - started, 2)},
                    allow_nan=False), flush=True)
            if status == "failed":
                if decisions >= max_decisions or learner.updates >= max_updates:
                    status = "completed-budget"
                elif stop_requested() and episodes > 0:
                    status = "signal-stopped"
                else:
                    status = "time-budget-stopped"
            terminal = {**protocol, "status": status, "engineIdentity": environment.contract["identity"],
                "decisions": decisions, "episodes": episodes, "updates": learner.updates,
                "initialActor": initial_actor_receipt, "actor": actor_receipt,
                "checkpoint": checkpoint_receipt, "lastEpisode": last_episode_receipt,
                "trainingExposureRejections": rejected_training_worlds,
                "checkpointSizeUpperBoundBytes": checkpoint_upper_bound,
                "freeDiskBytesAtStop": __import__("shutil").disk_usage(directory).free,
                "reservedDiskBytes": RUN_DISK_RESERVE_BYTES + EPISODE_OUTPUT_RESERVE_BYTES,
                "segmentElapsedSeconds": round(time.monotonic() - started, 3),
                "cumulativeElapsedSeconds": round(prior_elapsed + time.monotonic() - started, 3),
                "unresolvedExposure": unresolved_exposure,
                "learnedImprovementVerified": False, "independentQualificationPerformed": False,
                "terminalAtUtc": utc_now()}
            terminal["elapsedSeconds"] = terminal["cumulativeElapsedSeconds"]
            atomic_json(directory / "run.json", terminal)
            atomic_json(directory / "terminal.json", terminal)
            append_jsonl(directory / "segments.jsonl", {"event": "stopped", "resumeCount": resume_count,
                "status": status, "atUtc": terminal["terminalAtUtc"], "decisions": decisions,
                "episodes": episodes, "updates": learner.updates})
        return terminal
    except BaseException as error:
        failure = {**protocol, "status": "failed", "error": str(error),
            "traceback": traceback.format_exc(), "decisions": decisions, "episodes": episodes,
            "updates": learner.updates, "trainingExposureRejections": rejected_training_worlds,
            "unresolvedExposure": unresolved_exposure,
            "learnedImprovementVerified": False, "independentQualificationPerformed": False,
            "terminalAtUtc": utc_now()}
        atomic_json(directory / "run.json", failure)
        atomic_json(directory / "terminal.json", failure)
        append_jsonl(directory / "segments.jsonl", {"event": "failed", "resumeCount": resume_count,
            "atUtc": failure["terminalAtUtc"], "decisions": decisions, "episodes": episodes,
            "updates": learner.updates})
        raise


def _install_graceful_stop_handlers():
    stop_event = threading.Event()
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda _signum, _frame: stop_event.set())
    return stop_event, previous_handlers


def _restore_signal_handlers(previous_handlers):
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded remote-only GVF residual TD3 training on original Simulator/iraLIB")
    approval = parser.add_mutually_exclusive_group(required=True)
    approval.add_argument("--approved-amd-mi300x", action="store_true")
    approval.add_argument("--approved-free-t4", action="store_true")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--approved-max-credits", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=50101)
    parser.add_argument("--max-decisions", type=int, default=10000)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--network-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--exposure-ledger", type=Path)
    parser.add_argument("--exposure-ledger-sha256")
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args(argv)
    stop_event, previous_handlers = _install_graceful_stop_handlers()
    try:
        result = run(**vars(options), stop_requested=stop_event.is_set)
    finally:
        _restore_signal_handlers(previous_handlers)
    print(json.dumps({"status": result["status"], "decisions": result["decisions"],
        "episodes": result["episodes"], "updates": result["updates"],
        "learnedImprovementVerified": False, "independentQualificationPerformed": False}), flush=True)
    accepted_statuses = ("completed-budget", "time-budget-stopped", "storage-budget-stopped", "signal-stopped")
    return 0 if result["status"] in accepted_statuses else 1


if __name__ == "__main__":
    sys.exit(main())
