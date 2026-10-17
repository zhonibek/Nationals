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
import sys
import tempfile
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


def _safe_output(path):
    path = Path(path)
    parent = OUTPUT_ROOT
    if path.parent.resolve() != parent.resolve() or not path.name.startswith("gvf-td3-"):
        raise ValueError("New bounded run directory under roboproof/runs/motion/gvf-td3 required")
    for ancestor in [parent, *parent.parents]:
        if ancestor == ROOT.parent:
            break
        if ancestor.is_symlink():
            raise ValueError("Training output path/ancestor must not be a symlink")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise ValueError("Never overwrite or reuse a training run directory")
    path.mkdir(mode=0o700)
    return path


def _save_checkpoint(torch, learner, replay, rng, directory, identity, source_sha256,
                     config, episode_count, decisions, run_id):
    payload = {"schemaVersion": 1, "runId": run_id, "seed": config["seed"], "config": config,
        "sourceManifestSha256": source_sha256, "engineIdentity": identity,
        "episodeCount": episode_count, "decisions": decisions,
        "pythonRandomState": _random_state_to_json(rng.getstate()),
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


def run(approved_amd_mi300x, source_manifest, source_manifest_sha256, output, seed=50101,
        max_decisions=10000, max_updates=10000, max_seconds=1800, approved_max_credits=None,
        network_width=64, hidden_layers=2):
    require_approved_remote(approved_amd_mi300x)
    approved_credits = validate_approved_credits(approved_max_credits)
    config = validate_limits(seed, max_decisions, max_updates, max_seconds,
                             network_width, hidden_layers)
    source_bytes = verify_sources(source_manifest, source_manifest_sha256)
    import torch
    device, gpu = verify_mi300x(torch)
    from roboproof.motion_env import MotionEnvironment
    from . import gvf_contract
    from .gvf_model import atomic_actor_json, export
    from .gvf_td3 import GvfTD3
    from .train import training_world

    run_id = Path(output).name.removeprefix("gvf-td3-")
    if not run_id or len(run_id) > 64 or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
        raise ValueError("Bounded explicit run ID required in output directory name")
    directory = _safe_output(output)
    (directory / "episodes").mkdir(mode=0o700)
    (directory / "actors").mkdir(mode=0o700)
    (directory / "checkpoints").mkdir(mode=0o700)
    (directory / "source-manifest.json").write_bytes(source_bytes)
    for filename in (directory / "source-manifest.json",):
        os.chmod(filename, 0o600)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    learner = GvfTD3(width=network_width, hidden_layers=hidden_layers, device=device)
    replay = ReplayBuffer(torch)
    decisions = episodes = 0
    status = "failed"
    unresolved_exposure = None
    started = time.monotonic()
    deadline = started + max_seconds
    events_path = directory / "exposures.jsonl"
    events_path.touch(mode=0o600, exist_ok=False)
    transitions_path = directory / "transitions.jsonl"
    transitions_path.touch(mode=0o600, exist_ok=False)
    protocol = {"schemaVersion": 1, "runId": run_id, "config": config,
        "sourceManifestSha256": source_manifest_sha256, "createdAtUtc": utc_now(),
        "operatorDeclaredApproval": {"provider": "AMD Developer Cloud", "device": "MI300X",
            "maxCredits": approved_credits, "billingBasis": "existing AMD account credits only",
            "paymentMethodChargeAuthorized": False, "autoReloadAuthorized": False},
        "trainingDevice": gpu, "physicsDevice": "Original Simulator Node.js process on the same approved server",
        "trainingDriver": "roboproof.motion_learning.gvf_train", "resumeSupported": False,
        "learnedImprovementVerified": False, "independentQualificationPerformed": False,
        "scope": "Bounded experimental GVF residual TD3 fit; no promotion or physical/game-score claim"}
    atomic_json(directory / "run.json", {**protocol, "status": "running", "decisions": 0, "episodes": 0, "updates": 0})
    write_exclusive(directory / "host.json", {"checkedAtUtc": utc_now(), "system": platform.platform(),
        "python": platform.python_version(), "node": __import__("subprocess").check_output(["node", "--version"], text=True).strip(),
        "gpu": gpu, "gpuUsedForOptimizer": True})
    try:
        with MotionEnvironment(control_mode="gvf-residual") as environment:
            identity = environment.contract["identity"]
            initial_actor = export(learner.actor, identity, seed, 0, 0)
            initial_path = directory / "actors/actor-episode-000000.json"
            atomic_actor_json(initial_path, initial_actor)
            initial_sha = digest(initial_path.read_bytes())
            atomic_json(directory / "run.json", {**protocol, "status": "running", "engineIdentity": identity,
                "initialActor": {"path": initial_path.relative_to(directory).as_posix(), "sha256": initial_sha},
                "decisions": decisions, "episodes": episodes, "updates": learner.updates})
            while decisions < max_decisions and learner.updates < max_updates and time.monotonic() < deadline:
                world = training_world(rng, episodes, "mixed-full-reach-v5")
                exposure = {"event": "started", "episode": episodes + 1, "world": world,
                    "actorSha256": initial_sha if learner.updates == 0 else current_actor_sha,
                    "sourceManifestSha256": source_manifest_sha256, "startedAtUtc": utc_now()}
                append_jsonl(events_path, exposure)
                unresolved_exposure = exposure
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
                    unresolved_exposure = {**exposure, "status": "partial-unresolved", "episodeFileSha256": episode_sha}
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
                    directory, identity, source_manifest_sha256, config, episodes, decisions, run_id)
                append_jsonl(events_path, {"event": "completed", "episode": episodes, "world": world,
                    "reason": report["reason"], "decisionCount": episode_decisions,
                    "reward": report["totalReward"], "metrics": report["metrics"],
                    "actorSha256": current_actor_sha, "episodeFileSha256": episode_sha,
                    "sourceManifestSha256": source_manifest_sha256,
                    "checkpointSha256": checkpoint_sha, "completedAtUtc": utc_now()})
                unresolved_exposure = None
                current = {**protocol, "status": "running", "engineIdentity": identity,
                    "decisions": decisions, "episodes": episodes, "updates": learner.updates,
                    "actor": {"path": actor_path.relative_to(directory).as_posix(), "sha256": current_actor_sha},
                    "checkpoint": {"path": checkpoint_path.relative_to(directory).as_posix(), "sha256": checkpoint_sha},
                    "lastEpisode": {"path": episode_path.relative_to(directory).as_posix(), "sha256": episode_sha}}
                atomic_json(directory / "run.json", current)
                print(json.dumps({"episode": episodes, "decisions": decisions, "updates": learner.updates,
                    "reason": report["reason"], "elapsedSeconds": round(time.monotonic() - started, 2)},
                    allow_nan=False), flush=True)
            if status == "failed":
                status = "completed-budget" if decisions >= max_decisions or learner.updates >= max_updates else "time-budget-stopped"
            terminal = {**protocol, "status": status, "engineIdentity": environment.contract["identity"],
                "decisions": decisions, "episodes": episodes, "updates": learner.updates,
                "elapsedSeconds": round(time.monotonic() - started, 3),
                "unresolvedExposure": unresolved_exposure,
                "learnedImprovementVerified": False, "independentQualificationPerformed": False,
                "terminalAtUtc": utc_now()}
            atomic_json(directory / "run.json", terminal)
            write_exclusive(directory / "terminal.json", terminal)
        return terminal
    except BaseException as error:
        failure = {**protocol, "status": "failed", "error": str(error),
            "traceback": traceback.format_exc(), "decisions": decisions, "episodes": episodes,
            "updates": learner.updates, "unresolvedExposure": unresolved_exposure,
            "learnedImprovementVerified": False, "independentQualificationPerformed": False,
            "terminalAtUtc": utc_now()}
        atomic_json(directory / "run.json", failure)
        write_exclusive(directory / "terminal.json", failure)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded remote-only GVF residual TD3 training on original Simulator/iraLIB")
    parser.add_argument("--approved-amd-mi300x", action="store_true", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--approved-max-credits", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=50101)
    parser.add_argument("--max-decisions", type=int, default=10000)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--network-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    options = parser.parse_args(argv)
    result = run(**vars(options))
    print(json.dumps({"status": result["status"], "decisions": result["decisions"],
        "episodes": result["episodes"], "updates": result["updates"],
        "learnedImprovementVerified": False, "independentQualificationPerformed": False}), flush=True)
    return 0 if result["status"] in ("completed-budget", "time-budget-stopped") else 1


if __name__ == "__main__":
    sys.exit(main())
