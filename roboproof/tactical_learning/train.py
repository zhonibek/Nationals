from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch
from torch.distributions import Categorical

from .model import FEATURE_COUNT, MODEL_ID, TacticalActorCritic


ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "roboproof/tactical_learning/bridge.js"
SOURCE_FILES = ("roboproof/tactical_environment.js", "roboproof/tactical_baselines.js", "roboproof/tactical_learning/bridge.js",
                "roboproof/tactical_learning/model.py", "roboproof/tactical_learning/train.py",
                "simulator/override.js", "simulator/override-geometry.js", "simulator/override-dynamics.js")
# Paid Nebius AI Cloud Serverless AI job GPUs the user approved separately from the free T4:
# torch device-name fragment -> exact CUDA capability and memory floor of the real device.
APPROVED_NEBIUS_GPUS = {"L40S": {"capability": (8, 9), "minimumMemoryBytes": 40 * 1024 ** 3}}
MAX_APPROVED_NEBIUS_COST_USD = 10.0
# Optional behavior-cloning warm start (off unless --warm-start-demonstration-episodes is given): the actor imitates
# the non-learning team tactic and the critic fits its returns, then the unchanged RL loop runs from those weights.
WARM_START_POLICY = "loader-cycle"
WARM_START_COLLECTION_SHARE = 0.4  # no demonstration episode starts after this share of --max-seconds
WARM_START_SHARE = 0.5  # pretraining stops after this share; the rest of the budget is for RL and evaluation
WARM_START_DEFAULT_EPOCHS = 200
WARM_START_BATCH_SIZE = 256
WARM_START_LEARNING_RATE = 1e-3
WARM_START_CRITIC_WEIGHT = 0.5  # the RL loss weighs its critic term the same way
# With a warm start, RL stops while this many times the slowest RL episode per evaluation episode is still left.
EVALUATION_RESERVE_FACTOR = 2.0


class NodeTacticalEnvironment:
    def __init__(self):
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Node.js is required to run the original Override game engine")
        self.process = subprocess.Popen([node, str(BRIDGE)], cwd=ROOT, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                        encoding="utf-8", bufsize=1)

    def request(self, payload):
        if self.process.poll() is not None:
            raise RuntimeError("Original game-engine worker exited unexpectedly")
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read()[-4000:]
            raise RuntimeError("Original game-engine worker returned no response: " + error)
        response = json.loads(line)
        if response.get("ok") is not True:
            raise RuntimeError("Original game-engine request failed: " + str(response.get("error", "unknown error")))
        return response

    def reset(self, seed, alliance):
        return self.request({"operation": "reset", "seed": int(seed), "alliance": alliance})["observation"]

    def step(self, action_id):
        return self.request({"operation": "step", "actionId": action_id})

    def baselines(self, seeds, alliance):
        return self.request({"operation": "baselines", "seeds": [int(seed) for seed in seeds], "alliance": alliance})["baselines"]

    def demonstrate(self, policy=WARM_START_POLICY):
        """The non-learning tactic's candidate ID for the current state, or None when it makes no choice."""
        return self.request({"operation": "demonstrate", "policy": policy})["actionId"]

    def close(self, timeout=10):
        """Shut the bridge down within bounded time and return what happened; never raises on a stuck bridge."""
        process = self.process
        shutdown = {"method": "already-exited", "closeAcknowledged": False, "timeoutSeconds": timeout}
        if process.poll() is None:
            try:
                process.stdin.write(json.dumps({"operation": "close"}, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except (OSError, ValueError) as error:
                shutdown["closeRequestError"] = f"{type(error).__name__}: {error}"[:500]
            # EOF on stdin also ends the bridge's read loop, so a bridge that ignores the request still exits.
            with contextlib.suppress(OSError, ValueError):
                process.stdin.close()
            for method, stop in (("exited", None), ("terminated", process.terminate), ("killed", process.kill)):
                if stop is not None:
                    with contextlib.suppress(OSError):
                        stop()
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue
                shutdown["method"] = method
                break
            else:
                shutdown["method"] = "kill-unconfirmed"
        shutdown["returnCode"] = process.returncode
        if process.returncode is not None:
            # The bridge has exited, so reading its remaining output cannot block.
            with contextlib.suppress(OSError, ValueError):
                shutdown["closeAcknowledged"] = any(line.strip() == '{"ok":true}' for line in process.stdout.read().splitlines())
            for stream in (process.stdin, process.stdout, process.stderr):
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
        return shutdown


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_nebius_runtime(approved_nebius_gpu: str, approved_max_cost_usd):
    expected = APPROVED_NEBIUS_GPUS.get(approved_nebius_gpu)
    if expected is None:
        raise RuntimeError("Unapproved Nebius GPU type; approved: " + ", ".join(sorted(APPROVED_NEBIUS_GPUS)))
    if (isinstance(approved_max_cost_usd, bool) or not isinstance(approved_max_cost_usd, (int, float))
            or not math.isfinite(approved_max_cost_usd)
            or not 0 < approved_max_cost_usd <= MAX_APPROVED_NEBIUS_COST_USD):
        raise RuntimeError(f"A Nebius job needs an explicit approved maximum cost above 0 and at most "
                           f"{MAX_APPROVED_NEBIUS_COST_USD:g} USD")
    if platform.system() != "Linux" or torch.version.cuda is None or torch.version.hip is not None:
        raise RuntimeError("Tactical training is restricted to a remote Linux CUDA runtime; no local or ROCm fallback")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible NVIDIA CUDA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if (approved_nebius_gpu not in properties.name.upper()
            or (properties.major, properties.minor) != expected["capability"]
            or properties.total_memory < expected["minimumMemoryBytes"]):
        major, minor = expected["capability"]
        raise RuntimeError(f"This approved Nebius job is pinned to an actual NVIDIA {approved_nebius_gpu} "
                           f"(CUDA capability {major}.{minor})")
    if shutil.which("node") is None:
        raise RuntimeError("Original Simulator game engine requires Node.js")
    # The approved cost is the user's declared ceiling; the trainer cannot see provider billing.
    return {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda), "gpu": properties.name, "capability": [properties.major, properties.minor],
            "memoryBytes": properties.total_memory, "node": subprocess.run(["node", "--version"], check=True,
                capture_output=True, text=True).stdout.strip(), "device": "cuda:0",
            "provider": "nebius-serverless-ai-job", "approvedGpu": approved_nebius_gpu,
            "approvedMaxCostUsd": float(approved_max_cost_usd), "providerBillingVerified": False}


def validate_runtime(approved_free_t4: bool, approved_nebius_gpu: str | None = None, approved_max_cost_usd=None):
    if approved_free_t4 and approved_nebius_gpu is not None:
        raise RuntimeError("Choose exactly one approved runtime: approved-free-t4 or approved-nebius-gpu")
    if approved_nebius_gpu is not None:
        return validate_nebius_runtime(approved_nebius_gpu, approved_max_cost_usd)
    if approved_max_cost_usd is not None:
        raise RuntimeError("approved-max-cost-usd applies only to an approved Nebius GPU job")
    if not approved_free_t4:
        raise RuntimeError("Explicit approved-free-t4 or approved-nebius-gpu opt-in is required")
    if platform.system() != "Linux" or torch.version.cuda is None or torch.version.hip is not None:
        raise RuntimeError("Tactical training is restricted to a remote Linux CUDA runtime; no local or ROCm fallback")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible NVIDIA CUDA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if "T4" not in properties.name.upper() or (properties.major, properties.minor) != (7, 5):
        raise RuntimeError("This approved tactical pilot is pinned to an actual Tesla T4 (CUDA capability 7.5)")
    if shutil.which("node") is None:
        raise RuntimeError("Original Simulator game engine requires Node.js")
    return {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda), "gpu": properties.name, "capability": [properties.major, properties.minor],
            "memoryBytes": properties.total_memory, "node": subprocess.run(["node", "--version"], check=True,
                capture_output=True, text=True).stdout.strip(), "device": "cuda:0"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a score-driven tactical policy on the original Override rules engine")
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument("--approved-free-t4", action="store_true")
    approval.add_argument("--approved-nebius-gpu", choices=sorted(APPROVED_NEBIUS_GPUS))
    parser.add_argument("--approved-max-cost-usd", type=float)
    parser.add_argument("--seed", type=int, default=731001)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--evaluation-episodes", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "roboproof/runs/tactical/t4-pilot")
    parser.add_argument("--warm-start-demonstration-episodes", type=int, default=0,
                        help=f"behavior-clone the {WARM_START_POLICY} tactic from this many episodes before RL (0 = off)")
    parser.add_argument("--warm-start-epochs", type=int,
                        help=f"pretraining epochs over the demonstrations (default {WARM_START_DEFAULT_EPOCHS})")
    args = parser.parse_args()
    if not 0 <= args.seed <= 0xFFFFFFFF or not 1 <= args.episodes <= 10000 or not 1 <= args.max_seconds <= 600:
        parser.error("seed, episodes or bounded runtime is outside the allowed range")
    if not 1 <= args.evaluation_episodes <= 100:
        parser.error("evaluation-episodes must be between 1 and 100")
    if not 0 <= args.warm_start_demonstration_episodes <= 200:
        parser.error("warm-start-demonstration-episodes must be between 0 and 200")
    if args.warm_start_epochs is not None:
        if not args.warm_start_demonstration_episodes:
            parser.error("warm-start-epochs needs warm-start-demonstration-episodes")
        if not 1 <= args.warm_start_epochs <= 2000:
            parser.error("warm-start-epochs must be between 1 and 2000")
    elif args.warm_start_demonstration_episodes:
        args.warm_start_epochs = WARM_START_DEFAULT_EPOCHS
    return args


def packed_batch(rows, device):
    maximum = max(len(row["features"]) for row in rows)
    features = torch.zeros((len(rows), maximum, FEATURE_COUNT), dtype=torch.float32, device=device)
    mask = torch.zeros((len(rows), maximum), dtype=torch.bool, device=device)
    choices = torch.empty(len(rows), dtype=torch.long, device=device)
    returns = torch.empty(len(rows), dtype=torch.float32, device=device)
    for index, row in enumerate(rows):
        count = len(row["features"])
        features[index, :count] = torch.tensor(row["features"], dtype=torch.float32, device=device)
        mask[index, :count] = True
        choices[index] = row["choice"]
        returns[index] = row["return"]
    return features, mask, choices, returns


def discounted_rows(trajectories):
    rows = []
    for trajectory in trajectories:
        discounted_return = 0.0
        for transition in reversed(trajectory):
            discounted_return = transition["reward"] + (0.995 ** transition["durationSeconds"]) * discounted_return
            rows.append({"features": transition["features"], "choice": transition["choice"], "return": discounted_return})
    return rows


def update(model, optimizer, trajectories, device):
    rows = discounted_rows(trajectories)
    if not rows:
        return {"loss": 0.0, "samples": 0}
    features, mask, choices, returns = packed_batch(rows, device)
    model.train()
    logits, values = model(features, mask)
    log_probabilities = torch.log_softmax(logits, dim=-1).gather(1, choices.unsqueeze(1)).squeeze(1)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    advantages = returns - values.detach()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    actor_loss = -(log_probabilities * advantages).mean()
    critic_loss = torch.nn.functional.smooth_l1_loss(values, returns)
    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(loss.detach().cpu()), "samples": len(rows)}


def run_episode(environment, model, seed, alliance, device, deterministic=False):
    observation = environment.reset(seed, alliance)
    trajectory = []
    total_reward = 0.0
    while not observation["done"]:
        actions = observation["actions"]
        if not actions:
            break
        feature_rows = [action["features"] for action in actions]
        feature_tensor = torch.tensor(feature_rows, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.ones((1, len(actions)), dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, _ = model(feature_tensor, mask)
            if deterministic:
                choice = int(logits.argmax(dim=-1).item())
            else:
                choice = int(Categorical(logits=logits).sample().item())
        result = environment.step(actions[choice]["id"])
        trajectory.append({"features": feature_rows, "choice": choice,
                           "reward": float(result["reward"]), "durationSeconds": float(result["durationSeconds"])})
        total_reward += float(result["reward"])
        observation = result["observation"]
        if len(trajectory) >= 100:
            break
    return trajectory, {"seed": seed, "alliance": alliance, "score": observation["score"],
                        "clock": observation["clock"], "decisions": observation["decisions"],
                        "episodeReturn": total_reward, "done": observation["done"],
                        "timing": observation.get("timing")}


def run_demonstration_episode(environment, seed, alliance, policy=WARM_START_POLICY):
    """Play one episode with the non-learning tactic's choices, recording transitions the way run_episode does."""
    observation = environment.reset(seed, alliance)
    trajectory = []
    total_reward = 0.0
    rejected = 0
    stop_reason = "episode-complete"
    while not observation["done"]:
        actions = observation["actions"]
        if not actions:
            break
        action_id = environment.demonstrate(policy)
        if action_id is None:
            stop_reason = "no-choice"
            break
        choice = next((index for index, action in enumerate(actions) if action["id"] == action_id), None)
        if choice is None:
            raise RuntimeError("The demonstrated action is not in the observed candidate set")
        result = environment.step(action_id)
        rejected += bool(result.get("rejectedAction"))
        trajectory.append({"features": [action["features"] for action in actions], "choice": choice,
                           "reward": float(result["reward"]), "durationSeconds": float(result["durationSeconds"])})
        total_reward += float(result["reward"])
        observation = result["observation"]
        if len(trajectory) >= 100:
            stop_reason = "decision-cap"
            break
    return trajectory, {"seed": seed, "alliance": alliance, "policy": policy, "score": observation["score"],
                        "clock": observation["clock"], "decisions": observation["decisions"],
                        "episodeReturn": total_reward, "done": observation["done"], "timing": observation.get("timing"),
                        "stopReason": stop_reason, "rejectedActions": rejected}


def behavior_cloning_losses(model, features, mask, choices, returns):
    logits, values = model(features, mask)
    actor_loss = torch.nn.functional.cross_entropy(logits, choices)
    critic_loss = torch.nn.functional.smooth_l1_loss(values, returns)
    accuracy = (logits.argmax(dim=-1) == choices).float().mean()
    return actor_loss, critic_loss, accuracy


def pretrain(model, trajectories, device, epochs, deadline, seed):
    """Behavior cloning: actor cross-entropy on the demonstrated candidate, critic on the demonstrated discounted
    returns. It has its own optimizer and shuffle generator, so the RL optimizer and the global RNGs start untouched."""
    rows = discounted_rows(trajectories)
    features, mask, choices, returns = packed_batch(rows, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=WARM_START_LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed)

    def measure():
        with torch.no_grad():
            actor_loss, critic_loss, accuracy = behavior_cloning_losses(model, features, mask, choices, returns)
        return {"actorLoss": float(actor_loss.cpu()), "criticLoss": float(critic_loss.cpu()), "accuracy": float(accuracy.cpu())}

    model.train()
    initial = measure()
    history = []
    stop_reason = "all-requested-epochs"
    for epoch in range(epochs):
        if time.monotonic() >= deadline:
            stop_reason = "time-budget"
            break
        order = torch.randperm(len(rows), generator=generator).to(device)
        totals = {"loss": 0.0, "actorLoss": 0.0, "criticLoss": 0.0}
        batches = 0
        for start in range(0, len(rows), WARM_START_BATCH_SIZE):
            index = order[start:start + WARM_START_BATCH_SIZE]
            actor_loss, critic_loss, _ = behavior_cloning_losses(model, features[index], mask[index], choices[index], returns[index])
            loss = actor_loss + WARM_START_CRITIC_WEIGHT * critic_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in (("loss", loss), ("actorLoss", actor_loss), ("criticLoss", critic_loss)):
                totals[name] += float(value.detach().cpu())
            batches += 1
        history.append({"epoch": epoch + 1, **{name: total / batches for name, total in totals.items()}})
    return {"samples": len(rows), "epochsCompleted": len(history), "stopReason": stop_reason,
            "initial": initial, "final": measure(), "history": history}


def demonstration_seed(seed, episodes, index):
    # Demonstrations get the block right after the predeclared training seeds (seed + episode for episode < episodes).
    return (seed + episodes + index) & 0xFFFFFFFF


def warm_start_plan(args):
    count = args.warm_start_demonstration_episodes
    return {"status": "planned", "policy": WARM_START_POLICY,
            "method": "behavior cloning on the demonstrated choices (actor cross-entropy, critic smooth-L1 on the "
                      "demonstrated discounted returns), then the unchanged RL loop from the pretrained weights",
            "demonstrationEpisodesRequested": count,
            "predeclaredDemonstrationSeeds": {"first": demonstration_seed(args.seed, args.episodes, 0),
                                              "last": demonstration_seed(args.seed, args.episodes, count - 1),
                                              "rule": "(seed + episodes + i) mod 2^32 for i < count; red for even i, blue for odd i"},
            "epochsRequested": args.warm_start_epochs, "batchSize": WARM_START_BATCH_SIZE,
            "learningRate": WARM_START_LEARNING_RATE, "criticWeight": WARM_START_CRITIC_WEIGHT,
            "optimizer": "separate Adam; the RL optimizer starts fresh after the warm start",
            "collectionDeadlineSeconds": WARM_START_COLLECTION_SHARE * args.max_seconds,
            "deadlineSeconds": WARM_START_SHARE * args.max_seconds,
            "evaluationReserveFactor": EVALUATION_RESERVE_FACTOR}


def append_jsonl(stream, value):
    stream.write(json.dumps(value, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run_warm_start(environment, model, device, args, run_id, started, exposure_log, record):
    """Collect demonstration episodes (logged as exposure) and pretrain on them; updates `record` in place."""
    record["status"] = "running"
    record["collectionStopReason"] = "all-requested-episodes"
    trajectories, seeds, margins = [], [], []
    with (args.output / "warm-start.jsonl").open("x", encoding="utf-8") as log:
        for index in range(args.warm_start_demonstration_episodes):
            if time.monotonic() - started >= record["collectionDeadlineSeconds"]:
                record["collectionStopReason"] = "time-budget"
                break
            seed = demonstration_seed(args.seed, args.episodes, index)
            alliance = "red" if index % 2 == 0 else "blue"
            trajectory, result = run_demonstration_episode(environment, seed, alliance)
            result["demonstrationEpisode"] = index + 1
            append_jsonl(log, {"event": "demonstration", **result})
            append_jsonl(exposure_log, {"phase": "demonstration", "demonstrationEpisode": index + 1, "seed": seed,
                                        "alliance": alliance, "decisions": result["decisions"],
                                        "policy": WARM_START_POLICY, "runId": run_id})
            trajectories.append(trajectory)
            seeds.append(seed)
            margins.append(result["score"]["margin"])
        record.update({"demonstrationEpisodesCompleted": len(seeds), "demonstrationSeeds": seeds,
                       "demonstrationSamples": sum(len(trajectory) for trajectory in trajectories),
                       "demonstrationMeanScoreMargin": sum(margins) / len(margins) if margins else None,
                       "pretraining": None})
        if record["demonstrationSamples"]:
            record["status"] = "pretraining"
            pretraining = pretrain(model, trajectories, device, args.warm_start_epochs,
                                   started + record["deadlineSeconds"], args.seed)
            for row in pretraining.pop("history", []):
                append_jsonl(log, {"event": "pretrain-epoch", **row})
            record["pretraining"] = pretraining
    record.update({"status": "completed" if record["pretraining"] else "skipped-no-demonstrations",
                   "elapsedSeconds": round(time.monotonic() - started, 3),
                   "reinforcementStoppedForEvaluationReserve": False})


def write_json_atomic(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch_save(value, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as saved:
        os.fsync(saved.fileno())
    os.replace(temporary, path)


def main():
    args = parse_args()
    runtime = validate_runtime(args.approved_free_t4, args.approved_nebius_gpu, args.approved_max_cost_usd)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refusing to overwrite a previous tactical training run")
    args.output.mkdir(parents=True)
    run_id = str(uuid.uuid4())
    device = torch.device("cuda:0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = TacticalActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    source_hashes = {filename: sha256_file(ROOT / filename) for filename in SOURCE_FILES}
    run = {"schemaVersion": 1, "runId": run_id, "model": MODEL_ID, "status": "running",
           "seed": args.seed, "episodesRequested": args.episodes, "maxSeconds": args.max_seconds,
           "evaluationEpisodes": args.evaluation_episodes, "featureCount": FEATURE_COUNT,
           "width": model.width, "hiddenLayers": 2, "runtime": runtime, "sourceSha256": source_hashes,
           "reward": "change in exact Override simulator score differential; no LLM-generated reward",
           "movementAbstraction": "macro-action teleport plus a travel-time proxy; no physical motion or path feasibility",
           "tacticalTiming": "environment default profile, recorded per episode in episodes.jsonl ('measured' = calibrated to the team robot cycles: loader ~2 s, field ~4 s)",
           "trainingPerformed": True, "learnedImprovementVerified": False,
           "independentQualificationPerformed": False, "promotionAllowed": False,
           "warning": "This pilot can train game-score priorities in an abstract macro-action model only. It is not a physical tactics or scoring qualification."}
    warm_start = warm_start_plan(args) if args.warm_start_demonstration_episodes else None
    if warm_start is not None:
        run["warmStart"] = warm_start
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    environment = NodeTacticalEnvironment()
    started = time.monotonic()
    trajectories = []
    metrics = []
    completed = 0
    status = "interrupted"
    slowest_episode_seconds = 0.0
    try:
        with (args.output / "episodes.jsonl").open("x", encoding="utf-8") as episode_log, \
                (args.output / "exposure.jsonl").open("x", encoding="utf-8") as exposure_log:
            if warm_start is not None:
                run_warm_start(environment, model, device, args, run_id, started, exposure_log, warm_start)
                if warm_start["pretraining"]:
                    atomic_torch_save({"schemaVersion": 1, "runId": run_id, "model": MODEL_ID,
                                       "stage": "behavior-cloning-warm-start", "modelState": model.state_dict(),
                                       "trainingSeed": args.seed, "sourceSha256": source_hashes, "runtime": runtime,
                                       "warmStart": warm_start, "learnedImprovementVerified": False,
                                       "independentQualificationPerformed": False}, args.output / "warm-start.pt")
                    warm_start["checkpoint"] = "warm-start.pt"
                write_json_atomic(args.output / "run.json", run)
                print(json.dumps({"event": "warm-start", "status": warm_start["status"],
                                  "demonstrationEpisodes": warm_start["demonstrationEpisodesCompleted"],
                                  "samples": warm_start["demonstrationSamples"],
                                  "demonstrationMeanScoreMargin": warm_start["demonstrationMeanScoreMargin"],
                                  "final": (warm_start["pretraining"] or {}).get("final"),
                                  "elapsedSeconds": warm_start["elapsedSeconds"]}, sort_keys=True), flush=True)
            for episode in range(args.episodes):
                if time.monotonic() - started >= args.max_seconds:
                    break
                # The warm start spent part of the shared budget, so RL stops early enough for the evaluation.
                if warm_start is not None and (time.monotonic() - started + EVALUATION_RESERVE_FACTOR
                                               * args.evaluation_episodes * slowest_episode_seconds >= args.max_seconds):
                    warm_start["reinforcementStoppedForEvaluationReserve"] = True
                    break
                episode_seed = (args.seed + episode) & 0xFFFFFFFF
                alliance = "red" if episode % 2 == 0 else "blue"
                episode_started = time.monotonic()
                trajectory, result = run_episode(environment, model, episode_seed, alliance, device)
                slowest_episode_seconds = max(slowest_episode_seconds, time.monotonic() - episode_started)
                trajectories.append(trajectory)
                result["episode"] = episode + 1
                metrics.append(result)
                episode_log.write(json.dumps(result, sort_keys=True) + "\n")
                episode_log.flush()
                os.fsync(episode_log.fileno())
                exposure_log.write(json.dumps({"episode": episode + 1, "seed": episode_seed, "alliance": alliance,
                                               "decisions": result["decisions"], "runId": run_id}) + "\n")
                exposure_log.flush()
                os.fsync(exposure_log.fileno())
                completed += 1
                if completed % 8 == 0:
                    update_metrics = update(model, optimizer, trajectories, device)
                    trajectories.clear()
                    checkpoint = {"schemaVersion": 1, "runId": run_id, "model": MODEL_ID,
                                  "modelState": model.state_dict(), "optimizerState": optimizer.state_dict(),
                                  "episodesCompleted": completed, "trainingSeed": args.seed,
                                  "sourceSha256": source_hashes, "runtime": runtime,
                                  "learnedImprovementVerified": False, "independentQualificationPerformed": False,
                                  **({"warmStart": warm_start} if warm_start is not None else {})}
                    atomic_torch_save(checkpoint, args.output / "latest.pt")
                    metrics_row = {"event": "checkpoint", "episodes": completed, **update_metrics,
                                   "scoreMargin": result["score"]["margin"]}
                    with (args.output / "training.jsonl").open("a", encoding="utf-8") as training_log:
                        training_log.write(json.dumps(metrics_row, sort_keys=True) + "\n")
                        training_log.flush()
                        os.fsync(training_log.fileno())
                    print(json.dumps(metrics_row, sort_keys=True), flush=True)
            if trajectories:
                update_metrics = update(model, optimizer, trajectories, device)
                checkpoint = {"schemaVersion": 1, "runId": run_id, "model": MODEL_ID,
                              "modelState": model.state_dict(), "optimizerState": optimizer.state_dict(),
                              "episodesCompleted": completed, "trainingSeed": args.seed,
                              "sourceSha256": source_hashes, "runtime": runtime,
                              "learnedImprovementVerified": False, "independentQualificationPerformed": False,
                              **({"warmStart": warm_start} if warm_start is not None else {})}
                atomic_torch_save(checkpoint, args.output / "latest.pt")
                with (args.output / "training.jsonl").open("a", encoding="utf-8") as training_log:
                    training_log.write(json.dumps({"event": "checkpoint", "episodes": completed, **update_metrics}, sort_keys=True) + "\n")
            if completed == args.episodes:
                status = "completed"
            else:
                status = "time-budget-exhausted"
        evaluation = []
        model.eval()
        for offset in range(args.evaluation_episodes):
            if time.monotonic() - started >= args.max_seconds:
                break
            seed = (args.seed ^ 0x5BD1E995 ^ offset) & 0xFFFFFFFF
            _, result = run_episode(environment, model, seed, "red" if offset % 2 == 0 else "blue", device, deterministic=True)
            evaluation.append(result)
        # Same held-out seeds per alliance for the non-learning tactics (loader-cycle is the team tactic).
        baselines = {}
        for alliance in ("red", "blue"):
            seeds = [row["seed"] for row in evaluation if row["alliance"] == alliance]
            if seeds:
                baselines[alliance] = environment.baselines(seeds, alliance)
        learned_mean = (sum(row["score"]["margin"] for row in evaluation) / len(evaluation)) if evaluation else None
        baseline_means = {}
        for name in ("loader-cycle", "field-greedy", "greedy-all", "random"):
            margins = [margin for alliance in baselines.values() for margin in alliance[name]["margins"]]
            baseline_means[name] = sum(margins) / len(margins) if margins else None
        evaluation_record = {"schemaVersion": 2, "runId": run_id, "episodes": evaluation,
                             "meanScoreMargin": learned_mean, "baselineMeanScoreMargin": baseline_means,
                             "baselines": baselines,
                             "beatsLoaderCycleOnHoldout": (learned_mean is not None and baseline_means["loader-cycle"] is not None
                                                           and learned_mean > baseline_means["loader-cycle"]),
                             "scope": "randomized start-pose holdout in the abstract macro-action model only",
                             "learnedImprovementVerified": False, "independentQualificationPerformed": False}
        if warm_start is not None:
            # The learned policy started from imitating loader-cycle, so it is not a from-scratch comparison.
            evaluation_record["warmStart"] = {"policy": WARM_START_POLICY,
                                              "demonstrationEpisodes": warm_start.get("demonstrationEpisodesCompleted", 0)}
            warm_start["evaluationReserveSeconds"] = round(
                EVALUATION_RESERVE_FACTOR * args.evaluation_episodes * slowest_episode_seconds, 3)
        (args.output / "evaluation.json").write_text(json.dumps(evaluation_record, indent=2) + "\n", encoding="utf-8")
        run.update({"status": status, "episodesCompleted": completed, "elapsedSeconds": round(time.monotonic() - started, 3),
                    "evaluationEpisodesCompleted": len(evaluation), "learnedImprovementVerified": False,
                    "independentQualificationPerformed": False})
    finally:
        run.update({"status": status, "episodesCompleted": completed,
                    "elapsedSeconds": round(time.monotonic() - started, 3),
                    "learnedImprovementVerified": False, "independentQualificationPerformed": False})
        # The final record lands before the bridge is shut down, so a stuck bridge cannot leave it "running".
        try:
            write_json_atomic(args.output / "run.json", run)
        finally:
            run["bridgeShutdown"] = environment.close()
        write_json_atomic(args.output / "run.json", run)
    print(json.dumps({"event": "finished", "runId": run_id, "status": status, "episodesCompleted": completed,
                      "elapsedSeconds": run["elapsedSeconds"], "bridgeShutdown": run["bridgeShutdown"]["method"],
                      "output": str(args.output), "learnedImprovementVerified": False,
                      "independentQualificationPerformed": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
