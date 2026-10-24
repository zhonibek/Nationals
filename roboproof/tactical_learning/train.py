from __future__ import annotations

import argparse
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

    def close(self):
        if self.process.poll() is None:
            try:
                self.request({"operation": "close"})
            finally:
                self.process.wait(timeout=10)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_runtime(approved_free_t4: bool):
    if not approved_free_t4:
        raise RuntimeError("Explicit approved-free-t4 opt-in is required")
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
    parser.add_argument("--approved-free-t4", action="store_true")
    parser.add_argument("--seed", type=int, default=731001)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--evaluation-episodes", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "roboproof/runs/tactical/t4-pilot")
    args = parser.parse_args()
    if not 0 <= args.seed <= 0xFFFFFFFF or not 1 <= args.episodes <= 10000 or not 1 <= args.max_seconds <= 600:
        parser.error("seed, episodes or bounded runtime is outside the allowed range")
    if not 1 <= args.evaluation_episodes <= 100:
        parser.error("evaluation-episodes must be between 1 and 100")
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


def update(model, optimizer, trajectories, device):
    rows = []
    for trajectory in trajectories:
        discounted_return = 0.0
        for transition in reversed(trajectory):
            discounted_return = transition["reward"] + (0.995 ** transition["durationSeconds"]) * discounted_return
            rows.append({"features": transition["features"], "choice": transition["choice"], "return": discounted_return})
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
                        "episodeReturn": total_reward, "done": observation["done"]}


def atomic_torch_save(value, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as saved:
        os.fsync(saved.fileno())
    os.replace(temporary, path)


def main():
    args = parse_args()
    runtime = validate_runtime(args.approved_free_t4)
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
           "movementAbstraction": "macro-action teleport plus fixed nominal travel-time proxy; no physical motion or path feasibility",
           "trainingPerformed": True, "learnedImprovementVerified": False,
           "independentQualificationPerformed": False, "promotionAllowed": False,
           "warning": "This pilot can train game-score priorities in an abstract macro-action model only. It is not a physical tactics or scoring qualification."}
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    environment = NodeTacticalEnvironment()
    started = time.monotonic()
    trajectories = []
    metrics = []
    completed = 0
    status = "interrupted"
    try:
        with (args.output / "episodes.jsonl").open("x", encoding="utf-8") as episode_log, \
                (args.output / "exposure.jsonl").open("x", encoding="utf-8") as exposure_log:
            for episode in range(args.episodes):
                if time.monotonic() - started >= args.max_seconds:
                    break
                episode_seed = (args.seed + episode) & 0xFFFFFFFF
                alliance = "red" if episode % 2 == 0 else "blue"
                trajectory, result = run_episode(environment, model, episode_seed, alliance, device)
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
                                  "learnedImprovementVerified": False, "independentQualificationPerformed": False}
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
                              "learnedImprovementVerified": False, "independentQualificationPerformed": False}
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
        (args.output / "evaluation.json").write_text(json.dumps(evaluation_record, indent=2) + "\n", encoding="utf-8")
        run.update({"status": status, "episodesCompleted": completed, "elapsedSeconds": round(time.monotonic() - started, 3),
                    "evaluationEpisodesCompleted": len(evaluation), "learnedImprovementVerified": False,
                    "independentQualificationPerformed": False})
    finally:
        environment.close()
        if completed < args.episodes and status == "interrupted":
            status = "interrupted"
        run.update({"status": status, "episodesCompleted": completed,
                    "elapsedSeconds": round(time.monotonic() - started, 3),
                    "learnedImprovementVerified": False, "independentQualificationPerformed": False})
        (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "finished", "runId": run_id, "status": status, "episodesCompleted": completed,
                      "elapsedSeconds": round(time.monotonic() - started, 3),
                      "output": str(args.output), "learnedImprovementVerified": False,
                      "independentQualificationPerformed": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
