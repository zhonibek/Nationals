import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid

import torch

from roboproof.motion_env import MotionEnvironment
from .model import MotionActorCritic, export, update


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTORY = ROOT / "roboproof/runs/motion/learning"


def atomic_json(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(filename)


def digest(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def source_identity():
    names = ["roboproof/motion_learning/model.py", "roboproof/motion_learning/train.py",
             "roboproof/motion_env.py", "roboproof/motion-bridge.js",
             "tests/fixtures/motion-evaluation-worlds.json", "tests/fixtures/motion-evaluation-worlds-v2.json"]
    return {name: hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
            for name in names}


def compatible_config(previous, current):
    return (all(previous[key] == current[key] for key in ("rolloutSteps", "seed", "seedCount", "algorithm")) and
            current["updates"] >= previous["updates"] and current["maxSteps"] >= previous["maxSteps"])


def training_world(generator, episode_index):
    stage = 0 if episode_index < 8 else 1 if episode_index < 24 else 2
    start = {"xIn": generator.uniform(-8, 8), "yIn": generator.uniform(-8, 8),
             "headingDeg": generator.uniform(-10, 10) if stage else 0}
    direction = generator.uniform(-math.pi, math.pi) if stage else generator.uniform(-0.4, 0.4)
    distance = generator.uniform(10, 28) if stage else generator.uniform(8, 20)
    goal = {"xIn": start["xIn"] + math.sin(direction) * distance,
            "yIn": start["yIn"] + math.cos(direction) * distance,
            "headingDeg": generator.uniform(-50, 50) if stage else 0}
    configuration = {} if stage < 2 else {"massKg": generator.uniform(7, 12),
                    "muLong": generator.uniform(0.35, 0.85), "kSlip": generator.uniform(350, 850),
                    "batteryInternalR": generator.uniform(0.01, 0.09)}
    world = {"seed": generator.randrange(2**32), "task": {"start": start, "goal": goal,
             "deadlineSeconds": 4 if stage == 0 else 6}, "configuration": configuration, "stage": stage}
    for filename in ("motion-evaluation-worlds.json", "motion-evaluation-worlds-v2.json"):
        for frozen in json.loads((ROOT / "tests/fixtures" / filename).read_text(encoding="utf-8"))["cases"]:
            if world["seed"] == frozen["seed"] or world["task"] == frozen["task"]:
                raise RuntimeError("Frozen evaluation world is excluded from training")
    return world


def train_seed(environment, directory, seed, config, sources, deadline, resume=False, on_checkpoint=None):
    torch.manual_seed(seed)
    generator = random.Random(seed)
    model = MotionActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    history = []
    steps = episodes = updates = 0
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    identity = environment.contract["identity"]
    if resume:
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Bounded episode-boundary checkpoint required")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint["schemaVersion"] != 1 or checkpoint["identity"] != identity or
                checkpoint["sources"] != sources or not compatible_config(checkpoint["config"], config) or checkpoint["seed"] != seed or
                checkpoint["torchVersion"] != str(torch.__version__)):
            raise ValueError("Checkpoint source, environment, configuration or runtime mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.setstate(checkpoint["worldRng"])
        torch.set_rng_state(checkpoint["torchRng"])
        steps, episodes, updates, history = (checkpoint[key] for key in ("steps", "episodes", "updates", "history"))
    elif checkpoint_path.exists():
        raise ValueError("Use an explicit resume, never overwrite a run")
    initial_path = directory / "initial-policy.json"
    if not resume:
        atomic_json(initial_path, export(model, identity, seed, 0, 0))

    def save():
        exported = export(model, identity, seed, steps, updates)
        policy_name = f"policy-{updates:04d}.json"
        atomic_json(directory / policy_name, exported)
        checkpoint = {"schemaVersion": 1, "identity": identity, "sources": sources, "config": config,
                      "seed": seed, "torchVersion": str(torch.__version__), "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "worldRng": generator.getstate(),
                      "torchRng": torch.get_rng_state(), "steps": steps, "episodes": episodes,
                      "updates": updates, "history": history}
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        row = {"seed": seed, "steps": steps, "episodes": episodes, "updates": updates, "trained": updates > 0,
               "policyFile": policy_name, "policySha256": digest(directory / policy_name),
               "initialSha256": digest(initial_path), "actorWeightsChanged": updates > 0 and
               json.loads(initial_path.read_text())["layers"][-1] != exported["layers"][-1], "history": history}
        atomic_json(directory / "summary.json", row)
        if on_checkpoint:
            on_checkpoint(row)
        print(json.dumps({"event": "checkpoint", **{key: row[key] for key in ("seed", "steps", "episodes", "updates")}}), flush=True)
        return row

    summary = save()
    observed_steps = steps
    discarded_steps = 0
    while updates < config["updates"] and observed_steps < config["maxSteps"] and time.monotonic() < deadline:
        saved_world_rng = generator.getstate()
        saved_torch_rng = torch.get_rng_state()
        rows = []
        batch_episodes = []
        interrupted = False
        while len(rows) < config["rolloutSteps"]:
            world = training_world(generator, episodes + len(batch_episodes))
            observation, _ = environment.reset(seed=world["seed"], configuration=world["configuration"], task=world["task"],
                options={"recordTransitions": False, "policyIdentity": {"kind": "untrained", "id": "ppo-training-rollout"}})
            total_reward = 0
            while True:
                if observed_steps >= config["maxSteps"] or time.monotonic() >= deadline:
                    interrupted = True
                    break
                action, unit_action, log_prob, value = model.act(observation)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                observed_steps += 1
                _, _, _, next_value = model.act(next_observation, deterministic=True)
                rows.append({"observation": observation, "action": unit_action, "logProb": float(log_prob),
                             "value": float(value), "nextValue": float(next_value), "reward": reward,
                             "terminated": terminated, "ended": terminated or truncated})
                total_reward += reward
                observation = next_observation
                if terminated or truncated:
                    batch_episodes.append({"seed": world["seed"], "stage": world["stage"],
                                           "reason": info["reason"], "reward": total_reward})
                    break
            if interrupted:
                break
        if interrupted:
            generator.setstate(saved_world_rng)
            torch.set_rng_state(saved_torch_rng)
            discarded_steps = len(rows)
            break
        metrics = update(model, optimizer, rows)
        steps += len(rows)
        episodes += len(batch_episodes)
        updates += 1
        history.append({"update": updates, "steps": steps, "episodes": episodes, **metrics,
                        "meanTrainingReward": sum(row["reward"] for row in batch_episodes) / len(batch_episodes),
                        "trainingSuccessRate": sum(row["reason"] == "success" for row in batch_episodes) / len(batch_episodes),
                        "worlds": batch_episodes})
        summary = save()
    return {**summary, "discardedUncommittedSteps": discarded_steps,
            "budgetReached": updates < config["updates"]}


def run(directory=DEFAULT_DIRECTORY, seed=42, seed_count=3, updates=8, max_steps=4096,
        rollout_steps=128, max_seconds=180, resume=None):
    limits = [(seed, 0, 2**32 - 3), (seed_count, 1, 3), (updates, 1, 100),
              (max_steps, 1, 50000), (rollout_steps, 16, 1024), (max_seconds, 1, 900)]
    if any(isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high for value, low, high in limits):
        raise ValueError("Invalid bounded CPU training budget")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    directory = Path(directory).resolve()
    run_id = resume or str(uuid.uuid4())
    if str(uuid.UUID(run_id)) != run_id:
        raise ValueError("Invalid training run ID")
    run_directory = directory / run_id
    config = {"updates": updates, "maxSteps": max_steps, "rolloutSteps": rollout_steps,
              "seed": seed, "seedCount": seed_count, "algorithm": "ppo-beta-reference-v1"}
    sources = source_identity()
    manifest_path = run_directory / "run.json"
    if resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not compatible_config(previous["config"], config) or previous["sources"] != sources:
            raise ValueError("Resume configuration and source identity must match")
    elif run_directory.exists():
        raise ValueError("Run directory already exists")
    deadline = time.monotonic() + max_seconds
    manifest = {"schemaVersion": 1, "runId": run_id, "config": config, "sources": sources,
                "status": "running", "device": "cpu", "torch": str(torch.__version__),
                "models": [], "learnedImprovementVerified": False,
                "scope": "Canonical reach task only; no obstacle sensing, game-policy training, AMD or hardware execution"}

    def publish():
        atomic_json(manifest_path, manifest)
        atomic_json(directory / "latest.json", {"schemaVersion": 1, "runId": run_id})

    publish()
    try:
        with MotionEnvironment() as environment:
            manifest["identity"] = environment.contract["identity"]
            for offset in range(seed_count):
                model_directory = run_directory / f"seed-{seed + offset}"
                existing = resume and (model_directory / "checkpoint.pt").is_file()
                def checkpoint_saved(row):
                    manifest["models"] = [entry for entry in manifest["models"] if entry["seed"] != row["seed"]] + [row]
                    publish()
                summary = train_seed(environment, model_directory, seed + offset, config, sources, deadline, bool(existing), checkpoint_saved)
                manifest["models"][-1] = summary
                publish()
        manifest["status"] = "completed" if all(not row["budgetReached"] for row in manifest["models"]) else "budget-stopped"
        publish()
    except BaseException:
        manifest["status"] = "interrupted-or-failed"
        publish()
        raise
    print(json.dumps({"event": "finished", "runId": run_id, "status": manifest["status"],
                      "trainedModels": sum(row["trained"] for row in manifest["models"])}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Bounded PPO through the original Simulator and iraLIB; CPU only")
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--resume")
    arguments = parser.parse_args()
    run(**vars(arguments))


if __name__ == "__main__":
    main()
