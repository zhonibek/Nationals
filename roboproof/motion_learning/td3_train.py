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
from .imitation import MeasuredAnchor, collect_demonstration, demonstration_samples, imitation_update
from .model import MotionActorCritic, FEATURE_TRANSFORMS, DEADLINE_FEATURES, DEADLINE_CONTEXT, normalization_definition, public_deadline_floor, export
from .objective import LearningObjective, SETTLING_MARGIN_OBJECTIVE, PROFILES, objective_definition
from .td3 import ALGORITHM, ReplayBuffer, ReservoirReplayBuffer, TD3Learner, TrainingExploration, complete_episode_returns
from .train import ROOT, CURRICULA, atomic_json, digest, source_identity as ppo_source_identity, training_world, exclude_frozen_world, curriculum_definition


DEFAULT_DIRECTORY = ROOT / "roboproof/runs/motion/td3"


def source_identity():
    result = ppo_source_identity()
    for name in ("roboproof/motion_learning/td3.py", "roboproof/motion_learning/td3_train.py"):
        result[name] = hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
    return result


def compatible_config(previous, current):
    return (all(previous.get(name) == value for name, value in current.items() if name not in ("updates", "maxSteps")) and
            previous.keys() == current.keys() and current["updates"] >= previous["updates"] and current["maxSteps"] >= previous["maxSteps"])


def configuration(seed, seed_count, updates, max_steps, rollout_steps, warm_start_updates, feature_transform,
                  objective, curriculum, gradient_steps, replay_capacity, exploration_std,
                  demonstration_anchor_weight, policy_regularization, deadline_pace_loss_weight, n_steps=1,
                  critic_terminal_fraction=0, replay_strategy="recent-ring-v1", exploration_profile="iid-v1",
                  pose_exploration_std=None, exploration_correlation_seconds=0):
    limits = ((seed, 0, 2**32 - 3), (seed_count, 1, 3), (updates, 1, 100), (max_steps, 1, 50000),
              (rollout_steps, 16, 1024), (warm_start_updates, 0, 16), (gradient_steps, 1, 32), (replay_capacity, 64, 2048), (n_steps, 1, 32))
    if any(isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper for value, lower, upper in limits):
        raise ValueError("Invalid bounded TD3 CPU budget")
    if (feature_transform not in FEATURE_TRANSFORMS or objective not in PROFILES or curriculum not in CURRICULA or
            policy_regularization not in ("none", "public-deadline-residual-v1") or warm_start_updates >= updates or
            replay_strategy not in ("recent-ring-v1", "uniform-reservoir-v1")):
        raise ValueError("Unsupported TD3 configuration")
    for value, lower, upper in ((exploration_std, 0.001, 0.2), (demonstration_anchor_weight, 0, 100),
                                (deadline_pace_loss_weight, 0, 100), (critic_terminal_fraction, 0, 0.5)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError("Finite bounded TD3 noise/retention/prior required")
    if (demonstration_anchor_weight and not warm_start_updates) or (policy_regularization == "none" and deadline_pace_loss_weight != 1):
        raise ValueError("Active teacher anchor and declared prior required")
    exploration = TrainingExploration(exploration_std, exploration_profile, pose_exploration_std, exploration_correlation_seconds)
    return {"seed": seed, "seedCount": seed_count, "updates": updates, "maxSteps": max_steps, "rolloutSteps": rollout_steps,
            "algorithm": ALGORITHM, "initialization": "broad-pace-v1", "featureTransform": feature_transform,
            "normalizationDefinition": normalization_definition(feature_transform),
            "deadlineContextDefinition": dict(DEADLINE_CONTEXT) if feature_transform in DEADLINE_FEATURES else None,
            "objective": objective, "objectiveDefinition": objective_definition(objective), "curriculum": curriculum,
            "curriculumDefinition": curriculum_definition(curriculum), "warmStartUpdates": warm_start_updates,
            "teacherSelection": "pose-deadline-margin-v2", "demonstrationAnchorWeight": demonstration_anchor_weight,
            "demonstrationAnchorLossProfile": "pose-budget-v2", "gradientSteps": gradient_steps, "replayCapacity": replay_capacity,
            "batchSize": 64, "explorationStd": exploration_std, "policyRegularization": policy_regularization, "nStepReturns": n_steps,
            "deadlinePaceLossWeight": deadline_pace_loss_weight, "criticTerminalFraction": critic_terminal_fraction,
            "replayStrategy": replay_strategy,
            "explorationProfile": exploration_profile, "poseExplorationStd": exploration.pose_standard_deviation,
            "explorationCorrelationSeconds": exploration_correlation_seconds,
            "td3Definition": {"actor": "Deterministic Beta mean, every reference axis learned; no value-only critic branch",
                "critics": "Independent twin-Q32x32 value estimators; original Simulator remains the environment",
                "discountFactor": 1.0, "targetTau": 0.005, "policyDelay": 2, "targetNoiseStd": 0.05, "targetNoiseClip": 0.1,
                "actorLearningRate": 0.0003, "criticLearningRate": 0.0003, "imitationLearningRate": 0.001,
                "terminal": "Success, fault and original task deadline all stop bootstrap; never cross resets",
                "returns": {"method": "complete-intrinsic-episode-n-step-v1", "steps": n_steps, "discountFactor": 1.0,
                    "scope": "Recorded behavior rewards/actions until terminal; no reset crossing, evaluator actor input or substitute physics",
                    "offPolicy": "Later behavior actions introduce multi-step off-policy bias; this is an explicit TD3 variant, not canonical one-step equivalence"},
                "replay": "Only complete original-engine collections; every actual query/exposure charged",
                "replayRetention": {"method": replay_strategy, "capacity": replay_capacity,
                    "scope": "Recent ring by default; optional uniform reservoir over every accepted learning row, no reward/outcome selection or new query",
                    "replacement": "At rowN draw integer0..N-1; replace that slot only if below capacity. Every row counted; Torch RNG/checkpoint exact.",
                    "offPolicy": "Reservoir retains older behavior; eight-step return approximation unchanged, no deterministic-policy truth claim"},
                "criticSampling": {"method": "uniform-terminal-mixture-v1" if critic_terminal_fraction else "uniform-replay-v1",
                    "terminalMixtureWeight": critic_terminal_fraction, "weight": "1/(occupiedRows*rowProbability), no batch normalization or clipping",
                    "scope": "Critic objective retains uniform-row expectation before gradient clipping; actor/priors independently uniform. Terminal horizons may share real end states."},
                "exploration": "Independent Gaussian noise on all four training actions, clip to contract bounds" if exploration_profile == "iid-v1" else "IID pace and correlated OU pose noise, reset every real world; training only, clip to existing contract bounds",
                "explorationDefinition": {"method": exploration_profile, "policyDtSeconds": 0.05,
                    "paceStd": exploration_std, "poseStd": exploration.pose_standard_deviation,
                    "correlationSeconds": exploration_correlation_seconds, "reset": "Zero pose state at every real episode; no mid-episode checkpoint promise",
                    "scope": "Same four bounded reference actions; no inference noise, actor truth, wider action range or changed reference filter"},
                "inference": "Unchanged deterministic mean; no inference projection or frozen head"}}


def train_seed(environment, directory, seed, config, sources, deadline, resume=False, on_checkpoint=None):
    expected = configuration(**{argument: config[key] for argument, key in
        (("seed", "seed"), ("seed_count", "seedCount"), ("updates", "updates"), ("max_steps", "maxSteps"),
         ("rollout_steps", "rolloutSteps"), ("warm_start_updates", "warmStartUpdates"), ("feature_transform", "featureTransform"),
         ("objective", "objective"), ("curriculum", "curriculum"), ("gradient_steps", "gradientSteps"),
         ("replay_capacity", "replayCapacity"), ("exploration_std", "explorationStd"),
         ("demonstration_anchor_weight", "demonstrationAnchorWeight"), ("policy_regularization", "policyRegularization"),
         ("deadline_pace_loss_weight", "deadlinePaceLossWeight"), ("n_steps", "nStepReturns"),
         ("critic_terminal_fraction", "criticTerminalFraction"), ("replay_strategy", "replayStrategy"),
         ("exploration_profile", "explorationProfile"), ("pose_exploration_std", "poseExplorationStd"),
         ("exploration_correlation_seconds", "explorationCorrelationSeconds"))})
    if config != expected:
        raise ValueError("TD3 method definition mismatch")
    torch.manual_seed(seed)
    generator = random.Random(seed)
    actor = MotionActorCritic(config["initialization"], config["featureTransform"], value_critic=False)
    learner = TD3Learner(actor)
    exploration = TrainingExploration(config["explorationStd"], config["explorationProfile"],
                                      config["poseExplorationStd"], config["explorationCorrelationSeconds"])
    replay_type = ReservoirReplayBuffer if config["replayStrategy"] == "uniform-reservoir-v1" else ReplayBuffer
    replay = replay_type(len(actor.scales), config["replayCapacity"])
    anchor = MeasuredAnchor(len(actor.scales)) if config["demonstrationAnchorWeight"] else None
    history = []
    steps = episodes = updates = collection_steps = discarded_total = 0
    identity = environment.contract["identity"]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    if resume:
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Bounded TD3 checkpoint required")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint["schemaVersion"] != 1 or checkpoint["algorithm"] != ALGORITHM or checkpoint["identity"] != identity or
                checkpoint["sources"] != sources or checkpoint["seed"] != seed or checkpoint["torchVersion"] != str(torch.__version__) or
                not compatible_config(checkpoint["config"], config) or not torch.equal(checkpoint["model"]["scales"], actor.scales)):
            raise ValueError("TD3 checkpoint source/runtime/configuration/scale mismatch")
        actor.load_state_dict(checkpoint["model"])
        learner.restore(checkpoint["learner"])
        replay.restore(checkpoint["replay"])
        if anchor is not None:
            anchor.restore(checkpoint["measuredAnchor"])
        generator.setstate(checkpoint["worldRng"])
        torch.set_rng_state(checkpoint["torchRng"])
        steps, episodes, updates, history, collection_steps, discarded_total = (checkpoint[key] for key in
            ("steps", "episodes", "updates", "history", "collectionSteps", "discardedCollectionStepsTotal"))
    elif checkpoint_path.exists():
        raise ValueError("Explicit resume required; never overwrite TD3 history")
    initial_path = directory / "initial-policy.json"
    if not resume:
        atomic_json(initial_path, export(actor, identity, seed, 0, 0, ALGORITHM))
    exposure_path = directory / "exposure.json"
    exposures = json.loads(exposure_path.read_text())["cases"] if exposure_path.is_file() else []

    def save():
        exported = export(actor, identity, seed, steps, updates, ALGORITHM)
        policy_name = f"policy-{updates:04d}.json"
        atomic_json(directory / policy_name, exported)
        checkpoint = {"schemaVersion": 1, "algorithm": ALGORITHM, "identity": identity, "sources": sources, "config": config,
                      "seed": seed, "torchVersion": str(torch.__version__), "model": actor.state_dict(), "learner": learner.state(),
                      "replay": replay.state(), "worldRng": generator.getstate(), "torchRng": torch.get_rng_state(),
                      "steps": steps, "episodes": episodes, "updates": updates, "history": history,
                      "collectionSteps": collection_steps, "discardedCollectionStepsTotal": discarded_total}
        if anchor is not None:
            checkpoint["measuredAnchor"] = anchor.state()
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        if temporary.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("TD3 checkpoint exceeds bounded reader size")
        temporary.replace(checkpoint_path)
        summary = {"seed": seed, "steps": steps, "episodes": episodes, "updates": updates, "trained": updates > 0,
                   "collectionSteps": collection_steps, "discardedCollectionStepsTotal": discarded_total,
                   "policyFile": policy_name, "policySha256": digest(directory / policy_name), "initialSha256": digest(initial_path),
                   "actorWeightsChanged": updates > 0 and json.loads(initial_path.read_text())["layers"][-1] != exported["layers"][-1],
                   "history": history, "td3GradientUpdates": learner.gradient_updates, "td3ActorUpdates": learner.actor_updates,
                   "replayRows": replay.count}
        atomic_json(directory / "summary.json", summary)
        if on_checkpoint:
            on_checkpoint(summary)
        print(json.dumps({"event": "checkpoint", "algorithm": ALGORITHM, "seed": seed, "steps": steps,
                          "episodes": episodes, "updates": updates}), flush=True)
        return summary

    summary = save()
    observed = collection_steps
    discarded = 0
    while updates < config["updates"] and observed < config["maxSteps"] and time.monotonic() < deadline:
        world_state, torch_state = generator.getstate(), torch.get_rng_state()
        start_observed = observed
        rows, worlds, anchor_rows = [], [], []
        warm = updates < config["warmStartUpdates"]
        interrupted = False
        while len(rows) < config["rolloutSteps"]:
            if observed >= config["maxSteps"] or time.monotonic() >= deadline:
                interrupted = True
                break
            world = exclude_frozen_world(training_world(generator, episodes + len(worlds), config["curriculum"]))
            exposures.append({"seed": world["seed"], "task": world["task"], "configuration": world["configuration"],
                              "stage": world["stage"], "kind": "td3-teacher" if warm else "td3-collection", "update": updates + 1})
            atomic_json(exposure_path, {"schemaVersion": 1, "cases": exposures})
            if warm:
                demonstration = collect_demonstration(environment, actor, world, deadline, config["maxSteps"] - observed,
                                                      config["teacherSelection"])
                observed += demonstration["observedSteps"]
                if demonstration["interrupted"]:
                    interrupted = True
                    break
                rows.extend(demonstration["rows"])
                anchor_rows.extend(demonstration_samples(demonstration["rows"]))
                worlds.append({**world, "reason": "success" if demonstration["rows"] else "teacher-unavailable",
                               "canonicalReward": demonstration["canonicalReward"], "teacherAttempts": demonstration["attempts"],
                               "teacherSelectedPace": demonstration["selectedPace"]})
                continue
            observation, reset_info = environment.reset(seed=world["seed"], task=world["task"], configuration=world["configuration"],
                options={"recordTransitions": False, "policyIdentity": {
                    "kind": "learned" if updates else "untrained", "id": "td3-training-exploration",
                    **({"sha256": summary["policySha256"]} if updates else {})}})
            actor.reset_policy_state(reset_info["referenceDurationSeconds"])
            exploration.reset()
            objective = LearningObjective(world["task"], config["objective"])
            total_reward = canonical_reward = 0
            while True:
                if observed >= config["maxSteps"] or time.monotonic() >= deadline:
                    interrupted = True
                    break
                deterministic, _, _, _ = actor.act(observation, deterministic=True)
                encoded = list(actor.last_input)
                action = (torch.tensor(deterministic) + exploration.sample()).clamp(-1, 1).tolist()
                following, canonical, terminated, truncated, info = environment.step(action)
                observed += 1
                reward = objective.reward(canonical, info)
                rows.append({"modelInput": encoded, "nextModelInput": actor.encode_observation(following, advance_state=False).tolist(),
                             "action": action, "reward": reward, "requiredPace": public_deadline_floor(reset_info["referenceDurationSeconds"], observation),
                             "terminal": bool(terminated or truncated)})
                total_reward += reward
                canonical_reward += canonical
                observation = following
                if terminated or truncated:
                    worlds.append({**world, "reason": info["reason"], "reward": total_reward,
                                   "canonicalReward": canonical_reward, "terminalSettlingMarginScore": objective.terminal_margin_score})
                    break
            if interrupted:
                break
        if interrupted:
            discarded = observed - start_observed
            discarded_total += discarded
            collection_steps = observed
            generator.setstate(world_state)
            torch.set_rng_state(torch_state)
            summary = save()
            break
        if warm:
            for group in learner.actor_optimizer.param_groups:
                group["lr"] = 0.001
            metrics = imitation_update(actor, learner.actor_optimizer, rows)
            if anchor is not None:
                anchor.append(anchor_rows)
            learner.sync_actor_target()
        else:
            for group in learner.actor_optimizer.param_groups:
                group["lr"] = 0.0003
            replay.append(complete_episode_returns(rows, config["nStepReturns"], learner.gamma))
            results = []
            for _ in range(config["gradientSteps"]):
                batch_size = min(config["batchSize"], replay.count)
                batch = replay.sample(batch_size, config["criticTerminalFraction"])
                actor_batch = replay.sample(batch_size) if config["criticTerminalFraction"] else None
                results.append(learner.update(batch, anchor, config["demonstrationAnchorWeight"],
                    config["demonstrationAnchorLossProfile"], config["policyRegularization"],
                    config["deadlinePaceLossWeight"], actor_batch))
            actor_results = [result for result in results if result["actorUpdated"]]
            metrics = {"phase": "td3", "meanCriticLoss": sum(result["criticLoss"] for result in results) / len(results),
                       "meanActorLoss": sum(result["actorLoss"] for result in actor_results) / len(actor_results) if actor_results else None,
                       "gradientUpdates": learner.gradient_updates, "actorUpdates": learner.actor_updates,
                       "actorStepsThisCollection": len(actor_results), "replayRows": replay.count, "nStepReturns": config["nStepReturns"],
                       "criticTerminalFraction": config["criticTerminalFraction"],
                       "replayStrategy": config["replayStrategy"], "replaySeenRows": getattr(replay, "seen_rows", None),
                       "maximumCriticGradientNorm": max(result["criticGradientNorm"] for result in results),
                       "maximumActorGradientNorm": max((result["actorGradientNorm"] for result in actor_results), default=None)}
        steps += len(rows)
        episodes += len(worlds)
        updates += 1
        collection_steps = observed
        history.append({**metrics, "worlds": worlds, "actualDecisionsThisCollection": observed - start_observed,
                        "acceptedRows": len(rows), "explorationStd": config["explorationStd"]})
        summary = save()
    return {**summary, "budgetReached": updates < config["updates"], "discardedUncommittedSteps": discarded}


def run(directory=DEFAULT_DIRECTORY, seed=201, seed_count=1, updates=64, max_steps=50000, rollout_steps=256,
        max_seconds=900, resume=None, warm_start_updates=8, feature_transform="deadline-context-v1",
        objective=SETTLING_MARGIN_OBJECTIVE, curriculum="mixed-full-reach-v5", gradient_steps=32, replay_capacity=1024,
        exploration_std=0.05, demonstration_anchor_weight=10, policy_regularization="public-deadline-residual-v1",
        deadline_pace_loss_weight=40, n_steps=1, critic_terminal_fraction=0, replay_strategy="recent-ring-v1",
        exploration_profile="iid-v1", pose_exploration_std=None, exploration_correlation_seconds=0):
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, int) or not 1 <= max_seconds <= 900:
        raise ValueError("Bounded TD3 wall-clock budget required")
    config = configuration(seed, seed_count, updates, max_steps, rollout_steps, warm_start_updates, feature_transform,
                           objective, curriculum, gradient_steps, replay_capacity, exploration_std,
                           demonstration_anchor_weight, policy_regularization, deadline_pace_loss_weight, n_steps,
                           critic_terminal_fraction, replay_strategy, exploration_profile,
                           pose_exploration_std, exploration_correlation_seconds)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    directory = Path(directory).resolve()
    run_id = resume or str(uuid.uuid4())
    if str(uuid.UUID(run_id)) != run_id:
        raise ValueError("Valid explicit TD3 run ID required")
    run_directory = directory / run_id
    manifest_path = run_directory / "run.json"
    sources = source_identity()
    if resume:
        previous = json.loads(manifest_path.read_text())
        if previous["sources"] != sources or not compatible_config(previous["config"], config):
            raise ValueError("TD3 run source/configuration mismatch")
    elif run_directory.exists():
        raise ValueError("Never overwrite an existing TD3 run")
    manifest = {"schemaVersion": 1, "runId": run_id, "config": config, "sources": sources, "status": "running",
                "device": "cpu", "torch": str(torch.__version__), "models": previous["models"] if resume else [], "learnedImprovementVerified": False,
                "scope": "Experimental deterministic reference learner; original Simulator/iraLIB only, no obstacle/game/AMD/hardware claim"}

    def publish():
        atomic_json(manifest_path, manifest)
        atomic_json(directory / "latest.json", {"schemaVersion": 1, "runId": run_id})

    deadline = time.monotonic() + max_seconds
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
    print(json.dumps({"event": "finished", "algorithm": ALGORITHM, "runId": run_id, "status": manifest["status"]}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Bounded experimental deterministic learner through original Simulator/iraLIB")
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--resume")
    parser.add_argument("--warm-start-updates", type=int, default=8)
    parser.add_argument("--feature-transform", choices=FEATURE_TRANSFORMS, default="deadline-context-v1")
    parser.add_argument("--objective", choices=PROFILES, default=SETTLING_MARGIN_OBJECTIVE)
    parser.add_argument("--curriculum", choices=CURRICULA, default="mixed-full-reach-v5")
    parser.add_argument("--gradient-steps", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=1024)
    parser.add_argument("--exploration-std", type=float, default=0.05)
    parser.add_argument("--demonstration-anchor-weight", type=float, default=10)
    parser.add_argument("--policy-regularization", choices=("none", "public-deadline-residual-v1"), default="public-deadline-residual-v1")
    parser.add_argument("--deadline-pace-loss-weight", type=float, default=40)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--critic-terminal-fraction", type=float, default=0)
    parser.add_argument("--replay-strategy", choices=("recent-ring-v1", "uniform-reservoir-v1"), default="recent-ring-v1")
    parser.add_argument("--exploration-profile", choices=("iid-v1", "ou-pose-v1"), default="iid-v1")
    parser.add_argument("--pose-exploration-std", type=float, default=None)
    parser.add_argument("--exploration-correlation-seconds", type=float, default=0)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
