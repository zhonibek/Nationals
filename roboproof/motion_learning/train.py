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
from .model import MotionActorCritic, export, update, public_deadline_floor, FEATURE_TRANSFORMS, TEMPORAL_FEATURES, DEADLINE_FEATURES, HISTORY_FRAMES, DEADLINE_CONTEXT, normalization_definition, exploration_factor, validate_policy_kl_limit
from .objective import LearningObjective, OBJECTIVE, PROFILES, SETTLING_MARGIN_OBJECTIVE, objective_definition, discount_parameters, return_estimator
from .imitation import collect_demonstration, imitation_update, TEACHER_PACES, TEACHER_SELECTIONS, MeasuredAnchor, demonstration_samples, ANCHOR_CAPACITY, ANCHOR_ROWS_PER_WORLD, ANCHOR_LOSS_PROFILES, POSE_BUDGET_SCALES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTORY = ROOT / "roboproof/runs/motion/learning"
CURRICULUM = "mixed-reach-v2"
CURRICULA = (CURRICULUM, "field-reach-v3", "field-reach-v4", "mixed-full-reach-v5")


def curriculum_definition(curriculum):
    if curriculum not in CURRICULA:
        raise ValueError("Unsupported training curriculum")
    if curriculum != "mixed-full-reach-v5":
        return None
    return {"method": "mixed-full-reach-v5", "initialEasyEpisodes": 4, "familyCycle": ["field", "full", "short", "field"],
            "short": {"startLimitInches": 12, "distanceInches": [12, 40], "goalHeadingDegrees": [-135, 135]},
            "field": {"startLimitInches": 40, "goalLimitInches": 45, "goalHeadingDegrees": [-175, 175]},
            "full": {"startLimitInches": 50, "goalLimitInches": 60, "goalHeadingDegrees": [-180, 180],
                     "boundaryProbability": 0.5},
            "startHeadingDegrees": [-45, 45], "deadlineChoicesSeconds": [6, 10, 10, 10],
            "physicalVariation": "Uniform nominal/heavy/battery/wide categories within the public original-engine ranges",
            "scope": "Training distribution only; original reward, observations, actions, controller and pass gates unchanged"}


def validate_online_anchor_interval(value, warm_start_updates, anchor_weight):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 16:
        raise ValueError("Bounded online anchor interval from 0 to 16 required")
    if value and (isinstance(warm_start_updates, bool) or not isinstance(warm_start_updates, int) or warm_start_updates < 1 or
                  isinstance(anchor_weight, bool) or not isinstance(anchor_weight, (int, float)) or
                  not math.isfinite(anchor_weight) or anchor_weight <= 0):
        raise ValueError("Online anchors require an active measured warm-start anchor")
    return value


def online_anchor_definition(interval):
    return None if interval == 0 else {"method": "periodic-measured-retention-v1", "everyPpoUpdates": interval,
        "firstRefresh": "First PPO update", "worldsPerRefresh": 1, "worldSelection": "First training world of the selected rollout",
        "teacher": "Same measured three-pace original-Simulator teacher and declared selection",
        "buffer": "Existing512-row capacity and16 chronological rows per successful teacher world",
        "accounting": "Every teacher decision charged; commit refreshed rows only with a complete PPO collection",
        "scope": "Training-only soft retention on all four heads; no inference teacher or action projection"}


def validate_ppo_actor_learning_rate(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or
            not 0.00005 <= value <= 0.001):
        raise ValueError("Finite positive PPO actor learning rate from 0.00005 to 0.001 required")
    return float(value)


def learning_rate_definition(rate):
    return {"method": "phase-specific-actor-rate-v1", "imitationActor": 0.001, "ppoActor": rate, "critic": 0.0003,
            "switch": "At gradient update, preserving optimizer moments and the collected rollout",
            "scope": "Training optimizer only; all four action heads remain learnable"}


def optimizer_constraint_definition(limit):
    return None if limit is None else {"method": "collected-beta-kl-backtrack-v1", "meanKlLimit": limit,
        "maximumBacktracks": 6, "actorTrialRates": "Positive base rate times1,1/2,...,1/64",
        "rollback": "Restore all parameters and Adam moments before every retry; reuse the same clipped gradients",
        "critic": "One unchanged-rate update per minibatch, including rejected-actor fallback",
        "scope": "Optimizer constraint on collected states only; no inference projection or independent safety guarantee"}


def full_reach_world(generator, episode_index):
    stage = (1, 2, 0, 1)[(episode_index - 4) % 4]
    start_limit = (12, 40, 50)[stage]
    start = {"xIn": generator.uniform(-start_limit, start_limit), "yIn": generator.uniform(-start_limit, start_limit),
             "headingDeg": generator.uniform(-45, 45)}
    if stage == 0:
        direction = generator.uniform(-math.pi, math.pi)
        distance = generator.uniform(12, 40)
        goal = {"xIn": start["xIn"] + math.sin(direction) * distance,
                "yIn": start["yIn"] + math.cos(direction) * distance, "headingDeg": generator.uniform(-135, 135)}
    else:
        goal_limit, heading_limit = (45, 175) if stage == 1 else (60, 180)
        goal = {"xIn": generator.uniform(-goal_limit, goal_limit), "yIn": generator.uniform(-goal_limit, goal_limit),
                "headingDeg": generator.uniform(-heading_limit, heading_limit)}
        if stage == 2 and generator.random() < 0.5:
            goal[generator.choice(("xIn", "yIn"))] = generator.choice((-60, 60))
            goal["headingDeg"] = generator.choice((0, 90, 180))
    variation = generator.randrange(4)
    ranges = ({}, {"massKg": (10, 13.6), "moiKgM2": (0.35, 0.58), "muLong": (0.3, 0.55), "kSlip": (325, 500)},
              {"massKg": (8, 12), "muLong": (0.4, 0.8), "batteryInternalR": (0.04, 0.12)},
              {"massKg": (3.4, 13.6), "moiKgM2": (0.0725, 0.58), "muLong": (0.3, 1.1), "muLat": (0.02, 0.15),
               "kSlip": (325, 975), "batteryInternalR": (0, 0.12)})[variation]
    configuration = {name: generator.uniform(*bounds) for name, bounds in ranges.items()}
    return {"seed": generator.randrange(2**32), "task": {"start": start, "goal": goal,
            "deadlineSeconds": generator.choice((6, 10, 10, 10))}, "configuration": configuration, "stage": stage}


def exclude_frozen_world(world):
    for filename in ("motion-evaluation-worlds.json", "motion-evaluation-worlds-v2.json"):
        for frozen in json.loads((ROOT / "tests/fixtures" / filename).read_text(encoding="utf-8"))["cases"]:
            if world["seed"] == frozen["seed"] or world["task"] == frozen["task"]:
                raise RuntimeError("Frozen evaluation world is excluded from training")
    return world


def atomic_json(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"
    if len(encoded.encode("utf-8")) > 4 * 1024 * 1024:
        raise ValueError("Learning artifact exceeds bounded reader size")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(filename)


def digest(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def source_identity():
    names = ["roboproof/motion_learning/model.py", "roboproof/motion_learning/train.py", "roboproof/motion_learning/objective.py", "roboproof/motion_learning/imitation.py",
             "roboproof/motion_env.py", "roboproof/motion-bridge.js",
             "tests/fixtures/motion-evaluation-worlds.json", "tests/fixtures/motion-evaluation-worlds-v2.json"]
    return {name: hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
            for name in names}


def compatible_config(previous, current):
    return (all(previous[key] == current[key] for key in ("rolloutSteps", "seed", "seedCount", "algorithm")) and
            previous.get("curriculum") == current.get("curriculum") and
            previous.get("initialization") == current.get("initialization") and previous.get("objective") == current.get("objective") and
            previous.get("policyRegularization", "none") == current.get("policyRegularization", "none") and
            previous.get("featureTransform", "identity-v1") == current.get("featureTransform", "identity-v1") and
            previous.get("warmStartUpdates", 0) == current.get("warmStartUpdates", 0) and
            previous.get("demonstrationAnchorWeight", 0) == current.get("demonstrationAnchorWeight", 0) and
            previous.get("teacherSelection", "minimum-effort-v1") == current.get("teacherSelection", "minimum-effort-v1") and
            previous.get("deadlinePaceLossWeight", 1) == current.get("deadlinePaceLossWeight", 1) and
            previous.get("demonstrationAnchorLossProfile", "uniform-action-v1") == current.get("demonstrationAnchorLossProfile", "uniform-action-v1") and
            previous.get("explorationConcentration", 1) == current.get("explorationConcentration", 1) and
            previous.get("onlineAnchorInterval", 0) == current.get("onlineAnchorInterval", 0) and
            previous.get("ppoActorLearningRate", 0.001) == current.get("ppoActorLearningRate", 0.001) and
            previous.get("targetPolicyKL") == current.get("targetPolicyKL") and
            all(previous.get(key) == current.get(key) for key in ("actorCritic", "actorLearningRate", "criticLearningRate", "regularizationDefinition", "returnEstimator", "temporalDefinition", "deadlineContextDefinition", "warmStartDefinition", "demonstrationAnchorDefinition", "explorationDefinition", "curriculumDefinition", "onlineAnchorDefinition", "learningRateDefinition", "klDiagnosticDefinition", "optimizerConstraintDefinition", "objectiveDefinition", "normalizationDefinition")) and
            current["updates"] >= previous["updates"] and current["maxSteps"] >= previous["maxSteps"])


def training_world(generator, episode_index, curriculum=CURRICULUM):
    if curriculum not in CURRICULA:
        raise ValueError("Unsupported training curriculum")
    if curriculum == "mixed-full-reach-v5" and episode_index >= 4:
        return exclude_frozen_world(full_reach_world(generator, episode_index))
    stage = 0 if episode_index < 4 else (1, 2, 0, 2)[(episode_index - 4) % 4]
    start = {"xIn": generator.uniform(-8, 8), "yIn": generator.uniform(-8, 8),
             "headingDeg": generator.uniform(-10, 10) if stage else 0}
    direction = generator.uniform(-math.pi, math.pi) if stage else generator.uniform(-0.4, 0.4)
    distance = generator.uniform(10, 28) if stage else generator.uniform(8, 20)
    goal = {"xIn": start["xIn"] + math.sin(direction) * distance,
            "yIn": start["yIn"] + math.cos(direction) * distance,
            "headingDeg": generator.uniform(-50, 50) if stage else 0}
    if curriculum in ("field-reach-v3", "field-reach-v4") and stage:
        start = {"xIn": generator.uniform(-40, 40), "yIn": generator.uniform(-40, 40), "headingDeg": generator.uniform(-45, 45)}
        goal = {"xIn": generator.uniform(-45, 45), "yIn": generator.uniform(-45, 45), "headingDeg": generator.uniform(-175, 175)}
    configuration = {} if stage < 2 else {"massKg": generator.uniform(7, 12),
                    "muLong": generator.uniform(0.35, 0.85), "kSlip": generator.uniform(350, 850),
                    "batteryInternalR": generator.uniform(0.01, 0.09)}
    if curriculum == "field-reach-v4" and stage == 2:
        configuration = {"massKg": generator.uniform(3.4, 13.6), "moiKgM2": generator.uniform(0.0725, 0.58),
                         "muLong": generator.uniform(0.3, 1.1), "muLat": generator.uniform(0.02, 0.15), "kSlip": generator.uniform(325, 975),
                         "batteryInternalR": generator.uniform(0, 0.12)}
        if episode_index % 8 == 7:
            axis = generator.choice(("xIn", "yIn"))
            goal[axis] = generator.choice((-60, 60))
            goal["headingDeg"] = generator.choice((0, 90, 180))
    world = {"seed": generator.randrange(2**32), "task": {"start": start, "goal": goal,
             "deadlineSeconds": 4 if stage == 0 else 6}, "configuration": configuration, "stage": stage}
    return exclude_frozen_world(world)


def train_seed(environment, directory, seed, config, sources, deadline, resume=False, on_checkpoint=None):
    expected_normalization = normalization_definition(config.get("featureTransform", "identity-v1"))
    if config.get("normalizationDefinition") != expected_normalization:
        raise ValueError("Declared sensor normalization definition mismatch")
    expected_objective = objective_definition(config.get("objective", "canonical-v1"))
    if config.get("objectiveDefinition", expected_objective) != expected_objective:
        raise ValueError("Declared learning objective definition mismatch")
    if (config.get("curriculum") == "mixed-full-reach-v5" and
            config.get("curriculumDefinition") != curriculum_definition("mixed-full-reach-v5")):
        raise ValueError("Declared full-reach curriculum definition required")
    concentration = exploration_factor(config.get("explorationConcentration", 1))
    policy_kl_limit = validate_policy_kl_limit(config.get("targetPolicyKL"))
    expected_constraint = optimizer_constraint_definition(policy_kl_limit)
    if config.get("optimizerConstraintDefinition", expected_constraint) != expected_constraint:
        raise ValueError("Declared optimizer constraint mismatch")
    ppo_actor_rate = validate_ppo_actor_learning_rate(config.get("ppoActorLearningRate", 0.001))
    expected_rates = learning_rate_definition(ppo_actor_rate)
    if config.get("learningRateDefinition", expected_rates) != expected_rates:
        raise ValueError("Declared learning rate definition mismatch")
    refresh_interval = validate_online_anchor_interval(config.get("onlineAnchorInterval", 0), config.get("warmStartUpdates", 0),
                                                      config.get("demonstrationAnchorWeight", 0))
    expected_refresh = online_anchor_definition(refresh_interval)
    if config.get("onlineAnchorDefinition", expected_refresh) != expected_refresh:
        raise ValueError("Declared online anchor definition mismatch")
    anchor_profile = config.get("demonstrationAnchorLossProfile", "uniform-action-v1")
    if anchor_profile not in ANCHOR_LOSS_PROFILES or (anchor_profile != "uniform-action-v1" and not config.get("demonstrationAnchorWeight", 0)):
        raise ValueError("A declared active anchor loss profile is required")
    pace_weight = config.get("deadlinePaceLossWeight", 1)
    if (isinstance(pace_weight, bool) or not isinstance(pace_weight, (int, float)) or not math.isfinite(pace_weight) or
            not 0 <= pace_weight <= 100 or (config.get("policyRegularization", "none") == "none" and pace_weight != 1)):
        raise ValueError("Bounded active deadline pacing loss weight required")
    if config.get("featureTransform") in DEADLINE_FEATURES:
        definition = config.get("deadlineContextDefinition")
        if (not isinstance(definition, dict) or definition != DEADLINE_CONTEXT or type(definition.get("schemaVersion")) is not int):
            raise ValueError("Declared public deadline context mismatch")
    expected_estimator = return_estimator(config.get("objective", "canonical-v1"))
    if config.get("returnEstimator", expected_estimator) != expected_estimator:
        raise ValueError("Declared return estimator mismatch")
    torch.manual_seed(seed)
    generator = random.Random(seed)
    model = MotionActorCritic(config.get("initialization", "conservative-v1"), config.get("featureTransform", "identity-v1"))
    anchor_weight = config.get("demonstrationAnchorWeight", 0)
    measured_anchor = MeasuredAnchor(len(model.scales)) if anchor_weight else None
    optimizer = torch.optim.Adam([
        {"params": [*model.body.parameters(), *model.actor.parameters()], "lr": 1e-3},
        {"params": [*model.critic_body.parameters(), *model.critic.parameters()], "lr": 3e-4}])
    history = []
    steps = episodes = updates = 0
    collection_steps = 0
    discarded_collection_steps_total = 0
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
        if expected_normalization is not None:
            saved_scales = checkpoint["model"].get("scales")
            if (not isinstance(saved_scales, torch.Tensor) or saved_scales.dtype != torch.float32 or
                    not torch.equal(saved_scales, model.scales)):
                raise ValueError("Checkpoint sensor normalization mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.setstate(checkpoint["worldRng"])
        torch.set_rng_state(checkpoint["torchRng"])
        if measured_anchor is not None:
            measured_anchor.restore(checkpoint.get("measuredAnchor"))
        steps, episodes, updates, history = (checkpoint[key] for key in ("steps", "episodes", "updates", "history"))
        collection_steps = checkpoint.get("collectionSteps", steps)
        discarded_collection_steps_total = checkpoint.get("discardedCollectionStepsTotal", 0)
    elif checkpoint_path.exists():
        raise ValueError("Use an explicit resume, never overwrite a run")
    initial_path = directory / "initial-policy.json"
    if not resume:
        atomic_json(initial_path, export(model, identity, seed, 0, 0))
    exposure_path = directory / "exposure.json"
    if exposure_path.is_file() and exposure_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Teacher exposure journal exceeds bounded reader size")
    training_exposures = json.loads(exposure_path.read_text(encoding="utf-8"))["cases"] if exposure_path.is_file() else []

    def save():
        exported = export(model, identity, seed, steps, updates)
        policy_name = f"policy-{updates:04d}.json"
        atomic_json(directory / policy_name, exported)
        checkpoint = {"schemaVersion": 1, "identity": identity, "sources": sources, "config": config,
                      "seed": seed, "torchVersion": str(torch.__version__), "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "worldRng": generator.getstate(),
                      "torchRng": torch.get_rng_state(), "steps": steps, "episodes": episodes,
                      "updates": updates, "history": history, "collectionSteps": collection_steps,
                      "discardedCollectionStepsTotal": discarded_collection_steps_total}
        if measured_anchor is not None:
            checkpoint["measuredAnchor"] = measured_anchor.state()
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        if temporary.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Checkpoint exceeds bounded reader size")
        temporary.replace(checkpoint_path)
        row = {"seed": seed, "steps": steps, "episodes": episodes, "updates": updates, "trained": updates > 0,
               "collectionSteps": collection_steps,
               "discardedCollectionStepsTotal": discarded_collection_steps_total,
               "policyFile": policy_name, "policySha256": digest(directory / policy_name),
               "initialSha256": digest(initial_path), "actorWeightsChanged": updates > 0 and
               json.loads(initial_path.read_text())["layers"][-1] != exported["layers"][-1], "history": history}
        atomic_json(directory / "summary.json", row)
        if on_checkpoint:
            on_checkpoint(row)
        print(json.dumps({"event": "checkpoint", **{key: row[key] for key in ("seed", "steps", "episodes", "updates")}}), flush=True)
        return row

    summary = save()
    observed_steps = collection_steps
    discarded_steps = 0
    while updates < config["updates"] and observed_steps < config["maxSteps"] and time.monotonic() < deadline:
        imitation = updates < config.get("warmStartUpdates", 0)
        saved_world_rng = generator.getstate()
        saved_torch_rng = torch.get_rng_state()
        rows = []
        batch_episodes = []
        anchor_rows = []
        interrupted = False
        while len(rows) < config["rolloutSteps"]:
            world = training_world(generator, episodes + len(batch_episodes), config.get("curriculum", CURRICULUM))
            if not any(previous["seed"] == world["seed"] and previous["task"] == world["task"] for previous in training_exposures):
                training_exposures.append(world)
                atomic_json(exposure_path, {"schemaVersion": 1,
                    "scope": "All requested training worlds, including interrupted/uncommitted teacher and PPO exposure",
                    "cases": training_exposures})
            if imitation:
                demonstration = collect_demonstration(environment, model, world, deadline, config["maxSteps"] - observed_steps,
                    config.get("teacherSelection", "minimum-effort-v1"))
                observed_steps += demonstration["observedSteps"]
                if demonstration["interrupted"]:
                    interrupted = True
                    break
                rows.extend(demonstration["rows"])
                if measured_anchor is not None:
                    anchor_rows.extend(demonstration_samples(demonstration["rows"]))
                batch_episodes.append({"seed": world["seed"], "stage": world["stage"], "task": world["task"],
                    "configuration": world["configuration"], "reason": demonstration["reason"],
                    "reward": demonstration["canonicalReward"], "canonicalReward": demonstration["canonicalReward"],
                    "teacherSelectedPace": demonstration["selectedPace"], "teacherAttempts": demonstration["attempts"],
                    "teacherSelection": demonstration["teacherSelection"], "teacherSelectedMarginScore": demonstration["selectedMarginScore"]})
                continue
            online_demonstration = None
            if refresh_interval and not batch_episodes and (updates - config.get("warmStartUpdates", 0)) % refresh_interval == 0:
                online_demonstration = collect_demonstration(environment, model, world, deadline, config["maxSteps"] - observed_steps,
                    config.get("teacherSelection", "minimum-effort-v1"))
                observed_steps += online_demonstration["observedSteps"]
                if online_demonstration["interrupted"]:
                    interrupted = True
                    break
                anchor_rows.extend(demonstration_samples(online_demonstration["rows"]))
            observation, reset_info = environment.reset(seed=world["seed"], configuration=world["configuration"], task=world["task"],
                options={"recordTransitions": False, "policyIdentity": {"kind": "untrained", "id": "ppo-training-rollout"}})
            model.reset_policy_state(reset_info["referenceDurationSeconds"])
            total_reward = 0
            total_canonical_reward = 0
            objective = LearningObjective(world["task"], config.get("objective", "canonical-v1"))
            while True:
                if observed_steps >= config["maxSteps"] or time.monotonic() >= deadline:
                    interrupted = True
                    break
                action, unit_action, log_prob, value = model.act(observation, exploration_concentration=concentration)
                model_input = list(model.last_input)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                learning_reward = objective.reward(reward, info)
                observed_steps += 1
                _, _, _, next_value = model.act(next_observation, deterministic=True, advance_state=False)
                rows.append({"observation": observation, "modelInput": model_input, "action": unit_action, "logProb": float(log_prob),
                             "explorationConcentration": concentration,
                             "value": float(value), "nextValue": float(next_value), "reward": learning_reward,
                             "requiredPace": public_deadline_floor(reset_info["referenceDurationSeconds"], observation),
                             "terminated": objective.terminal(terminated, truncated, info), "ended": terminated or truncated})
                total_reward += learning_reward
                total_canonical_reward += reward
                observation = next_observation
                if terminated or truncated:
                    batch_episodes.append({"seed": world["seed"], "stage": world["stage"],
                                           "task": world["task"], "configuration": world["configuration"],
                                           "reason": info["reason"], "reward": total_reward, "canonicalReward": total_canonical_reward,
                                           **({"terminalSettlingMarginScore": objective.terminal_margin_score}
                                              if objective.profile == SETTLING_MARGIN_OBJECTIVE else {}),
                                           **({"onlineTeacherAttempts": online_demonstration["attempts"],
                                               "onlineTeacherSelectedPace": online_demonstration["selectedPace"],
                                               "onlineTeacherSelectedMarginScore": online_demonstration["selectedMarginScore"]}
                                              if online_demonstration is not None else {})})
                    break
            if interrupted:
                break
        if interrupted:
            generator.setstate(saved_world_rng)
            torch.set_rng_state(saved_torch_rng)
            discarded_steps = observed_steps - collection_steps
            discarded_collection_steps_total += discarded_steps
            collection_steps = observed_steps
            summary = save()
            break
        gamma, decay = discount_parameters(config.get("objective", "canonical-v1"))
        optimizer.param_groups[0]["lr"] = 0.001 if imitation else ppo_actor_rate
        if not imitation and anchor_rows:
            measured_anchor.append(anchor_rows)
        metrics = imitation_update(model, optimizer, rows) if imitation else update(model, optimizer, rows, gamma=gamma, decay=decay,
                         policy_regularization=config.get("policyRegularization", "none"),
                         measured_anchor=measured_anchor, anchor_weight=anchor_weight, deadline_pace_loss_weight=pace_weight,
                         anchor_loss_profile=anchor_profile, target_policy_kl=policy_kl_limit)
        if imitation and measured_anchor is not None:
            measured_anchor.append(anchor_rows)
        steps += len(rows)
        collection_steps = observed_steps
        episodes += len(batch_episodes)
        updates += 1
        history.append({"update": updates, "steps": steps, "episodes": episodes, "phase": "measured-controller-imitation" if imitation else "ppo",
                        "demonstrationAnchorRows": len(measured_anchor.features) if measured_anchor is not None else 0,
                        "teacherQueries": sum(len(row.get("teacherAttempts", [])) + len(row.get("onlineTeacherAttempts", []))
                                              for row in batch_episodes),
                        "onlineTeacherQueries": sum(len(row.get("onlineTeacherAttempts", [])) for row in batch_episodes),
                        "onlineAnchorInterval": refresh_interval, **metrics,
                        "actorLearningRate": optimizer.param_groups[0]["lr"], "criticLearningRate": optimizer.param_groups[1]["lr"],
                        "meanTrainingReward": sum(row["reward"] for row in batch_episodes) / len(batch_episodes),
                        "meanCanonicalReward": sum(row["canonicalReward"] for row in batch_episodes) / len(batch_episodes),
                        "trainingSuccessRate": sum(row["reason"] == "success" for row in batch_episodes) / len(batch_episodes),
                        "curriculumStageCounts": {str(stage): sum(row["stage"] == stage for row in batch_episodes)
                                                  for stage in range(3)},
                        "worlds": batch_episodes})
        summary = save()
    return {**summary, "discardedUncommittedSteps": discarded_steps,
            "budgetReached": updates < config["updates"]}


def run(directory=DEFAULT_DIRECTORY, seed=42, seed_count=3, updates=8, max_steps=4096,
        rollout_steps=128, max_seconds=180, resume=None, initialization="broad-pace-v1", objective=OBJECTIVE, curriculum=CURRICULUM,
        policy_regularization="none", feature_transform="identity-v1", warm_start_updates=0, demonstration_anchor_weight=0,
        teacher_selection="minimum-effort-v1", deadline_pace_loss_weight=1, demonstration_anchor_loss_profile="uniform-action-v1",
        exploration_concentration=1, online_anchor_interval=0, ppo_actor_learning_rate=0.001, target_policy_kl=None):
    exploration_concentration = exploration_factor(exploration_concentration)
    target_policy_kl = validate_policy_kl_limit(target_policy_kl)
    ppo_actor_learning_rate = validate_ppo_actor_learning_rate(ppo_actor_learning_rate)
    refresh_interval = validate_online_anchor_interval(online_anchor_interval, warm_start_updates, demonstration_anchor_weight)
    limits = [(seed, 0, 2**32 - 3), (seed_count, 1, 3), (updates, 1, 100),
              (max_steps, 1, 50000), (rollout_steps, 16, 1024), (max_seconds, 1, 900), (warm_start_updates, 0, 16)]
    if any(isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high for value, low, high in limits):
        raise ValueError("Invalid bounded CPU training budget")
    if (initialization not in ("conservative-v1", "broad-pace-v1") or objective not in PROFILES or
            curriculum not in CURRICULA or
            policy_regularization not in ("none", "public-deadline-residual-v1") or
            feature_transform not in FEATURE_TRANSFORMS or teacher_selection not in TEACHER_SELECTIONS):
        raise ValueError("Unsupported declared learning experiment")
    if warm_start_updates >= updates:
        raise ValueError("Warm start must leave at least one PPO update")
    if (demonstration_anchor_loss_profile not in ANCHOR_LOSS_PROFILES or
            (demonstration_anchor_loss_profile != "uniform-action-v1" and demonstration_anchor_weight == 0)):
        raise ValueError("A declared active anchor loss profile is required")
    if (isinstance(deadline_pace_loss_weight, bool) or not isinstance(deadline_pace_loss_weight, (int, float)) or
            not math.isfinite(deadline_pace_loss_weight) or not 0 <= deadline_pace_loss_weight <= 100 or
            (policy_regularization == "none" and deadline_pace_loss_weight != 1)):
        raise ValueError("Bounded active deadline pacing loss weight required")
    if teacher_selection != "minimum-effort-v1" and warm_start_updates == 0:
        raise ValueError("A margin-aware teacher requires a warm start")
    if (isinstance(demonstration_anchor_weight, bool) or not isinstance(demonstration_anchor_weight, (int, float)) or
            not math.isfinite(demonstration_anchor_weight) or not 0 <= demonstration_anchor_weight <= 100 or
            (demonstration_anchor_weight > 0 and warm_start_updates == 0)):
        raise ValueError("Bounded demonstration anchor weight requires a warm start")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    directory = Path(directory).resolve()
    run_id = resume or str(uuid.uuid4())
    if str(uuid.UUID(run_id)) != run_id:
        raise ValueError("Invalid training run ID")
    run_directory = directory / run_id
    config = {"updates": updates, "maxSteps": max_steps, "rolloutSteps": rollout_steps,
              "seed": seed, "seedCount": seed_count, "algorithm": "ppo-beta-reference-v1", "curriculum": curriculum,
              "curriculumDefinition": curriculum_definition(curriculum),
              "initialization": initialization, "objective": objective, "objectiveDefinition": objective_definition(objective),
              "policyRegularization": policy_regularization,
              "deadlinePaceLossWeight": deadline_pace_loss_weight,
              "explorationConcentration": exploration_concentration,
              "onlineAnchorInterval": refresh_interval, "onlineAnchorDefinition": online_anchor_definition(refresh_interval),
              "ppoActorLearningRate": ppo_actor_learning_rate, "learningRateDefinition": learning_rate_definition(ppo_actor_learning_rate),
              "targetPolicyKL": target_policy_kl, "optimizerConstraintDefinition": optimizer_constraint_definition(target_policy_kl),
              "klDiagnosticDefinition": {"method": "rollout-beta-analytic-kl-v1", "direction": "collected actor to updated actor",
                  "sampling": "Same recorded per-row exploration concentration", "precision": "float64 Beta KL",
                  "states": "Every collected encoded observation, after each optimizer step",
                  "scope": "Read-only training diagnostic; no optimizer constraint or action projection"},
              "explorationDefinition": {"method": "scaled-beta-concentration-v1", "factor": exploration_concentration,
                  "sampling": "Beta(alpha*factor,beta*factor); full support in every action dimension",
                  "likelihood": "Replay per-transition factor for PPO unit-action log probabilities and entropy",
                  "inference": "Unchanged deterministic Beta mean; no action projection or frozen head"},
              "featureTransform": feature_transform,
              "normalizationDefinition": normalization_definition(feature_transform),
              "warmStartUpdates": warm_start_updates,
              "teacherSelection": teacher_selection,
              "demonstrationAnchorWeight": demonstration_anchor_weight,
              "demonstrationAnchorLossProfile": demonstration_anchor_loss_profile,
              "demonstrationAnchorDefinition": None if demonstration_anchor_weight == 0 else
                  {"method": "measured-mean-anchor-v1", "capacity": ANCHOR_CAPACITY, "rowsPerWorld": ANCHOR_ROWS_PER_WORLD,
                   "selection": "Even chronological samples of successful warm-start trajectories; retain last 512 rows",
                   "minibatchRows": 32, "loss": "Signed Beta-mean MSE, alongside PPO; same actor clipping",
                   "lossProfile": demonstration_anchor_loss_profile,
                   "axisScales": list(POSE_BUDGET_SCALES) if demonstration_anchor_loss_profile == "pose-budget-v2" else [1, 1, 1, 1],
                   "precisionDefinition": None if demonstration_anchor_loss_profile == "uniform-action-v1" else
                       {"translationOffsetMeters": 0.08, "headingOffsetRadians": 0.1, "positionReferenceMeters": 0.02032,
                        "headingReferenceRadians": 0.035, "fraction": 0.2, "scope": "Training loss units only; unchanged physical pass thresholds"},
                   "scope": "Training-only measured demonstrations; no deployed action projection"},
              "warmStartDefinition": None if warm_start_updates == 0 else
                  {"method": "measured-controller-imitation-v1", "teacherPaces": list(TEACHER_PACES), "epochs": 32,
                   "selection": teacher_selection,
                   "marginDefinition": None if teacher_selection == "minimum-effort-v1" else
                       {"positionReferenceMeters": 0.02032, "headingReferenceRadians": 0.035,
                        "minimumPoseMarginFraction": 0.2, "minimumSettlingReserveSeconds": 1.0,
                        "ranking": "Success and lowest contact first; least effort within margin-qualified teachers, else best worst margin",
                        "scope": "Training teacher only; unchanged original pass thresholds and actor inputs"},
                   "initialSnapshot": "Before imitation; no evaluator truth in actor features"},
              "deadlineContextDefinition": dict(DEADLINE_CONTEXT) if feature_transform in DEADLINE_FEATURES else None,
              "temporalDefinition": None if feature_transform not in TEMPORAL_FEATURES else
                  {"historyFrames": HISTORY_FRAMES[feature_transform], "initialContextRawIndices": [10, 11, 12, 13], "padding": "repeat-first-observation",
                   "reset": "every episode", "inputs": "Current/past raw sensor/reference values and frozen initial task error only"},
              "regularizationDefinition": None if policy_regularization == "none" else
                  {"paceLossWeight": deadline_pace_loss_weight, "residualKLWeight": 0.02, "residualBetaConcentration": 60.0,
                   "settlingReserveSeconds": 2.0, "inputs": "Known reference duration/time and task deadline only; no evaluator truth"},
              "actorCritic": "separate-bodies-v1", "actorLearningRate": 0.001, "criticLearningRate": 0.0003}
    config["returnEstimator"] = return_estimator(objective)
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
    parser.add_argument("--initialization", choices=("conservative-v1", "broad-pace-v1"), default="broad-pace-v1")
    parser.add_argument("--objective", choices=PROFILES, default=OBJECTIVE)
    parser.add_argument("--curriculum", choices=CURRICULA, default=CURRICULUM)
    parser.add_argument("--policy-regularization", choices=("none", "public-deadline-residual-v1"), default="none")
    parser.add_argument("--deadline-pace-loss-weight", type=float, default=1)
    parser.add_argument("--exploration-concentration", type=float, default=1)
    parser.add_argument("--online-anchor-interval", type=int, default=0)
    parser.add_argument("--ppo-actor-learning-rate", type=float, default=0.001)
    parser.add_argument("--target-policy-kl", type=float)
    parser.add_argument("--feature-transform", choices=FEATURE_TRANSFORMS, default="identity-v1")
    parser.add_argument("--warm-start-updates", type=int, default=0)
    parser.add_argument("--demonstration-anchor-weight", type=float, default=0)
    parser.add_argument("--demonstration-anchor-loss-profile", choices=ANCHOR_LOSS_PROFILES, default="uniform-action-v1")
    parser.add_argument("--teacher-selection", choices=TEACHER_SELECTIONS, default="minimum-effort-v1")
    arguments = parser.parse_args()
    run(**vars(arguments))


if __name__ == "__main__":
    main()
