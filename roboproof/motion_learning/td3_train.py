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
from .checkpoint_limits import checkpoint_size_limit
from .actor_artifact import atomic_actor_json
from .execution import DEVICES, prepare_execution, snapshot_cuda_rng, restore_cuda_rng
from .imitation import MeasuredAnchor, collect_demonstration, demonstration_samples, imitation_update
from .model import MotionActorCritic, FEATURE_TRANSFORMS, DEADLINE_FEATURES, DEADLINE_CONTEXT, NETWORK_WIDTHS, NETWORK_DEPTHS, validate_network_width, validate_network_depth, normalization_definition, public_deadline_floor, export
from .objective import LearningObjective, SETTLING_MARGIN_OBJECTIVE, PROFILES, objective_definition
from .teacher_profiles import DEFAULT_TEACHER_PROFILE, TEACHER_PROFILES, teacher_profile_definition
from .rollout_retention import retained_success_rows
from .corpus_warm import load_corpus, validate_definition as validate_corpus_definition, LOSS_PROFILES as CORPUS_LOSS_PROFILES, WORLD_LOSS_PROFILE, loss_definition as corpus_loss_definition, imitation as corpus_imitation, ANCHOR_SOURCES as BASE_ANCHOR_SOURCES, anchor_definition, FullCorpusAnchor
from .directional_anchor import ANCHOR_SOURCE as DIRECTIONAL_ANCHOR_SOURCE, DirectionalPaceAnchor, definition as directional_anchor_definition
from .corpus_history import history_definition, extend_corpus, TARGET_TRANSFORM as LONG_CORPUS_TRANSFORM
from .corpus_portability import load_transfer, consumer_sources, validate_engine_runtime, METHOD_ID as CORPUS_TRANSFER_METHOD, SCOPE as CORPUS_TRANSFER_SCOPE
from .expert_data import read_json
from .td3 import ALGORITHM, ReplayBuffer, ReservoirReplayBuffer, TD3Learner, TrainingExploration, complete_episode_returns, validate_critic_warmup_steps
from .episode_replay import STRATEGY as EPISODE_REPLAY_STRATEGY, EpisodeBalancedReplayBuffer, tag_complete_episodes, definition as episode_replay_definition
from .train import ROOT, CURRICULA, atomic_json, digest, source_identity as ppo_source_identity, training_world, exclude_frozen_world, curriculum_definition, validate_online_anchor_interval


DEFAULT_DIRECTORY = ROOT / "roboproof/runs/motion/td3"
ANCHOR_SOURCES = (*BASE_ANCHOR_SOURCES, DIRECTIONAL_ANCHOR_SOURCE)


def validate_checkpoint_exposure(directory, checkpoint):
    exposure_path = Path(directory) / "exposure.json"
    if "exposureSha256" not in checkpoint or checkpoint["exposureSha256"] != (digest(exposure_path) if exposure_path.is_file() else None):
        raise ValueError("TD3 exposure differs from its checkpoint; uncommitted work cannot silently resume")


def online_anchor_definition(interval):
    return {"method": "periodic-measured-td3-retention-v1", "everyTd3Collections": interval,
            "firstRefresh": "First TD3 collection after warm-start", "worldsPerRefresh": 1,
            "worldSelection": "First training world of the selected collection",
            "teacher": "Same explicitly declared measured teacher profile and ranking",
            "buffer": "Existing512-row measured teacher anchor,16 chronological rows per successful world",
            "accounting": "Charge every actual teacher and rollout query, including interruptions; no inferred labels or refunds",
            "commit": "Append pending teacher rows only after the entire TD3 collection completes",
            "scope": "Soft training retention on all four learned heads; no inference teacher, projection or extra actor input"}


def source_identity(include_corpus_transfer=False, include_directional_anchor=False, include_corpus_history=False):
    result = ppo_source_identity()
    for name in ("roboproof/motion_learning/td3.py", "roboproof/motion_learning/td3_train.py", "roboproof/motion_learning/execution.py",
                 "roboproof/motion_learning/actor_artifact.py", "roboproof/motion_learning/episode_replay.py"):
        result[name] = hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
    name = "roboproof/motion_learning/corpus_warm.py"
    result[name] = hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
    if include_corpus_transfer:
        result.update(consumer_sources())
    if include_directional_anchor:
        name = "roboproof/motion_learning/directional_anchor.py"
        result[name] = hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
    if include_corpus_history:
        name = "roboproof/motion_learning/corpus_history.py"
        result[name] = hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
    return result


def validate_corpus_transfer(value):
    keys = {"method", "attestationSha256", "producerTorchVersion", "consumerTorchVersion", "consumerPlatform",
            "producerEngineRuntime", "consumerEngineRuntime", "engineSourceIdentityPreserved",
            "engineRuntimeParityVerified", "newNativeQueries", "scope"}
    if (not isinstance(value, dict) or set(value) != keys or value["method"] != CORPUS_TRANSFER_METHOD or
            value["scope"] != CORPUS_TRANSFER_SCOPE or not isinstance(value["attestationSha256"], str) or
            len(value["attestationSha256"]) != 64 or any(character not in "0123456789abcdef" for character in value["attestationSha256"]) or
            not isinstance(value["producerTorchVersion"], str) or not 1 <= len(value["producerTorchVersion"]) <= 128 or
            value["consumerTorchVersion"] != str(torch.__version__) or value["consumerPlatform"] not in ("Windows", "Linux", "Darwin") or
            value["engineSourceIdentityPreserved"] is not True or value["engineRuntimeParityVerified"] is not False or
            type(value["newNativeQueries"]) is not int or value["newNativeQueries"] != 0):
        raise ValueError("Exact explicit corpus transfer provenance required")
    validate_engine_runtime(value["producerEngineRuntime"])
    validate_engine_runtime(value["consumerEngineRuntime"])
    return value


def load_expert_corpus(directory, expected=None, attestation_file=None, attestation_sha256=None, consumer_identity=None):
    if (attestation_file is None) != (attestation_sha256 is None):
        raise ValueError("Explicit corpus attestation file and pinned SHA required together")
    if directory is None:
        if expected is not None:
            raise ValueError("An expert-corpus definition cannot replace its actual data path")
        if attestation_file is not None:
            raise ValueError("Explicit original expert-corpus path required for transfer")
        return None
    if attestation_file is None:
        return load_corpus(Path(directory), expected)
    corpus = load_transfer(Path(directory), attestation_file, attestation_sha256)
    validate_corpus_transfer(corpus["portability"])
    if expected is not None and corpus["definition"] != expected:
        raise ValueError("Exact transferred expert-corpus definition required")
    if consumer_identity is not None:
        receipt, receipt_sha = read_json(attestation_file)
        expected_identity = {**receipt["engineIdentity"], "runtime": corpus["portability"]["consumerEngineRuntime"]}
        if receipt_sha != attestation_sha256 or consumer_identity != expected_identity:
            raise ValueError("Active native Simulator/iraLIB must match the admitted transfer consumer")
    return corpus


def compatible_config(previous, current):
    return (all(previous.get(name) == value for name, value in current.items() if name not in ("updates", "maxSteps")) and
            previous.keys() == current.keys() and current["updates"] >= previous["updates"] and current["maxSteps"] >= previous["maxSteps"])


def critic_warmup_definition(steps):
    steps = validate_critic_warmup_steps(steps)
    return None if steps == 0 else {"method": "initial-native-critic-only-v1", "gradientSteps": steps,
        "timing": "First accepted native critic gradient steps after actor-only imitation",
        "actor": "All actor parameters, optimizer moments and target actor unchanged during the prefix; then existing global delayed cadence",
        "targets": "Existing delayed target cadence unchanged; target critics remain frozen during the critic-only prefix",
        "budget": "No extra native query or critic gradient step; fewer actor steps in the same declared training budget",
        "scope": "Training-only scheduling hypothesis, not a fitted critic-quality threshold, new inference policy or movement qualification"}


def configuration(seed, seed_count, updates, max_steps, rollout_steps, warm_start_updates, feature_transform,
                  objective, curriculum, gradient_steps, replay_capacity, exploration_std,
                  demonstration_anchor_weight, policy_regularization, deadline_pace_loss_weight, n_steps=1,
                  critic_terminal_fraction=0, replay_strategy="recent-ring-v1", exploration_profile="iid-v1",
                  pose_exploration_std=None, exploration_correlation_seconds=0, network_width=32, training_device="cpu",
                  teacher_profile=DEFAULT_TEACHER_PROFILE, network_depth=2, self_retention_weight=0, online_anchor_interval=0,
                  expert_dataset_definition=None, corpus_warm_loss_profile="uniform-action-v1", corpus_anchor_source="balanced-bank-v1",
                  expert_dataset_transfer=None, expert_history_frames=4, critic_warmup_steps=0):
    network_width = validate_network_width(network_width)
    network_depth = validate_network_depth(network_depth)
    critic_warmup = critic_warmup_definition(critic_warmup_steps)
    if (isinstance(self_retention_weight, bool) or not isinstance(self_retention_weight, (int, float)) or
            not math.isfinite(self_retention_weight) or not 0 <= self_retention_weight <= 100):
        raise ValueError("Bounded successful-rollout retention weight required")
    if training_device not in DEVICES:
        raise ValueError("Explicit supported training device required")
    if teacher_profile not in TEACHER_PROFILES:
        raise ValueError("Explicit supported measured teacher profile required")
    limits = ((seed, 0, 2**32 - 3), (seed_count, 1, 3), (updates, 1, 100), (max_steps, 1, 50000),
              (rollout_steps, 16, 1024), (warm_start_updates, 0, 16), (gradient_steps, 1, 32), (replay_capacity, 64, 2048), (n_steps, 1, 32))
    if any(isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper for value, lower, upper in limits):
        raise ValueError("Invalid bounded TD3 CPU budget")
    if (feature_transform not in FEATURE_TRANSFORMS or objective not in PROFILES or curriculum not in CURRICULA or
            policy_regularization not in ("none", "public-deadline-residual-v1") or warm_start_updates >= updates or
            replay_strategy not in ("recent-ring-v1", "uniform-reservoir-v1", EPISODE_REPLAY_STRATEGY)):
        raise ValueError("Unsupported TD3 configuration")
    for value, lower, upper in ((exploration_std, 0.001, 0.2), (demonstration_anchor_weight, 0, 100),
                                (deadline_pace_loss_weight, 0, 100), (critic_terminal_fraction, 0, 0.5)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError("Finite bounded TD3 noise/retention/prior required")
    if replay_strategy == EPISODE_REPLAY_STRATEGY and critic_terminal_fraction != 0:
        raise ValueError("Episode-balanced replay cannot combine terminal mixture sampling")
    if (demonstration_anchor_weight and not warm_start_updates) or (policy_regularization == "none" and deadline_pace_loss_weight != 1):
        raise ValueError("Active teacher anchor and declared prior required")
    refresh_interval = validate_online_anchor_interval(online_anchor_interval, warm_start_updates, demonstration_anchor_weight)
    history_view = history_definition(expert_history_frames, expert_dataset_definition)
    if expert_dataset_definition is not None:
        validate_corpus_definition(expert_dataset_definition)
        expected_transform = LONG_CORPUS_TRANSFORM if history_view is not None else expert_dataset_definition["featureTransform"]
        if not warm_start_updates or not demonstration_anchor_weight or feature_transform != expected_transform:
            raise ValueError("Expert corpus requires its matching public encoder and active warm-start/teacher anchor")
    if history_view is not None and expert_dataset_transfer is not None:
        raise ValueError("Long-history derivation requires the exact original producing runtime, not cross-runtime transfer")
    if expert_dataset_transfer is not None:
        validate_corpus_transfer(expert_dataset_transfer)
        if expert_dataset_definition is None:
            raise ValueError("Explicit transferred corpus requires its actual source-bound definition")
    if corpus_warm_loss_profile not in CORPUS_LOSS_PROFILES or (corpus_warm_loss_profile != "uniform-action-v1" and expert_dataset_definition is None):
        raise ValueError("Explicit original expert corpus required for precision warm loss")
    if corpus_anchor_source not in ANCHOR_SOURCES or (corpus_anchor_source != "balanced-bank-v1" and
            (expert_dataset_definition is None or not warm_start_updates or not demonstration_anchor_weight or refresh_interval)):
        raise ValueError("Full-corpus retention requires the actual immutable corpus, active warm/anchor and no online refresh")
    exploration = TrainingExploration(exploration_std, exploration_profile, pose_exploration_std, exploration_correlation_seconds)
    return {"seed": seed, "seedCount": seed_count, "updates": updates, "maxSteps": max_steps, "rolloutSteps": rollout_steps,
            "algorithm": ALGORITHM, "initialization": "broad-pace-v1", "featureTransform": feature_transform,
            "networkWidth": network_width, "trainingDevice": training_device,
            **({"criticWarmupSteps": critic_warmup_steps, "criticWarmupDefinition": critic_warmup} if critic_warmup is not None else {}),
            **({"expertDatasetDefinition": expert_dataset_definition} if expert_dataset_definition is not None else {}),
            **({"expertDatasetTransfer": expert_dataset_transfer} if expert_dataset_transfer is not None else {}),
            **({"expertHistoryFrames": expert_history_frames, "expertCorpusHistoryDefinition": history_view} if history_view is not None else {}),
            **({"corpusAnchorSource": corpus_anchor_source, "corpusAnchorDefinition":
                directional_anchor_definition(expert_dataset_definition) if corpus_anchor_source == DIRECTIONAL_ANCHOR_SOURCE else
                anchor_definition(corpus_anchor_source, expert_dataset_definition)}
               if corpus_anchor_source != "balanced-bank-v1" else {}),
            **({"corpusWarmLossProfile": corpus_warm_loss_profile, "corpusWarmLossDefinition": corpus_loss_definition(corpus_warm_loss_profile)}
               if corpus_warm_loss_profile != "uniform-action-v1" else {}),
            **({"networkDepth": network_depth} if network_depth != 2 else {}),
            **({"onlineAnchorInterval": refresh_interval, "onlineAnchorDefinition": online_anchor_definition(refresh_interval)}
               if refresh_interval else {}),
            **({"selfRetentionWeight": self_retention_weight,
                "selfRetentionDefinition": {"method": "measured-robust-rollout-retention-v1", "capacity": 512,
                    "rowsPerEpisode": 16, "requiredReason": "success", "contactSeconds": 0,
                    "minimumPoseMarginFraction": 0.2, "minimumDeadlineReserveSeconds": 1,
                    "labels": "Actual public encoded inputs and applied exploratory actions from complete episodes",
                    "accounting": "No additional environment queries; partial collections never commit retained rows",
                    "scope": "Separate soft training anchor; original teacher bank, inference and gates unchanged"}}
               if self_retention_weight else {}),
            "normalizationDefinition": normalization_definition(feature_transform),
            "deadlineContextDefinition": dict(DEADLINE_CONTEXT) if feature_transform in DEADLINE_FEATURES else None,
            "objective": objective, "objectiveDefinition": objective_definition(objective), "curriculum": curriculum,
            "curriculumDefinition": curriculum_definition(curriculum), "warmStartUpdates": warm_start_updates,
            "teacherSelection": "pose-deadline-margin-v2", "teacherProfile": teacher_profile,
            "teacherProfileDefinition": teacher_profile_definition(teacher_profile),
            "demonstrationAnchorWeight": demonstration_anchor_weight,
            "demonstrationAnchorLossProfile": "pose-budget-v2", "gradientSteps": gradient_steps, "replayCapacity": replay_capacity,
            "batchSize": 64, "explorationStd": exploration_std, "policyRegularization": policy_regularization, "nStepReturns": n_steps,
            "deadlinePaceLossWeight": deadline_pace_loss_weight, "criticTerminalFraction": critic_terminal_fraction,
            "replayStrategy": replay_strategy,
            **({"episodeReplayDefinition": episode_replay_definition()} if replay_strategy == EPISODE_REPLAY_STRATEGY else {}),
            "explorationProfile": exploration_profile, "poseExplorationStd": exploration.pose_standard_deviation,
            "explorationCorrelationSeconds": exploration_correlation_seconds,
            "td3Definition": {"actor": "Deterministic Beta mean, every reference axis learned; no value-only critic branch",
                "critics": f"Independent twin-Q{network_width}x{network_width} value estimators; original Simulator remains the environment" if network_depth == 2 else f"Independent twin-Q with{network_depth} hidden layers of{network_width}; original Simulator remains the environment",
                "discountFactor": 1.0, "targetTau": 0.005, "policyDelay": 2, "targetNoiseStd": 0.05, "targetNoiseClip": 0.1,
                "actorLearningRate": 0.0003, "criticLearningRate": 0.0003, "imitationLearningRate": 0.001,
                "terminal": "Success, fault and original task deadline all stop bootstrap; never cross resets",
                "returns": {"method": "complete-intrinsic-episode-n-step-v1", "steps": n_steps, "discountFactor": 1.0,
                    "scope": "Recorded behavior rewards/actions until terminal; no reset crossing, evaluator actor input or substitute physics",
                    "offPolicy": "Later behavior actions introduce multi-step off-policy bias; this is an explicit TD3 variant, not canonical one-step equivalence"},
                "replay": "Only complete original-engine collections; every actual query/exposure charged",
                "replayRetention": {"method": replay_strategy, "capacity": replay_capacity,
                    "scope": ("Uniform accepted-row reservoir with episode-length-balanced training draws, no reward/outcome filtering or new query"
                              if replay_strategy == EPISODE_REPLAY_STRATEGY else
                              "Recent ring by default; optional uniform reservoir over every accepted learning row, no reward/outcome selection or new query"),
                    "replacement": "At rowN draw integer0..N-1; replace that slot only if below capacity. Every row counted; Torch RNG/checkpoint exact.",
                    "offPolicy": ("Reservoir retains older behavior; eight-step return approximation unchanged, no deterministic-policy truth claim"
                                  if n_steps == 8 else f"Reservoir retains older behavior;{n_steps}-step returns remain an off-policy approximation, no deterministic-policy truth claim")},
                "criticSampling": episode_replay_definition() if replay_strategy == EPISODE_REPLAY_STRATEGY else {"method": "uniform-terminal-mixture-v1" if critic_terminal_fraction else "uniform-replay-v1",
                    "terminalMixtureWeight": critic_terminal_fraction, "weight": "1/(occupiedRows*rowProbability), no batch normalization or clipping",
                    "scope": "Critic objective retains uniform-row expectation before gradient clipping; actor/priors independently uniform. Terminal horizons may share real end states."},
                "exploration": "Independent Gaussian noise on all four training actions, clip to contract bounds" if exploration_profile == "iid-v1" else "IID pace and correlated OU pose noise, reset every real world; training only, clip to existing contract bounds",
                "explorationDefinition": {"method": exploration_profile, "policyDtSeconds": 0.05,
                    "paceStd": exploration_std, "poseStd": exploration.pose_standard_deviation,
                    "correlationSeconds": exploration_correlation_seconds, "reset": "Zero pose state at every real episode; no mid-episode checkpoint promise",
                    "scope": "Same four bounded reference actions; no inference noise, actor truth, wider action range or changed reference filter"},
                "inference": "Unchanged deterministic mean; no inference projection or frozen head"}}


def train_seed(environment, directory, seed, config, sources, deadline, resume=False, on_checkpoint=None, expert_dataset=None,
               expert_dataset_attestation=None, expert_dataset_attestation_sha256=None):
    expected = configuration(**{argument: config[key] for argument, key in
        (("seed", "seed"), ("seed_count", "seedCount"), ("updates", "updates"), ("max_steps", "maxSteps"),
         ("rollout_steps", "rolloutSteps"), ("warm_start_updates", "warmStartUpdates"), ("feature_transform", "featureTransform"),
         ("objective", "objective"), ("curriculum", "curriculum"), ("gradient_steps", "gradientSteps"),
         ("replay_capacity", "replayCapacity"), ("exploration_std", "explorationStd"),
         ("demonstration_anchor_weight", "demonstrationAnchorWeight"), ("policy_regularization", "policyRegularization"),
         ("deadline_pace_loss_weight", "deadlinePaceLossWeight"), ("n_steps", "nStepReturns"),
         ("critic_terminal_fraction", "criticTerminalFraction"), ("replay_strategy", "replayStrategy"),
         ("exploration_profile", "explorationProfile"), ("pose_exploration_std", "poseExplorationStd"),
         ("exploration_correlation_seconds", "explorationCorrelationSeconds"), ("network_width", "networkWidth"),
         ("training_device", "trainingDevice"), ("teacher_profile", "teacherProfile"))}, network_depth=config.get("networkDepth", 2),
         self_retention_weight=config.get("selfRetentionWeight", 0), online_anchor_interval=config.get("onlineAnchorInterval", 0),
         expert_dataset_definition=config.get("expertDatasetDefinition"), corpus_warm_loss_profile=config.get("corpusWarmLossProfile", "uniform-action-v1"),
         corpus_anchor_source=config.get("corpusAnchorSource", "balanced-bank-v1"), expert_dataset_transfer=config.get("expertDatasetTransfer"),
         expert_history_frames=config.get("expertHistoryFrames", 4), critic_warmup_steps=config.get("criticWarmupSteps", 0))
    if config != expected:
        raise ValueError("TD3 method definition mismatch")
    if (expert_dataset is None) != (config.get("expertDatasetDefinition") is None):
        raise ValueError("Explicit original expert-corpus path and definition required together")
    if (config.get("expertDatasetTransfer") is None) != (expert_dataset_attestation is None):
        raise ValueError("Declared corpus transfer and actual attestation required together")
    corpus = load_expert_corpus(expert_dataset, config.get("expertDatasetDefinition"), expert_dataset_attestation,
                                expert_dataset_attestation_sha256, environment.contract["identity"])
    if (corpus.get("portability") if corpus is not None else None) != config.get("expertDatasetTransfer"):
        raise ValueError("Actual corpus transfer must match its complete pinned declaration")
    directional = config.get("corpusAnchorSource") == DIRECTIONAL_ANCHOR_SOURCE
    long_history = config.get("expertHistoryFrames", 4) == 16
    if long_history:
        corpus = extend_corpus(expert_dataset, corpus)
        if corpus["historyView"] != config["expertCorpusHistoryDefinition"]:
            raise ValueError("Actual original public-history view must match its complete source-bound definition")
    if (config.get("expertDatasetTransfer") is not None or directional or long_history) and sources != source_identity(config.get("expertDatasetTransfer") is not None, directional, long_history):
        raise ValueError("Explicit transferred/directional TD3 consumer sources required")
    checkpoint_limit = checkpoint_size_limit(config)
    device, execution = prepare_execution(config["trainingDevice"])
    torch.manual_seed(seed)
    generator = random.Random(seed)
    actor = MotionActorCritic(config["initialization"], config["featureTransform"], value_critic=False,
                             network_width=config["networkWidth"], network_depth=config.get("networkDepth", 2)).to(device)
    learner = TD3Learner(actor, critic_warmup_steps=config.get("criticWarmupSteps", 0))
    exploration = TrainingExploration(config["explorationStd"], config["explorationProfile"],
                                      config["poseExplorationStd"], config["explorationCorrelationSeconds"])
    replay_type = (EpisodeBalancedReplayBuffer if config["replayStrategy"] == EPISODE_REPLAY_STRATEGY else
                   ReservoirReplayBuffer if config["replayStrategy"] == "uniform-reservoir-v1" else ReplayBuffer)
    replay = replay_type(len(actor.scales), config["replayCapacity"])
    anchor = MeasuredAnchor(len(actor.scales)) if config["demonstrationAnchorWeight"] else None
    anchor_type = DirectionalPaceAnchor if directional else FullCorpusAnchor
    full_anchor = (anchor_type(corpus["rows"], corpus["rowWeights"], len(actor.scales))
                   if config.get("corpusAnchorSource") in ("full-corpus-equal-world-v1", DIRECTIONAL_ANCHOR_SOURCE) else None)
    successful_anchor = MeasuredAnchor(len(actor.scales)) if config.get("selfRetentionWeight", 0) else None
    refresh_interval = config.get("onlineAnchorInterval", 0)
    history = []
    steps = episodes = updates = collection_steps = discarded_total = 0
    identity = environment.contract["identity"]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    exposure_path = directory / "exposure.json"
    if resume:
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size > checkpoint_limit:
            raise ValueError("Bounded TD3 checkpoint required")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint["schemaVersion"] != 1 or checkpoint["algorithm"] != ALGORITHM or checkpoint["identity"] != identity or
                checkpoint["sources"] != sources or checkpoint["seed"] != seed or checkpoint["torchVersion"] != str(torch.__version__) or
                checkpoint.get("execution") != execution or not compatible_config(checkpoint["config"], config) or
                not torch.equal(checkpoint["model"]["scales"], actor.scales.detach().cpu())):
            raise ValueError("TD3 checkpoint source/runtime/configuration/scale mismatch")
        validate_checkpoint_exposure(directory, checkpoint)
        actor.load_state_dict(checkpoint["model"])
        learner.restore(checkpoint["learner"])
        replay.restore(checkpoint["replay"])
        if anchor is not None:
            anchor.restore(checkpoint["measuredAnchor"])
        if successful_anchor is not None:
            successful_anchor.restore(checkpoint["successfulRolloutAnchor"])
        generator.setstate(checkpoint["worldRng"])
        torch.set_rng_state(checkpoint["torchRng"])
        restore_cuda_rng(checkpoint.get("cudaRng"), device)
        steps, episodes, updates, history, collection_steps, discarded_total = (checkpoint[key] for key in
            ("steps", "episodes", "updates", "history", "collectionSteps", "discardedCollectionStepsTotal"))
    elif checkpoint_path.exists():
        raise ValueError("Explicit resume required; never overwrite TD3 history")
    initial_path = directory / "initial-policy.json"
    if not resume:
        atomic_actor_json(initial_path, export(actor, identity, seed, 0, 0, ALGORITHM))
    exposures = json.loads(exposure_path.read_text())["cases"] if exposure_path.is_file() else []
    if corpus is not None:
        declared_exposure = [{**world, "kind": "td3-expert-corpus", "update": 0} for world in corpus["worlds"]]
        if resume:
            if [world for world in exposures if world.get("kind") == "td3-expert-corpus"] != declared_exposure:
                raise ValueError("Every original expert world, including failed labels, must remain exposed")
        else:
            exposures.extend(declared_exposure)
            atomic_json(exposure_path, {"schemaVersion": 1, "cases": exposures})

    def save():
        exported = export(actor, identity, seed, steps, updates, ALGORITHM)
        policy_name = f"policy-{updates:04d}.json"
        atomic_actor_json(directory / policy_name, exported)
        checkpoint = {"schemaVersion": 1, "algorithm": ALGORITHM, "identity": identity, "sources": sources, "config": config,
                      "execution": execution, "cudaRng": snapshot_cuda_rng(device),
                      "seed": seed, "torchVersion": str(torch.__version__), "model": actor.state_dict(), "learner": learner.state(),
                      "replay": replay.state(), "worldRng": generator.getstate(), "torchRng": torch.get_rng_state(),
                      "steps": steps, "episodes": episodes, "updates": updates, "history": history,
                      "collectionSteps": collection_steps, "discardedCollectionStepsTotal": discarded_total,
                      "exposureSha256": digest(exposure_path) if exposure_path.is_file() else None}
        if anchor is not None:
            checkpoint["measuredAnchor"] = anchor.state()
        if successful_anchor is not None:
            checkpoint["successfulRolloutAnchor"] = successful_anchor.state()
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        if temporary.stat().st_size > checkpoint_limit:
            raise ValueError("TD3 checkpoint exceeds bounded reader size")
        temporary.replace(checkpoint_path)
        summary = {"seed": seed, "steps": steps, "episodes": episodes, "updates": updates, "trained": updates > 0,
                   "collectionSteps": collection_steps, "discardedCollectionStepsTotal": discarded_total,
                   "policyFile": policy_name, "policySha256": digest(directory / policy_name), "initialSha256": digest(initial_path),
                   "actorWeightsChanged": updates > 0 and json.loads(initial_path.read_text())["layers"][-1] != exported["layers"][-1],
                   "history": history, "td3GradientUpdates": learner.gradient_updates, "td3ActorUpdates": learner.actor_updates,
                   "replayRows": replay.count, "networkWidth": config["networkWidth"], "trainingDevice": config["trainingDevice"],
                   "cudaPeakAllocatedBytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
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
        cuda_state = snapshot_cuda_rng(device)
        start_observed = observed
        rows, worlds, anchor_rows, retained_rows = [], [], [], []
        online_teacher_queries = online_teacher_decisions = 0
        warm = updates < config["warmStartUpdates"]
        corpus_warm = warm and corpus is not None
        if corpus_warm:
            rows = corpus["rows"]
            anchor_rows = corpus["anchorRows"]
        interrupted = False
        while not corpus_warm and len(rows) < config["rolloutSteps"]:
            if observed >= config["maxSteps"] or time.monotonic() >= deadline:
                interrupted = True
                break
            curriculum_offset = config.get("expertDatasetDefinition", {}).get("nativeCurriculumOffset", 0)
            world = exclude_frozen_world(training_world(generator, episodes + len(worlds) + curriculum_offset, config["curriculum"]))
            exposures.append({"seed": world["seed"], "task": world["task"], "configuration": world["configuration"],
                              "stage": world["stage"], "kind": "td3-teacher" if warm else "td3-collection", "update": updates + 1})
            atomic_json(exposure_path, {"schemaVersion": 1, "cases": exposures})
            if warm:
                demonstration = collect_demonstration(environment, actor, world, deadline, config["maxSteps"] - observed,
                                                      config["teacherSelection"], config["teacherProfile"])
                observed += demonstration["observedSteps"]
                if demonstration["interrupted"]:
                    interrupted = True
                    break
                rows.extend(demonstration["rows"])
                anchor_rows.extend(demonstration_samples(demonstration["rows"]))
                worlds.append({**world, "reason": "success" if demonstration["rows"] else "teacher-unavailable",
                               "canonicalReward": demonstration["canonicalReward"], "teacherAttempts": demonstration["attempts"],
                               "teacherSelectedPace": demonstration["selectedPace"]})
                if config["teacherProfile"] != DEFAULT_TEACHER_PROFILE:
                    worlds[-1]["teacherSelectedPlan"] = demonstration["selectedPlan"]
                continue
            online_demonstration = None
            if refresh_interval and not worlds and (updates - config["warmStartUpdates"]) % refresh_interval == 0:
                exposures.append({"seed": world["seed"], "task": world["task"], "configuration": world["configuration"],
                                  "stage": world["stage"], "kind": "td3-online-teacher", "update": updates + 1})
                atomic_json(exposure_path, {"schemaVersion": 1, "cases": exposures})
                online_demonstration = collect_demonstration(environment, actor, world, deadline, config["maxSteps"] - observed,
                                                            config["teacherSelection"], config["teacherProfile"])
                observed += online_demonstration["observedSteps"]
                if online_demonstration["interrupted"]:
                    interrupted = True
                    break
                online_teacher_queries += len(online_demonstration["attempts"])
                online_teacher_decisions += online_demonstration["observedSteps"]
                anchor_rows.extend(demonstration_samples(online_demonstration["rows"]))
            observation, reset_info = environment.reset(seed=world["seed"], task=world["task"], configuration=world["configuration"],
                options={"recordTransitions": False, "policyIdentity": {
                    "kind": "learned" if updates else "untrained", "id": "td3-training-exploration",
                    **({"sha256": summary["policySha256"]} if updates else {})}})
            actor.reset_policy_state(reset_info["referenceDurationSeconds"])
            exploration.reset()
            objective = LearningObjective(world["task"], config["objective"])
            total_reward = canonical_reward = 0
            episode_start = len(rows)
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
                    if successful_anchor is not None:
                        retained_rows.extend(retained_success_rows(rows[episode_start:], environment.report(), world["task"]))
                    worlds.append({**world, "reason": info["reason"], "reward": total_reward,
                                   "canonicalReward": canonical_reward, "terminalSettlingMarginScore": objective.terminal_margin_score})
                    if online_demonstration is not None:
                        worlds[-1].update({"onlineTeacherAttempts": online_demonstration["attempts"],
                                          "onlineTeacherSelectedPace": online_demonstration["selectedPace"],
                                          "onlineTeacherRows": len(online_demonstration["rows"]),
                                          "onlineTeacherObservedSteps": online_demonstration["observedSteps"]})
                        if config["teacherProfile"] != DEFAULT_TEACHER_PROFILE:
                            worlds[-1]["onlineTeacherSelectedPlan"] = online_demonstration["selectedPlan"]
                    break
            if interrupted:
                break
        if interrupted:
            discarded = observed - start_observed
            discarded_total += discarded
            collection_steps = observed
            generator.setstate(world_state)
            torch.set_rng_state(torch_state)
            restore_cuda_rng(cuda_state, device)
            summary = save()
            break
        if warm:
            for group in learner.actor_optimizer.param_groups:
                group["lr"] = 0.001
            metrics = (corpus_imitation(actor, learner.actor_optimizer, rows, config.get("corpusWarmLossProfile", "uniform-action-v1"),
                                       **({"row_weights": corpus["rowWeights"]} if config.get("corpusWarmLossProfile") == WORLD_LOSS_PROFILE else {}))
                       if corpus_warm else imitation_update(actor, learner.actor_optimizer, rows))
            if anchor is not None:
                anchor.append(anchor_rows)
            learner.sync_actor_target()
        else:
            if anchor is not None:
                anchor.append(anchor_rows)
            if successful_anchor is not None:
                successful_anchor.append(retained_rows)
            for group in learner.actor_optimizer.param_groups:
                group["lr"] = 0.0003
            replay_rows = tag_complete_episodes(rows) if config["replayStrategy"] == EPISODE_REPLAY_STRATEGY else rows
            replay.append(complete_episode_returns(replay_rows, config["nStepReturns"], learner.gamma))
            results = []
            for _ in range(config["gradientSteps"]):
                batch_size = min(config["batchSize"], replay.count)
                batch = replay.sample(batch_size, config["criticTerminalFraction"])
                actor_batch = replay.sample(batch_size) if config["criticTerminalFraction"] else None
                results.append(learner.update(batch, full_anchor if full_anchor is not None else anchor, config["demonstrationAnchorWeight"],
                    config["demonstrationAnchorLossProfile"], config["policyRegularization"],
                    config["deadlinePaceLossWeight"], actor_batch,
                    **({"successful_anchor": successful_anchor, "self_retention_weight": config["selfRetentionWeight"]}
                       if successful_anchor is not None and len(successful_anchor.features) else {})))
            actor_results = [result for result in results if result["actorUpdated"]]
            metrics = {"phase": "td3", "meanCriticLoss": sum(result["criticLoss"] for result in results) / len(results),
                       "meanActorLoss": sum(result["actorLoss"] for result in actor_results) / len(actor_results) if actor_results else None,
                       "gradientUpdates": learner.gradient_updates, "actorUpdates": learner.actor_updates,
                       "actorStepsThisCollection": len(actor_results), "replayRows": replay.count, "nStepReturns": config["nStepReturns"],
                       "criticTerminalFraction": config["criticTerminalFraction"],
                       "replayStrategy": config["replayStrategy"], "replaySeenRows": getattr(replay, "seen_rows", None),
                       "maximumCriticGradientNorm": max(result["criticGradientNorm"] for result in results),
                       "maximumActorGradientNorm": max((result["actorGradientNorm"] for result in actor_results), default=None),
                       **({"nativeAnchorSource": config["corpusAnchorSource"], "nativeAnchorRows": len(full_anchor.features),
                           "nativeAnchorBatchRows": min(32, len(full_anchor.features))} if full_anchor is not None else {})}
        steps += len(rows)
        episodes += len(worlds)
        updates += 1
        collection_steps = observed
        history.append({**metrics, "worlds": worlds, "actualDecisionsThisCollection": observed - start_observed,
                        "acceptedRows": len(rows), "explorationStd": config["explorationStd"],
                        **({"expertDatasetWarm": corpus_warm, "expertDatasetRowsUsed": len(rows) if corpus_warm else 0,
                            "expertDatasetWorldCount": config["expertDatasetDefinition"]["worldCount"]}
                           if corpus is not None else {}),
                        **({"onlineTeacherQueries": online_teacher_queries, "onlineTeacherDecisions": online_teacher_decisions,
                            "onlineAnchorRowsAdded": len(anchor_rows), "onlineAnchorRows": len(anchor.features)}
                           if refresh_interval else {}),
                        **({"successfulRolloutRowsAdded": len(retained_rows),
                            "successfulRolloutAnchorRows": len(successful_anchor.features)} if successful_anchor is not None else {})})
        summary = save()
    return {**summary, "budgetReached": updates < config["updates"], "discardedUncommittedSteps": discarded}


def run(directory=DEFAULT_DIRECTORY, seed=201, seed_count=1, updates=64, max_steps=50000, rollout_steps=256,
        max_seconds=900, resume=None, warm_start_updates=8, feature_transform="deadline-context-v1",
        objective=SETTLING_MARGIN_OBJECTIVE, curriculum="mixed-full-reach-v5", gradient_steps=32, replay_capacity=1024,
        exploration_std=0.05, demonstration_anchor_weight=10, policy_regularization="public-deadline-residual-v1",
        deadline_pace_loss_weight=40, n_steps=1, critic_terminal_fraction=0, replay_strategy="recent-ring-v1",
        exploration_profile="iid-v1", pose_exploration_std=None, exploration_correlation_seconds=0,
        network_width=32, training_device="cpu", teacher_profile=DEFAULT_TEACHER_PROFILE, network_depth=2, self_retention_weight=0,
        online_anchor_interval=0, expert_dataset_definition=None, expert_dataset=None, corpus_warm_loss_profile="uniform-action-v1",
        corpus_anchor_source="balanced-bank-v1", expert_dataset_transfer=None,
        expert_dataset_attestation=None, expert_dataset_attestation_sha256=None, expert_history_frames=4, critic_warmup_steps=0):
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, int) or not 1 <= max_seconds <= 900:
        raise ValueError("Bounded TD3 wall-clock budget required")
    corpus = load_expert_corpus(expert_dataset, expert_dataset_definition, expert_dataset_attestation, expert_dataset_attestation_sha256)
    actual_transfer = corpus.get("portability") if corpus is not None else None
    if expert_dataset_transfer is not None and expert_dataset_transfer != actual_transfer:
        raise ValueError("Actual transferred data cannot be replaced by a caller declaration")
    expert_dataset_transfer = actual_transfer
    expert_dataset_definition = corpus["definition"] if corpus is not None else None
    config = configuration(seed, seed_count, updates, max_steps, rollout_steps, warm_start_updates, feature_transform,
                           objective, curriculum, gradient_steps, replay_capacity, exploration_std,
                           demonstration_anchor_weight, policy_regularization, deadline_pace_loss_weight, n_steps,
                           critic_terminal_fraction, replay_strategy, exploration_profile,
                           pose_exploration_std, exploration_correlation_seconds, network_width, training_device, teacher_profile, network_depth,
                           self_retention_weight, online_anchor_interval, expert_dataset_definition, corpus_warm_loss_profile, corpus_anchor_source,
                           expert_dataset_transfer, expert_history_frames, critic_warmup_steps)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device, execution = prepare_execution(training_device)
    directory = Path(directory).resolve()
    run_id = resume or str(uuid.uuid4())
    if str(uuid.UUID(run_id)) != run_id:
        raise ValueError("Valid explicit TD3 run ID required")
    run_directory = directory / run_id
    manifest_path = run_directory / "run.json"
    sources = source_identity(expert_dataset_transfer is not None, corpus_anchor_source == DIRECTIONAL_ANCHOR_SOURCE, expert_history_frames == 16)
    if resume:
        previous = json.loads(manifest_path.read_text())
        if previous["sources"] != sources or previous.get("execution") != execution or not compatible_config(previous["config"], config):
            raise ValueError("TD3 run source/configuration mismatch")
        for offset in range(seed_count):
            model_directory = run_directory / f"seed-{seed + offset}"
            checkpoint_path = model_directory / "checkpoint.pt"
            if checkpoint_path.is_file():
                if checkpoint_path.stat().st_size > checkpoint_size_limit(config):
                    raise ValueError("Bounded TD3 checkpoint required")
                validate_checkpoint_exposure(model_directory, torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    elif run_directory.exists():
        raise ValueError("Never overwrite an existing TD3 run")
    manifest = {"schemaVersion": 1, "runId": run_id, "config": config, "sources": sources, "status": "running",
                "device": training_device, "execution": execution, "torch": str(torch.__version__),
                "models": previous["models"] if resume else [], "learnedImprovementVerified": False,
                "scope": "Experimental deterministic reference learner; original Simulator/iraLIB only, no obstacle/game/AMD/hardware claim"}
    manifest_published = False

    def publish():
        nonlocal manifest_published
        atomic_json(manifest_path, manifest)
        atomic_json(directory / "latest.json", {"schemaVersion": 1, "runId": run_id})
        manifest_published = True

    deadline = time.monotonic() + max_seconds
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if not resume:
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
                summary = train_seed(environment, model_directory, seed + offset, config, sources, deadline, bool(existing), checkpoint_saved,
                                     expert_dataset=expert_dataset, expert_dataset_attestation=expert_dataset_attestation,
                                     expert_dataset_attestation_sha256=expert_dataset_attestation_sha256)
                manifest["models"][-1] = summary
                publish()
        manifest["status"] = "completed" if all(not row["budgetReached"] for row in manifest["models"]) else "budget-stopped"
        publish()
    except BaseException:
        if not resume or manifest_published:
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
    parser.add_argument("--critic-warmup-steps", type=int, default=0, help="Initial native critic-only gradient steps, not extra environment steps (0..512)")
    parser.add_argument("--replay-strategy", choices=("recent-ring-v1", "uniform-reservoir-v1", EPISODE_REPLAY_STRATEGY), default="recent-ring-v1")
    parser.add_argument("--exploration-profile", choices=("iid-v1", "ou-pose-v1"), default="iid-v1")
    parser.add_argument("--pose-exploration-std", type=float, default=None)
    parser.add_argument("--exploration-correlation-seconds", type=float, default=0)
    parser.add_argument("--network-width", type=int, choices=NETWORK_WIDTHS, default=32)
    parser.add_argument("--network-depth", type=int, choices=NETWORK_DEPTHS, default=2)
    parser.add_argument("--self-retention-weight", type=float, default=0)
    parser.add_argument("--online-anchor-interval", type=int, default=0)
    parser.add_argument("--expert-dataset", type=Path)
    parser.add_argument("--expert-history-frames", type=int, choices=(4, 16), default=4)
    parser.add_argument("--expert-dataset-attestation", type=Path)
    parser.add_argument("--expert-dataset-attestation-sha256")
    parser.add_argument("--corpus-warm-loss-profile", choices=CORPUS_LOSS_PROFILES, default="uniform-action-v1")
    parser.add_argument("--corpus-anchor-source", choices=ANCHOR_SOURCES, default="balanced-bank-v1")
    parser.add_argument("--training-device", choices=DEVICES, default="cpu")
    parser.add_argument("--teacher-profile", choices=TEACHER_PROFILES, default=DEFAULT_TEACHER_PROFILE)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
