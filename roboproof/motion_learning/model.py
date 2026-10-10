import copy
import math

import torch
from torch import nn
from torch.distributions import Beta, kl_divergence


SCALES = [1.524, 1.524, 3.142, 2, 2, 4, 3, 3, 3, 3,
          1.524, 1.524, 1, 1, 1, 1, 1, 1, 10, 1.524, 1.524, 3.142,
          2, 2, 4, 10, 1, 0.114, 0.114, 0.1, 0.08, 0.08, 0.25, 40]
SETTLING_NORMALIZATION = "settling-normalized-deadline-v1"
SETTLING_FRAME_SCALES = {3: 0.0254, 4: 0.0254, 5: 0.0873, 10: 0.02032, 11: 0.02032, 12: 0.035}
COARSE_AUGMENTATION = "coarse-augmentation-deadline-v1"
MULTISCALE_FEATURES = "multiscale-deadline-v1"
AUGMENTED_FEATURES = (COARSE_AUGMENTATION, MULTISCALE_FEATURES)
AUGMENTATION_INDICES = tuple(SETTLING_FRAME_SCALES)
FEATURE_TRANSFORMS = ("identity-v1", "reference-frame-v1", "history-context-v1", "deadline-context-v1", "long-history-deadline-v1", SETTLING_NORMALIZATION, *AUGMENTED_FEATURES)
HISTORY_FRAMES = {"history-context-v1": 4, "deadline-context-v1": 4, "long-history-deadline-v1": 16, SETTLING_NORMALIZATION: 4,
                  COARSE_AUGMENTATION: 4, MULTISCALE_FEATURES: 4}
TEMPORAL_FEATURES = tuple(HISTORY_FRAMES)
DEADLINE_FEATURES = ("deadline-context-v1", "long-history-deadline-v1", SETTLING_NORMALIZATION, *AUGMENTED_FEATURES)
DEADLINE_CONTEXT = {"schemaVersion": 1, "settlingReserveSeconds": 2, "input": "public-reference-duration-at-reset"}


def normalization_definition(feature_transform):
    if feature_transform not in FEATURE_TRANSFORMS:
        raise ValueError("Unsupported motion feature transform")
    if feature_transform in AUGMENTED_FEATURES:
        return {"schemaVersion": 1, "method": "bounded-sensor-augmentation-v1", "historyFrames": 4,
                "baseFrameSize": 35, "augmentedFrameSize": 41, "sourceIndices": list(AUGMENTATION_INDICES),
                "units": augmentation_units(feature_transform), "transform": "tanh(reference-frame-feature/unit)",
                "baseChannels": "Unchanged field-scale reference-frame35", "addedChannelScales": [1] * 6,
                "initialContext": "Unchanged field-scale initial sensor goal context", "clip": [-10, 10],
                "scope": "Derived public sensor/reference channels only; additional169-input capacity, no evaluator truth or fitted statistics"}
    return None if feature_transform != SETTLING_NORMALIZATION else {
        "schemaVersion": 1, "method": "sensor-settling-units-v1", "historyFrames": 4,
        "frameScales": {str(index): value for index, value in SETTLING_FRAME_SCALES.items()},
        "heading": "Goal-heading sine divided by0.035; cosine retained at scale1, not a success classifier",
        "initialContext": "Unchanged field-scale initial sensor goal context", "clip": [-10, 10],
        "scope": "Fixed public sensor/reference units only; no evaluator truth, fitted statistics or new input capacity"}


def augmentation_units(feature_transform):
    if feature_transform not in AUGMENTED_FEATURES:
        raise ValueError("Augmented sensor profile required")
    return [SETTLING_FRAME_SCALES[index] if feature_transform == MULTISCALE_FEATURES else SCALES[index]
            for index in AUGMENTATION_INDICES]


def exploration_factor(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 16):
        raise ValueError("Finite exploration concentration from 1 to 16 required")
    return float(value)


def sampling_distribution(distribution, concentration=1):
    if isinstance(concentration, torch.Tensor):
        if (concentration.dtype == torch.bool or concentration.shape != distribution.mean.shape[:-1] or
                not torch.isfinite(concentration).all() or torch.any(concentration < 1) or torch.any(concentration > 16)):
            raise ValueError("Bounded per-transition exploration concentration required")
        if torch.all(concentration == 1):
            return distribution
        factor = concentration.unsqueeze(-1)
    else:
        factor = exploration_factor(concentration)
        if factor == 1:
            return distribution
    return Beta(distribution.concentration1 * factor, distribution.concentration0 * factor)


def reference_frame_features(observations):
    if observations.shape[-1] != 34:
        raise ValueError("The raw sensor contract remains 34 values")
    result = observations.clone()
    heading = observations[..., 21]
    cosine, sine = torch.cos(heading), torch.sin(heading)
    for first, second in ((3, 4), (19, 20), (22, 23), (27, 28), (30, 31)):
        right, forward = observations[..., first], observations[..., second]
        if first == 19:
            right, forward = right - observations[..., 0], forward - observations[..., 1]
        result[..., first] = right * cosine - forward * sine
        result[..., second] = right * sine + forward * cosine
    difference = heading - observations[..., 2]
    delta_cosine, delta_sine = torch.cos(difference), torch.sin(difference)
    result[..., 10] = observations[..., 10] * delta_cosine - observations[..., 11] * delta_sine
    result[..., 11] = observations[..., 10] * delta_sine + observations[..., 11] * delta_cosine
    result[..., 21] = delta_sine
    return torch.cat((result, delta_cosine.unsqueeze(-1)), dim=-1)


class MotionActorCritic(nn.Module):
    def __init__(self, initialization="conservative-v1", feature_transform="identity-v1", value_critic=True):
        super().__init__()
        if feature_transform not in FEATURE_TRANSFORMS:
            raise ValueError("Unsupported motion feature transform")
        self.feature_transform = feature_transform
        self.history_frames = HISTORY_FRAMES.get(feature_transform, 0)
        scales = list(SCALES)
        if feature_transform != "identity-v1":
            scales[21] = 1
            scales.append(1)
        if feature_transform == SETTLING_NORMALIZATION:
            for index, scale in SETTLING_FRAME_SCALES.items():
                scales[index] = scale
        if feature_transform in AUGMENTED_FEATURES:
            scales.extend([1] * 6)
        if feature_transform in TEMPORAL_FEATURES:
            scales = scales * self.history_frames + SCALES[10:14]
        if feature_transform in DEADLINE_FEATURES:
            scales.append(1)
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.body = nn.Sequential(nn.Linear(len(scales), 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.actor = nn.Linear(32, 8)
        if not isinstance(value_critic, bool):
            raise ValueError("Explicit value critic flag required")
        self.critic_body = nn.Sequential(nn.Linear(len(scales), 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()) if value_critic else None
        self.critic = nn.Linear(32, 1) if value_critic else None
        nn.init.zeros_(self.actor.weight)
        with torch.no_grad():
            if initialization == "conservative-v1":
                biases = [18, 19, 19, 19, -2, 19, 19, 19]
            elif initialization == "broad-pace-v1":
                biases = [math.log(math.expm1(3.5)), 59, 59, 59, math.log(math.expm1(0.5)), 59, 59, 59]
            else:
                raise ValueError("Unsupported actor initialization")
            self.actor.bias.copy_(torch.tensor(biases, dtype=torch.float32))
        self.initialization = initialization
        self.reset_policy_state()

    def reset_policy_state(self, public_reference_duration=None):
        self.observation_history = []
        self.initial_context = None
        self.last_input = None
        self.reference_duration = None
        if public_reference_duration is not None:
            if (isinstance(public_reference_duration, bool) or not isinstance(public_reference_duration, (int, float)) or
                    not math.isfinite(public_reference_duration) or public_reference_duration < 0):
                raise ValueError("Finite nonnegative public reference duration required")
            self.reference_duration = float(public_reference_duration)

    def encode_observation(self, observation, advance_state=True):
        raw = torch.tensor(observation, dtype=torch.float32)
        if raw.shape != (34,) or not torch.isfinite(raw).all():
            raise ValueError("Exactly 34 finite raw sensor values required")
        if self.feature_transform == "identity-v1":
            return raw
        current = reference_frame_features(raw)
        if self.feature_transform == "reference-frame-v1":
            return current
        if self.feature_transform in AUGMENTED_FEATURES:
            units = torch.tensor(augmentation_units(self.feature_transform), dtype=torch.float32)
            local = torch.tanh(current[list(AUGMENTATION_INDICES)] / units)
            current = torch.cat((current, local))
        floor = None
        if self.feature_transform in DEADLINE_FEATURES:
            if self.reference_duration is None:
                raise ValueError("Reset with public reference duration before deadline-context inference")
            floor = public_deadline_floor(self.reference_duration, observation, DEADLINE_CONTEXT["settlingReserveSeconds"])
        context = raw[10:14].clone() if self.initial_context is None else self.initial_context
        history = [*self.observation_history, current][-self.history_frames:]
        padded = [history[0]] * (self.history_frames - len(history)) + history
        if advance_state:
            self.observation_history = history
            self.initial_context = context
        inputs = (*padded, context)
        if floor is not None:
            inputs = (*inputs, torch.tensor([floor], dtype=torch.float32))
        return torch.cat(inputs)

    def forward(self, observations, encoded=False):
        if encoded:
            if observations.shape[-1] != len(self.scales):
                raise ValueError("Encoded feature count mismatch")
        elif self.feature_transform in TEMPORAL_FEATURES:
            raise ValueError("Temporal replay minibatches require encoded observations")
        elif self.feature_transform == "reference-frame-v1":
            observations = reference_frame_features(observations)
        normalized = torch.clamp(observations / self.scales, -10, 10)
        hidden = self.body(normalized)
        concentrations = torch.nn.functional.softplus(self.actor(hidden)) + 1
        distribution = Beta(concentrations[..., :4], concentrations[..., 4:])
        value = self.critic(self.critic_body(normalized)).squeeze(-1) if self.critic is not None else torch.zeros_like(concentrations[..., 0])
        return distribution, value

    def act(self, observation, deterministic=False, advance_state=True, exploration_concentration=1):
        concentration = exploration_factor(exploration_concentration)
        with torch.no_grad():
            features = self.encode_observation(observation, advance_state)
            if advance_state:
                self.last_input = features.tolist()
            distribution, value = self(features, encoded=True)
            sampling = sampling_distribution(distribution, concentration)
            unit_action = distribution.mean if deterministic else sampling.sample()
            return (unit_action * 2 - 1).tolist(), unit_action, sampling.log_prob(unit_action).sum(), value


def advantages(rewards, values, next_values, terminated, ended, gamma=0.99, decay=0.95):
    result = torch.zeros(len(rewards), dtype=torch.float32)
    accumulated = 0.0
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + gamma * next_values[index] * (not terminated[index]) - values[index]
        accumulated = delta + gamma * decay * (not ended[index]) * accumulated
        result[index] = accumulated
    return result, result + torch.tensor(values, dtype=torch.float32)


def public_deadline_floor(reference_duration, observation, settling_reserve=2.0):
    if (not math.isfinite(reference_duration) or reference_duration < 0 or
            len(observation) != 34 or not all(math.isfinite(value) for value in observation) or
            not math.isfinite(settling_reserve) or settling_reserve < 0):
        raise ValueError("Finite public reference timing and sensor observation required")
    remaining_reference = max(0.0, reference_duration - observation[25])
    available = max(0.01, observation[18] - settling_reserve)
    return max(0.25, min(1.0, remaining_reference / available))


def control_prior_loss(distribution, required_pace):
    if (required_pace.shape != distribution.mean.shape[:-1] or not torch.isfinite(required_pace).all() or
            torch.any(required_pace < 0.25) or torch.any(required_pace > 1)):
        raise ValueError("Bounded public deadline pace targets required")
    physical_pace = 0.25 + 0.75 * distribution.mean[..., 0]
    pace_loss = torch.relu(required_pace - physical_pace).square().mean()
    residual = Beta(distribution.concentration1[..., 1:], distribution.concentration0[..., 1:])
    concentration = torch.full_like(residual.concentration1, 60.0)
    residual_kl = kl_divergence(residual, Beta(concentration, concentration)).sum(-1).mean()
    return pace_loss, residual_kl


def analytic_policy_kl(previous, current):
    if (previous.concentration1.ndim < 1 or previous.concentration1.shape != current.concentration1.shape or
            previous.concentration1.shape[-1] != 4):
        raise ValueError("Matching four-action Beta distributions required")
    old = Beta(previous.concentration1.detach().double(), previous.concentration0.detach().double())
    new = Beta(current.concentration1.detach().double(), current.concentration0.detach().double())
    result = kl_divergence(old, new)
    if not torch.isfinite(result).all() or torch.any(result < -1e-8):
        raise RuntimeError("Invalid analytic policy KL; no checkpoint published")
    return result.clamp_min(0)


def validate_policy_kl_limit(value):
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or
            not 0.001 <= value <= 0.1):
        raise ValueError("Finite positive policy KL limit from 0.001 to 0.1 required")
    return float(value)


def optimizer_step_with_kl_bound(model, optimizer, observations, encoded, initial_sampling, concentrations, limit=None, backtracks=6):
    limit = validate_policy_kl_limit(limit)
    if isinstance(backtracks, bool) or not isinstance(backtracks, int) or not 0 <= backtracks <= 6:
        raise ValueError("Bounded optimizer backtracking required")
    def measure():
        with torch.no_grad():
            distribution, _ = model(observations, encoded=encoded)
            return analytic_policy_kl(initial_sampling, sampling_distribution(distribution, concentrations))
    if limit is None:
        optimizer.step()
        return measure(), {"attempts": 1, "accepted": True, "actorRate": optimizer.param_groups[0]["lr"]}
    actor_parameters = [*model.body.parameters(), *model.actor.parameters()]
    critic_parameters = [*model.critic_body.parameters(), *model.critic.parameters()]
    if (len(optimizer.param_groups) != 2 or
            len(optimizer.param_groups[0]["params"]) != len(actor_parameters) or
            len(optimizer.param_groups[1]["params"]) != len(critic_parameters) or
            {id(parameter) for parameter in optimizer.param_groups[0]["params"]} != {id(parameter) for parameter in actor_parameters} or
            {id(parameter) for parameter in optimizer.param_groups[1]["params"]} != {id(parameter) for parameter in critic_parameters}):
        raise ValueError("Separate complete actor and critic optimizer groups required")
    parameters = list(model.parameters())
    saved_parameters = [parameter.detach().clone() for parameter in parameters]
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    rates = [group["lr"] for group in optimizer.param_groups]
    if any(not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Positive finite actor and critic learning rates required")
    def restore():
        with torch.no_grad():
            for parameter, saved in zip(parameters, saved_parameters):
                parameter.copy_(saved)
        optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
    try:
        for attempt in range(backtracks + 1):
            if attempt:
                restore()
            effective_rate = rates[0] * 0.5 ** attempt
            optimizer.param_groups[0]["lr"] = effective_rate
            optimizer.step()
            divergence = measure()
            if float(divergence.sum(-1).mean()) <= limit:
                for group, rate in zip(optimizer.param_groups, rates):
                    group["lr"] = rate
                return divergence, {"attempts": attempt + 1, "accepted": True, "actorRate": effective_rate}
        restore()
        gradients = [parameter.grad for parameter in actor_parameters]
        try:
            for parameter in actor_parameters:
                parameter.grad = None
            optimizer.step()
        finally:
            for parameter, gradient in zip(actor_parameters, gradients):
                parameter.grad = gradient
        divergence = measure()
        if float(divergence.sum(-1).mean()) > limit:
            raise RuntimeError("Collected policy KL bound violated; no checkpoint published")
        return divergence, {"attempts": backtracks + 1, "accepted": False, "actorRate": None}
    except BaseException:
        restore()
        raise


def update(model, optimizer, rows, epochs=4, gamma=0.99, decay=0.95, policy_regularization="none", measured_anchor=None, anchor_weight=0,
           deadline_pace_loss_weight=1, anchor_loss_profile="uniform-action-v1", target_policy_kl=None):
    target_policy_kl = validate_policy_kl_limit(target_policy_kl)
    if policy_regularization not in ("none", "public-deadline-residual-v1"):
        raise ValueError("Unsupported policy regularization")
    if anchor_loss_profile not in ("uniform-action-v1", "pose-budget-v2") or (anchor_loss_profile != "uniform-action-v1" and not anchor_weight):
        raise ValueError("A declared active anchor loss profile is required")
    if (isinstance(deadline_pace_loss_weight, bool) or not isinstance(deadline_pace_loss_weight, (int, float)) or
            not math.isfinite(deadline_pace_loss_weight) or not 0 <= deadline_pace_loss_weight <= 100 or
            (policy_regularization == "none" and deadline_pace_loss_weight != 1)):
        raise ValueError("Bounded active deadline pacing loss weight required")
    if (isinstance(anchor_weight, bool) or not isinstance(anchor_weight, (int, float)) or
            not math.isfinite(anchor_weight) or not 0 <= anchor_weight <= 100 or
            (anchor_weight > 0 and measured_anchor is None)):
        raise ValueError("Bounded measured anchor weight and demonstrations required")
    encoded = all("modelInput" in row for row in rows)
    if model.feature_transform in TEMPORAL_FEATURES and not encoded:
        raise ValueError("Temporal PPO rows require their collected model inputs")
    observations = torch.tensor([row["modelInput"] if encoded else row["observation"] for row in rows], dtype=torch.float32)
    actions = torch.stack([row["action"] for row in rows])
    old_log_prob = torch.tensor([row["logProb"] for row in rows])
    concentrations = torch.tensor([exploration_factor(row.get("explorationConcentration", 1)) for row in rows], dtype=torch.float32)
    with torch.no_grad():
        initial_distribution, _ = model(observations, encoded=encoded)
        initial_sampling = sampling_distribution(initial_distribution, concentrations)
    advantage, returns = advantages(
        [row["reward"] for row in rows], [row["value"] for row in rows],
        [row["nextValue"] for row in rows], [row["terminated"] for row in rows],
        [row["ended"] for row in rows],
        gamma=gamma, decay=decay,
    )
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    losses = []
    gradient_norms = []
    actor_losses = []
    critic_losses = []
    entropies = []
    divergences = []
    clipped_fractions = []
    weights = []
    actor_norms = []
    critic_norms = []
    pace_losses = []
    residual_divergences = []
    anchor_losses = []
    rollout_divergences = []
    optimizer_steps = []
    required_pace = torch.tensor([row["requiredPace"] for row in rows], dtype=torch.float32) if policy_regularization != "none" else None
    actor_parameters = [*model.body.parameters(), *model.actor.parameters()]
    critic_parameters = [*model.critic_body.parameters(), *model.critic.parameters()]
    for _ in range(epochs):
        for indices in torch.randperm(len(rows)).tensor_split(math.ceil(len(rows) / 64)):
            distribution, values = model(observations[indices], encoded=encoded)
            sampling = sampling_distribution(distribution, concentrations[indices])
            log_ratio = sampling.log_prob(actions[indices]).sum(-1) - old_log_prob[indices]
            ratio = log_ratio.exp()
            actor_loss = -torch.minimum(ratio * advantage[indices],
                                        ratio.clamp(0.8, 1.2) * advantage[indices]).mean()
            critic_loss = (values - returns[indices]).square().mean()
            entropy = sampling.entropy().sum(-1).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
            pace_loss = residual_kl = torch.tensor(0.0)
            if required_pace is not None:
                pace_loss, residual_kl = control_prior_loss(distribution, required_pace[indices])
                loss = loss + deadline_pace_loss_weight * pace_loss + 0.02 * residual_kl
            anchor_loss = measured_anchor.loss(model, anchor_loss_profile) if anchor_weight else torch.tensor(0.0)
            loss = loss + anchor_weight * anchor_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite PPO loss; no checkpoint published")
            optimizer.zero_grad()
            loss.backward()
            actor_norm = nn.utils.clip_grad_norm_(actor_parameters, 0.5, error_if_nonfinite=True)
            critic_norm = nn.utils.clip_grad_norm_(critic_parameters, 0.5, error_if_nonfinite=True)
            norm = math.hypot(float(actor_norm), float(critic_norm))
            rollout_kl, step = optimizer_step_with_kl_bound(model, optimizer, observations, encoded,
                initial_sampling, concentrations, target_policy_kl)
            optimizer_steps.append(step)
            rollout_divergences.append(float(rollout_kl.sum(-1).mean()))
            losses.append(float(loss.detach()))
            gradient_norms.append(float(norm))
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(critic_loss.detach()))
            entropies.append(float(entropy.detach()))
            divergences.append(float(((ratio - 1) - log_ratio).mean().detach()))
            clipped_fractions.append(float(((ratio - 1).abs() > 0.2).float().mean().detach()))
            weights.append(len(indices))
            actor_norms.append(float(actor_norm))
            critic_norms.append(float(critic_norm))
            pace_losses.append(float(pace_loss.detach()))
            residual_divergences.append(float(residual_kl.detach()))
            anchor_losses.append(float(anchor_loss.detach()))
    with torch.no_grad():
        _, predicted = model(observations, encoded=encoded)
        variance = returns.var(unbiased=False)
        explained = float(1 - (returns - predicted).var(unbiased=False) / variance) if float(variance) > 1e-8 else None
        sampled = actions * 2 - 1
    weighted_mean = lambda values: sum(value * weight for value, weight in zip(values, weights)) / sum(weights)
    return {"meanLoss": weighted_mean(losses), "maximumGradientNorm": max(gradient_norms),
            "meanActorLoss": weighted_mean(actor_losses),
            "meanCriticLoss": weighted_mean(critic_losses),
            "meanEntropy": weighted_mean(entropies),
            "meanApproxKL": weighted_mean(divergences),
            "meanRolloutAnalyticKL": weighted_mean(rollout_divergences),
            "maximumRolloutMeanAnalyticKL": max(rollout_divergences),
            "finalRolloutMeanAnalyticKL": float(rollout_kl.sum(-1).mean()),
            "finalRolloutMaxStateAnalyticKL": float(rollout_kl.sum(-1).max()),
            "finalRolloutActionMeanAnalyticKL": rollout_kl.mean(0).tolist(),
            "targetPolicyKL": target_policy_kl,
            "optimizerActorStepsAccepted": sum(step["accepted"] for step in optimizer_steps),
            "optimizerActorStepsRejected": sum(not step["accepted"] for step in optimizer_steps),
            "optimizerProposalCount": sum(step["attempts"] for step in optimizer_steps),
            "minimumAcceptedActorRate": min((step["actorRate"] for step in optimizer_steps if step["accepted"]), default=None),
            "clipFraction": weighted_mean(clipped_fractions),
            "valueExplainedVariance": explained,
            "sampledActionMean": sampled.mean(dim=0).tolist(),
            "sampledActionStd": sampled.std(dim=0, unbiased=False).tolist(),
            "maximumActorGradientNorm": max(actor_norms), "maximumCriticGradientNorm": max(critic_norms),
            "meanDeadlinePaceLoss": weighted_mean(pace_losses), "meanResidualPriorKL": weighted_mean(residual_divergences),
            "deadlinePaceLossWeight": deadline_pace_loss_weight,
            "meanWeightedDeadlinePaceLoss": deadline_pace_loss_weight * weighted_mean(pace_losses),
            "meanExplorationConcentration": float(concentrations.mean()),
            "meanMeasuredAnchorLoss": weighted_mean(anchor_losses),
            "demonstrationAnchorLossProfile": anchor_loss_profile,
            "sampledActionBoundaryFraction": float((sampled.abs() >= 0.95).float().mean())}


def export(model, identity, seed, steps, updates, algorithm="ppo-beta-reference-v1"):
    if algorithm not in ("ppo-beta-reference-v1", "td3-beta-mean-reference-v1"):
        raise ValueError("Unsupported motion learner algorithm")
    def layer(module):
        return {"weight": module.weight.detach().tolist(), "bias": module.bias.detach().tolist()}
    return {"schemaVersion": 1, "algorithm": algorithm, "policyContractVersion": 1,
            "observationSize": 34, "actionSize": 4, "width": 32, "scales": model.scales.detach().tolist(),
            "featureTransform": model.feature_transform, "featureInputSize": len(model.scales),
            **({"historyFrames": model.history_frames, "initialContextSize": 4} if model.feature_transform in TEMPORAL_FEATURES else {}),
            **({"deadlineContext": dict(DEADLINE_CONTEXT)} if model.feature_transform in DEADLINE_FEATURES else {}),
            **({"normalizationDefinition": normalization_definition(model.feature_transform)}
               if normalization_definition(model.feature_transform) is not None else {}),
            **({"frameInputSize": 41} if model.feature_transform in AUGMENTED_FEATURES else {}),
            "layers": [layer(model.body[0]), layer(model.body[2]), layer(model.actor)],
            "identity": identity, "trainingSeed": seed, "steps": steps, "updates": updates,
            "trained": updates > 0, "initialization": model.initialization, "inference": "deterministic Beta mean; bounded reference actions",
            "normalization": "fixed engineering-unit scales; no evaluation-fitted statistics"}
