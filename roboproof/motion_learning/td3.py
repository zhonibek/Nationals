import copy
import math

import torch
from torch import nn

from .model import control_prior_loss


ALGORITHM = "td3-beta-mean-reference-v1"


class TrainingExploration:
    def __init__(self, standard_deviation=0.05, profile="iid-v1", pose_standard_deviation=None, correlation_seconds=0):
        pose_standard_deviation = standard_deviation if pose_standard_deviation is None else pose_standard_deviation
        values = (standard_deviation, pose_standard_deviation, correlation_seconds)
        if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values) or
                not 0.001 <= standard_deviation <= 0.2 or not 0.001 <= pose_standard_deviation <= 0.2 or
                profile not in ("iid-v1", "ou-pose-v1") or
                (profile == "iid-v1" and (pose_standard_deviation != standard_deviation or correlation_seconds != 0)) or
                (profile == "ou-pose-v1" and not 0.05 <= correlation_seconds <= 2)):
            raise ValueError("Explicit bounded training exploration profile required")
        self.standard_deviation = standard_deviation
        self.pose_standard_deviation = pose_standard_deviation
        self.profile = profile
        self.correlation_seconds = correlation_seconds
        self.pose_state = torch.zeros(3, dtype=torch.float32)

    def reset(self):
        self.pose_state.zero_()

    def sample(self):
        draw = torch.randn(4)
        if self.profile == "iid-v1":
            return draw * self.standard_deviation
        decay = math.exp(-0.05 / self.correlation_seconds)
        self.pose_state = decay * self.pose_state + self.pose_standard_deviation * math.sqrt(1 - decay ** 2) * draw[1:]
        return torch.cat((draw[:1] * self.standard_deviation, self.pose_state))


def complete_episode_returns(rows, n_steps=1, gamma=1.0):
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or not 1 <= n_steps <= 32:
        raise ValueError("Bounded n-step return length from1 to32 required")
    if not rows or rows[-1].get("terminal") is not True:
        raise ValueError("Complete intrinsic-terminal episode collections required")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("Finite bounded return discount required")
    if n_steps == 1:
        return rows
    packed = []
    for index, row in enumerate(rows):
        reward = 0.0
        final_index = index
        for following in range(index, min(len(rows), index + n_steps)):
            current = rows[following]
            if not isinstance(current.get("terminal"), bool) or not math.isfinite(current["reward"]):
                raise ValueError("Finite measured complete-episode rewards required")
            reward += gamma ** (following - index) * current["reward"]
            final_index = following
            if current["terminal"]:
                break
        final = rows[final_index]
        packed.append({**row, "reward": reward, "nextModelInput": final["nextModelInput"],
                       "terminal": final["terminal"], "horizon": final_index - index + 1})
    return packed


def mean_action(actor, observations):
    distribution, _ = actor(observations, encoded=True)
    return distribution.mean * 2 - 1


class ReplayBuffer:
    def __init__(self, feature_size, capacity=1024):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 64 <= capacity <= 2048:
            raise ValueError("Replay capacity from64 to2048 required")
        self.feature_size = feature_size
        self.capacity = capacity
        self.count = self.cursor = 0
        self.observations = torch.zeros((capacity, feature_size), dtype=torch.float32)
        self.next_observations = torch.zeros_like(self.observations)
        self.actions = torch.zeros((capacity, 4), dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.terminals = torch.zeros(capacity, dtype=torch.bool)
        self.required_paces = torch.zeros(capacity, dtype=torch.float32)
        self.horizons = torch.zeros(capacity, dtype=torch.int64)

    def _prepare(self, rows):
        prepared = []
        for row in rows:
            observation = torch.as_tensor(row["modelInput"], dtype=torch.float32)
            following = torch.as_tensor(row["nextModelInput"], dtype=torch.float32)
            action = torch.as_tensor(row["action"], dtype=torch.float32)
            reward, pace, terminal = row["reward"], row["requiredPace"], row["terminal"]
            horizon = row.get("horizon", 1)
            if (observation.shape != (self.feature_size,) or following.shape != observation.shape or action.shape != (4,) or
                    not torch.isfinite(observation).all() or not torch.isfinite(following).all() or
                    not torch.isfinite(action).all() or torch.any(action.abs() > 1) or
                    isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward) or
                    isinstance(pace, bool) or not isinstance(pace, (int, float)) or not 0.25 <= pace <= 1 or not isinstance(terminal, bool)):
                raise ValueError("Finite real transition and bounded actions/pace required")
            if abs(reward) > 1e6 or isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 32:
                raise ValueError("Bounded finite training reward required")
            prepared.append((observation, following, action, reward, pace, terminal, horizon))
        return prepared

    def append(self, rows):
        prepared = self._prepare(rows)
        for observation, following, action, reward, pace, terminal, horizon in prepared:
            self.observations[self.cursor] = observation
            self.next_observations[self.cursor] = following
            self.actions[self.cursor] = action
            self.rewards[self.cursor] = reward
            self.terminals[self.cursor] = terminal
            self.required_paces[self.cursor] = pace
            self.horizons[self.cursor] = horizon
            self.cursor = (self.cursor + 1) % self.capacity
            self.count = min(self.capacity, self.count + 1)

    def sampling_probabilities(self, terminal_fraction):
        if (isinstance(terminal_fraction, bool) or not isinstance(terminal_fraction, (int, float)) or
                not math.isfinite(terminal_fraction) or not 0 <= terminal_fraction <= 0.5 or not self.count):
            raise ValueError("Occupied replay and bounded terminal mixture required")
        probabilities = torch.full((self.count,), 1 / self.count, dtype=torch.float64)
        terminal_count = int(self.terminals[:self.count].sum())
        if terminal_fraction and terminal_count:
            probabilities.mul_(1 - terminal_fraction)
            probabilities[self.terminals[:self.count]] += terminal_fraction / terminal_count
        return probabilities / probabilities.sum()

    def sample(self, batch_size, terminal_fraction=0):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= self.count:
            raise ValueError("Bounded replay batch required")
        if (isinstance(terminal_fraction, bool) or not isinstance(terminal_fraction, (int, float)) or
                not math.isfinite(terminal_fraction) or not 0 <= terminal_fraction <= 0.5):
            raise ValueError("Bounded critic terminal mixture required")
        if terminal_fraction:
            probabilities = self.sampling_probabilities(terminal_fraction)
            indices = torch.multinomial(probabilities, batch_size, replacement=True)
        else:
            indices = torch.randint(self.count, (batch_size,))
        result = {name: getattr(self, name)[indices] for name in
                  ("observations", "next_observations", "actions", "rewards", "terminals", "required_paces", "horizons")}
        if terminal_fraction:
            result["sampling_probabilities"] = probabilities[indices]
            result["importance_weights"] = (1 / (self.count * probabilities[indices])).to(self.rewards.dtype)
        return result

    def state(self):
        return {"schemaVersion": 2, "capacity": self.capacity, "featureSize": self.feature_size,
                "count": self.count, "cursor": self.cursor,
                **{name: getattr(self, name).clone() for name in
                   ("observations", "next_observations", "actions", "rewards", "terminals", "required_paces", "horizons")}}

    def restore(self, state):
        if (not isinstance(state, dict) or state.get("schemaVersion") != 2 or state.get("capacity") != self.capacity or
                state.get("featureSize") != self.feature_size or isinstance(state.get("count"), bool) or
                not isinstance(state.get("count"), int) or not 0 <= state["count"] <= self.capacity or
                isinstance(state.get("cursor"), bool) or not isinstance(state.get("cursor"), int) or
                not 0 <= state["cursor"] < self.capacity or (state["count"] < self.capacity and state["cursor"] != state["count"])):
            raise ValueError("Versioned bounded replay cursor required")
        for name in ("observations", "next_observations", "actions", "rewards", "terminals", "required_paces", "horizons"):
            value = state.get(name)
            expected = getattr(self, name)
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype or
                    not torch.isfinite(value).all()):
                raise ValueError("Finite matching replay tensors required")
        if (torch.any(state["actions"].abs() > 1) or torch.any(state["required_paces"][:state["count"]] < 0.25) or
                torch.any(state["required_paces"][:state["count"]] > 1) or
                torch.any(state["horizons"][:state["count"]] < 1) or torch.any(state["horizons"][:state["count"]] > 32)):
            raise ValueError("Bounded replay actions and public pace required")
        self.count, self.cursor = state["count"], state["cursor"]
        for name in ("observations", "next_observations", "actions", "rewards", "terminals", "required_paces", "horizons"):
            getattr(self, name).copy_(state[name])


class ReservoirReplayBuffer(ReplayBuffer):
    def __init__(self, feature_size, capacity=1024):
        super().__init__(feature_size, capacity)
        self.seen_rows = 0

    def append(self, rows):
        prepared = self._prepare(rows)
        if self.seen_rows + len(prepared) > 50000:
            raise ValueError("Bounded actual accepted reservoir row count required")
        for observation, following, action, reward, pace, terminal, horizon in prepared:
            self.seen_rows += 1
            if self.count < self.capacity:
                selected = self.count
                self.count += 1
            else:
                selected = int(torch.randint(self.seen_rows, (1,)))
            self.cursor = self.seen_rows % self.capacity
            if selected >= self.capacity:
                continue
            self.observations[selected] = observation
            self.next_observations[selected] = following
            self.actions[selected] = action
            self.rewards[selected] = reward
            self.required_paces[selected] = pace
            self.terminals[selected] = terminal
            self.horizons[selected] = horizon

    def state(self):
        return {**super().state(), "schemaVersion": 3, "strategy": "uniform-reservoir-v1", "seenRows": self.seen_rows}

    def restore(self, state):
        if (not isinstance(state, dict) or state.get("schemaVersion") != 3 or state.get("strategy") != "uniform-reservoir-v1" or
                isinstance(state.get("seenRows"), bool) or not isinstance(state.get("seenRows"), int) or
                not 0 <= state["seenRows"] <= 50000 or state.get("count") != min(self.capacity, state["seenRows"]) or
                state.get("cursor") != state["seenRows"] % self.capacity):
            raise ValueError("Versioned bounded reservoir method/count/cursor required")
        super().restore({**state, "schemaVersion": 2})
        self.seen_rows = state["seenRows"]


class TwinQ(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.register_buffer("scales", scales.detach().clone())
        inputs = len(scales) + 4
        self.first = nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.second = nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, observations, actions):
        inputs = torch.cat((torch.clamp(observations / self.scales, -10, 10), actions), dim=-1)
        return self.first(inputs).squeeze(-1), self.second(inputs).squeeze(-1)


def clipped_target(target_actor, target_critics, batch, gamma, noise_std, noise_clip):
    with torch.no_grad():
        noise = (torch.randn_like(batch["actions"]) * noise_std).clamp(-noise_clip, noise_clip)
        actions = (mean_action(target_actor, batch["next_observations"]) + noise).clamp(-1, 1)
        first, second = target_critics(batch["next_observations"], actions)
        discount = torch.pow(gamma, batch["horizons"].float()) if "horizons" in batch else gamma
        values = batch["rewards"] + discount * (~batch["terminals"]).float() * torch.minimum(first, second)
        if not torch.isfinite(values).all():
            raise RuntimeError("Nonfinite TD3 target; no checkpoint published")
        return values, actions


class TD3Learner:
    def __init__(self, actor, actor_rate=0.0003, critic_rate=0.0003, gamma=1.0, target_tau=0.005,
                 policy_delay=2, target_noise_std=0.05, target_noise_clip=0.1):
        values = (actor_rate, critic_rate, gamma, target_tau, target_noise_std, target_noise_clip)
        if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values) or
                not 0 < actor_rate <= 0.001 or not 0 < critic_rate <= 0.001 or not 0 < gamma <= 1 or
                not 0 < target_tau <= 1 or not 0 <= target_noise_std <= 0.2 or not 0 <= target_noise_clip <= 0.5 or
                isinstance(policy_delay, bool) or not isinstance(policy_delay, int) or not 1 <= policy_delay <= 4 or actor.critic is not None):
            raise ValueError("Bounded deterministic actor/twin-Q configuration required")
        self.actor = actor
        self.target_actor = copy.deepcopy(actor).requires_grad_(False)
        self.critics = TwinQ(actor.scales)
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_rate)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=critic_rate)
        self.gamma, self.target_tau, self.policy_delay = gamma, target_tau, policy_delay
        self.target_noise_std, self.target_noise_clip = target_noise_std, target_noise_clip
        self.gradient_updates = self.actor_updates = 0

    def sync_actor_target(self):
        self.target_actor.load_state_dict(self.actor.state_dict())

    def state(self):
        return {"schemaVersion": 1, "targetActor": self.target_actor.state_dict(), "critics": self.critics.state_dict(),
                "targetCritics": self.target_critics.state_dict(), "actorOptimizer": self.actor_optimizer.state_dict(),
                "criticOptimizer": self.critic_optimizer.state_dict(), "gradientUpdates": self.gradient_updates,
                "actorUpdates": self.actor_updates}

    def restore(self, state):
        if (not isinstance(state, dict) or state.get("schemaVersion") != 1 or
                isinstance(state.get("gradientUpdates"), bool) or not isinstance(state.get("gradientUpdates"), int) or
                not 0 <= state["gradientUpdates"] <= 3200 or
                isinstance(state.get("actorUpdates"), bool) or not isinstance(state.get("actorUpdates"), int) or
                state.get("actorUpdates") != state["gradientUpdates"] // self.policy_delay):
            raise ValueError("Bounded TD3 update counters required")
        for name in ("targetActor", "critics", "targetCritics"):
            values = state.get(name)
            expected = self.target_actor.state_dict() if name == "targetActor" else self.critics.state_dict()
            if (not isinstance(values, dict) or values.keys() != expected.keys() or
                    any(not isinstance(values[key], torch.Tensor) or values[key].shape != tensor.shape or
                        values[key].dtype != tensor.dtype or not torch.isfinite(values[key]).all() for key, tensor in expected.items()) or
                    not torch.equal(values["scales"], self.actor.scales)):
                raise ValueError("Finite TD3 model states and identical sensor scales required")
        self.target_actor.load_state_dict(state["targetActor"])
        self.critics.load_state_dict(state["critics"])
        self.target_critics.load_state_dict(state["targetCritics"])
        self.actor_optimizer.load_state_dict(state["actorOptimizer"])
        self.critic_optimizer.load_state_dict(state["criticOptimizer"])
        self.gradient_updates, self.actor_updates = state["gradientUpdates"], state["actorUpdates"]

    def update(self, batch, measured_anchor=None, anchor_weight=0, anchor_profile="uniform-action-v1",
               policy_regularization="none", pace_weight=1, actor_batch=None):
        if policy_regularization not in ("none", "public-deadline-residual-v1"):
            raise ValueError("Declared TD3 control prior required")
        if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100
                for value in (anchor_weight, pace_weight)) or (anchor_weight and measured_anchor is None) or
                (policy_regularization == "none" and pace_weight != 1)):
            raise ValueError("Finite active TD3 anchor/prior weights required")
        weights = batch.get("importance_weights")
        if weights is not None and (not isinstance(weights, torch.Tensor) or weights.shape != batch["rewards"].shape or
                weights.dtype != batch["rewards"].dtype or not torch.isfinite(weights).all() or
                torch.any(weights <= 0) or torch.any(weights > 2) or actor_batch is None):
            raise ValueError("Finite inverse-probability critic weights and independent uniform actor batch required")
        if actor_batch is not None and "importance_weights" in actor_batch:
            raise ValueError("Actor and its priors require independent uniform replay")
        actor_data = batch if actor_batch is None else actor_batch
        targets, target_actions = clipped_target(self.target_actor, self.target_critics, batch,
            self.gamma, self.target_noise_std, self.target_noise_clip)
        first, second = self.critics(batch["observations"], batch["actions"])
        if weights is None:
            critic_loss = (first - targets).square().mean() + (second - targets).square().mean()
        else:
            critic_loss = (weights * (first - targets).square()).mean() + (weights * (second - targets).square()).mean()
        if not torch.isfinite(critic_loss):
            raise RuntimeError("Nonfinite twin-Q loss; no checkpoint published")
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_norm = nn.utils.clip_grad_norm_(self.critics.parameters(), 0.5, error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.gradient_updates += 1
        actor_updated = self.gradient_updates % self.policy_delay == 0
        result = {"criticLoss": float(critic_loss.detach()), "criticGradientNorm": float(critic_norm),
                  "gradientUpdates": self.gradient_updates, "actorUpdated": actor_updated,
                  "meanTarget": float(targets.mean()), "maximumAbsoluteTargetAction": float(target_actions.abs().max())}
        if actor_updated:
            self.critics.requires_grad_(False)
            try:
                distribution, _ = self.actor(actor_data["observations"], encoded=True)
                actor_loss = -self.critics(actor_data["observations"], distribution.mean * 2 - 1)[0].mean()
                pace_loss = residual_kl = torch.tensor(0.0)
                if policy_regularization != "none":
                    pace_loss, residual_kl = control_prior_loss(distribution, actor_data["required_paces"])
                    actor_loss = actor_loss + pace_weight * pace_loss + 0.02 * residual_kl
                anchor_loss = measured_anchor.loss(self.actor, anchor_profile) if anchor_weight else torch.tensor(0.0)
                actor_loss = actor_loss + anchor_weight * anchor_loss
                if not torch.isfinite(actor_loss):
                    raise RuntimeError("Nonfinite mean actor loss; no checkpoint published")
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5, error_if_nonfinite=True)
                self.actor_optimizer.step()
            finally:
                self.critics.requires_grad_(True)
            self.actor_updates += 1
            with torch.no_grad():
                for target, current in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target.mul_(1 - self.target_tau).add_(current, alpha=self.target_tau)
                for target, current in zip(self.target_critics.parameters(), self.critics.parameters()):
                    target.mul_(1 - self.target_tau).add_(current, alpha=self.target_tau)
            result.update({"actorLoss": float(actor_loss.detach()), "actorGradientNorm": float(actor_norm),
                           "anchorLoss": float(anchor_loss.detach()), "paceLoss": float(pace_loss.detach())})
        return result
