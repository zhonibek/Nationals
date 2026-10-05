import torch
from torch import nn
from torch.distributions import Beta


SCALES = [1.524, 1.524, 3.142, 2, 2, 4, 3, 3, 3, 3,
          1.524, 1.524, 1, 1, 1, 1, 1, 1, 10, 1.524, 1.524, 3.142,
          2, 2, 4, 10, 1, 0.114, 0.114, 0.1, 0.08, 0.08, 0.25, 40]


class MotionActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scales", torch.tensor(SCALES, dtype=torch.float32))
        self.body = nn.Sequential(nn.Linear(34, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.actor = nn.Linear(32, 8)
        self.critic = nn.Linear(32, 1)
        nn.init.zeros_(self.actor.weight)
        with torch.no_grad():
            self.actor.bias.copy_(torch.tensor([18, 19, 19, 19, -2, 19, 19, 19], dtype=torch.float32))

    def forward(self, observations):
        hidden = self.body(torch.clamp(observations / self.scales, -10, 10))
        concentrations = torch.nn.functional.softplus(self.actor(hidden)) + 1
        distribution = Beta(concentrations[..., :4], concentrations[..., 4:])
        return distribution, self.critic(hidden).squeeze(-1)

    def act(self, observation, deterministic=False):
        with torch.no_grad():
            distribution, value = self(torch.tensor(observation, dtype=torch.float32))
            unit_action = distribution.mean if deterministic else distribution.sample()
            return (unit_action * 2 - 1).tolist(), unit_action, distribution.log_prob(unit_action).sum(), value


def advantages(rewards, values, next_values, terminated, ended, gamma=0.99, decay=0.95):
    result = torch.zeros(len(rewards), dtype=torch.float32)
    accumulated = 0.0
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + gamma * next_values[index] * (not terminated[index]) - values[index]
        accumulated = delta + gamma * decay * (not ended[index]) * accumulated
        result[index] = accumulated
    return result, result + torch.tensor(values, dtype=torch.float32)


def update(model, optimizer, rows, epochs=4):
    observations = torch.tensor([row["observation"] for row in rows], dtype=torch.float32)
    actions = torch.stack([row["action"] for row in rows])
    old_log_prob = torch.tensor([row["logProb"] for row in rows])
    advantage, returns = advantages(
        [row["reward"] for row in rows], [row["value"] for row in rows],
        [row["nextValue"] for row in rows], [row["terminated"] for row in rows],
        [row["ended"] for row in rows],
    )
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    losses = []
    gradient_norms = []
    for _ in range(epochs):
        for indices in torch.randperm(len(rows)).split(64):
            distribution, values = model(observations[indices])
            ratio = (distribution.log_prob(actions[indices]).sum(-1) - old_log_prob[indices]).exp()
            actor_loss = -torch.minimum(ratio * advantage[indices],
                                        ratio.clamp(0.8, 1.2) * advantage[indices]).mean()
            critic_loss = (values - returns[indices]).square().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.001 * distribution.entropy().sum(-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite PPO loss; no checkpoint published")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            gradient_norms.append(float(norm))
    return {"meanLoss": sum(losses) / len(losses), "maximumGradientNorm": max(gradient_norms)}


def export(model, identity, seed, steps, updates):
    def layer(module):
        return {"weight": module.weight.detach().tolist(), "bias": module.bias.detach().tolist()}
    return {"schemaVersion": 1, "algorithm": "ppo-beta-reference-v1", "policyContractVersion": 1,
            "observationSize": 34, "actionSize": 4, "width": 32, "scales": SCALES,
            "layers": [layer(model.body[0]), layer(model.body[2]), layer(model.actor)],
            "identity": identity, "trainingSeed": seed, "steps": steps, "updates": updates,
            "trained": updates > 0, "inference": "deterministic Beta mean; bounded reference actions",
            "normalization": "fixed engineering-unit scales; no evaluation-fitted statistics"}
