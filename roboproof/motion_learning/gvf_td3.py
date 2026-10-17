"""GVF-only TD3 gradients, not a fit driver, collector, or promotion mechanism.

Gamma discounts one five-physics-tick decision. Callers supply unchanged native
rewards and bool done including ALL intrinsic deadlines/faults/truncations.
Checkpoints cover this core and Torch RNG on its unchanged device/runtime only;
actual fit integration still needs engine/identity, replay, environment, exposure,
sampler and other RNG/runtime custody. No checkpoint conversion is supported.
After any failed optimizer step, restore a previously verified checkpoint before
continuing; updates are not transactional. The contract caps updates at 100000.
"""

import copy
import os
import sys

import torch
from torch.nn import functional as functional

from . import gvf_contract as contract
from .gvf_model import GvfActor, TwinQ


def _finite(tensors, label):
    contract.require(all(torch.isfinite(tensor).all().item() for tensor in tensors),
                     "Nonfinite " + label)


class GvfTD3:
    def __init__(self, width=64, hidden_layers=2, *, device="cpu", actor_lr=3e-4,
                 critic_lr=3e-4, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5):
        width, hidden_layers = contract.validate_architecture(width, hidden_layers)
        contract.require(torch.get_default_dtype() == torch.float32, "Explicit default float32 runtime required")
        self.config = dict(actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma,
                           tau=tau, policy_noise=policy_noise, noise_clip=noise_clip)
        for name, value in self.config.items():
            lower = 0 < value if contract.finite(value) else False
            contract.require(contract.finite(value) and value <= 1 and
                             (lower if name in ("actor_lr", "critic_lr", "tau") else value >= 0),
                             "Invalid GVF TD3 hyperparameter: " + name)
        self.device = torch.device(device)
        contract.require((self.device.type == "cpu" and self.device.index is None) or
                         (self.device.type == "cuda" and self.device.index is not None),
                         "Only CPU or explicitly indexed CUDA devices are supported; no fallback")
        self.actor = GvfActor(width, hidden_layers).to(device=self.device, dtype=torch.float32)
        self.critics = TwinQ(width, hidden_layers).to(device=self.device, dtype=torch.float32)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critics_target = copy.deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=critic_lr)
        self._optimizer_groups = {name: copy.deepcopy(optimizer.state_dict()["param_groups"])
            for name, optimizer in (("actor", self.actor_optimizer), ("critics", self.critic_optimizer))}
        self.updates = self.actor_updates = 0
        self._requires_restore = False
        self._metadata = dict(schemaVersion=1, algorithm=contract.ALGORITHM,
            policyContractName=contract.CONTRACT_NAME, policyContractVersion=contract.CONTRACT_VERSION,
            observationSchema=contract.OBSERVATION_SCHEMA, observationSize=contract.OBSERVATION_SIZE,
            featureTransform=contract.FEATURE_TRANSFORM, featureInputSize=contract.FEATURE_SIZE,
            actionSize=contract.ACTION_SIZE, scales=list(contract.SCALES),
            normalization=copy.deepcopy(contract.NORMALIZATION), width=width, hiddenLayers=hidden_layers,
            activation="tanh", dtype="float32", device=str(self.device),
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda),
            config={**self.config, "policy_delay": 2, "decision_ticks": 5})

    def _models(self):
        return {name: getattr(self, name) for name in
                ("actor", "critics", "actor_target", "critics_target")}

    def _batch(self, observations, actions, rewards, next_observations, done):
        values = (observations, actions, rewards, next_observations, done)
        contract.require(all(isinstance(value, torch.Tensor) for value in values),
                         "Explicit tensor batch required")
        batch_size = observations.shape[0] if observations.ndim else 0
        shapes = ((batch_size, 36), (batch_size, 3), (batch_size,), (batch_size, 36), (batch_size,))
        contract.require(batch_size > 0 and all(tuple(value.shape) == shape and
            value.device == self.device and value.layout == torch.strided and
            value.dtype == (torch.bool if index == 4 else torch.float32)
            for index, (value, shape) in enumerate(zip(values, shapes))),
            "Matching nonempty float32 (B,36)/(B,3)/(B,) and bool done on the exact device required")
        _finite(values, "batch")
        contract.require((actions.abs() <= 1).all().item(), "Actions must lie in [-1,1]")
        return tuple(value.detach() for value in values)

    @torch.no_grad()
    def _target(self, rewards, next_observations, done):
        actions = self.actor_target(next_observations)
        noise = (torch.randn_like(actions) * self.config["policy_noise"]).clamp(
            -self.config["noise_clip"], self.config["noise_clip"])
        first, second = self.critics_target(next_observations, (actions + noise).clamp(-1, 1))
        _finite((actions, first, second), "target predictions")
        return rewards + (~done).to(rewards.dtype) * self.config["gamma"] * torch.minimum(
            first, second).squeeze(-1)

    def _step(self, loss, model, optimizer):
        _finite((loss,), "loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        contract.require(all(gradient is not None for gradient in gradients), "Missing gradients")
        _finite(gradients, "gradients")
        optimizer.step()
        _finite(model.parameters(), "updated parameters")
        _finite((value for state in optimizer.state.values() for value in state.values()
                 if isinstance(value, torch.Tensor)), "optimizer state")

    def update(self, observations, actions, rewards, next_observations, done):
        """One update; rejected before tensor work unless explicitly approved Linux."""
        if sys.platform != "linux" or os.environ.get("ROBOPROOF_APPROVED_REMOTE") != "1":
            raise RuntimeError("GVF optimization requires Linux and ROBOPROOF_APPROVED_REMOTE=1")
        contract.require(not self._requires_restore, "Failed optimization requires verified checkpoint restore before another update")
        observations, actions, rewards, next_observations, done = self._batch(
            observations, actions, rewards, next_observations, done)
        contract.require(self.updates < 100000, "GVF contract update limit reached")
        _finite((value for model in self._models().values() for value in model.state_dict().values()),
                "model state")
        try:
            return self._optimize(observations, actions, rewards, next_observations, done)
        except BaseException:
            self._requires_restore = True
            raise

    def _optimize(self, observations, actions, rewards, next_observations, done):
        target = self._target(rewards, next_observations, done)
        first, second = self.critics(observations, actions)
        critic_loss = functional.mse_loss(first.squeeze(-1), target) + functional.mse_loss(
            second.squeeze(-1), target)
        self._step(critic_loss, self.critics, self.critic_optimizer)
        self.updates += 1
        actor_loss = None
        if self.updates % 2 == 0:
            self.critics.requires_grad_(False)
            try:
                actor_loss = -self.critics(observations, self.actor(observations))[0].mean()
                self._step(actor_loss, self.actor, self.actor_optimizer)
            finally:
                self.critics.requires_grad_(True)
            self.actor_updates += 1
            with torch.no_grad():
                for source, destination in ((self.actor, self.actor_target), (self.critics, self.critics_target)):
                    for parameter, target_parameter in zip(source.parameters(), destination.parameters()):
                        target_parameter.mul_(1 - self.config["tau"]).add_(parameter, alpha=self.config["tau"])
            _finite((parameter for model in (self.actor_target, self.critics_target)
                     for parameter in model.parameters()), "updated targets")
        return dict(critic_loss=critic_loss.item(), actor_loss=None if actor_loss is None else actor_loss.item(),
                    target_mean=target.mean().item(), updates=self.updates, actor_updates=self.actor_updates)

    def state_dict(self):
        """Detached snapshot, including CPU and (if used) this CUDA device's RNG."""
        contract.require(not self._requires_restore, "Failed partial update is not a resumable checkpoint")
        contract.bounded_integer(self.updates, 100000, "updates")
        contract.bounded_integer(self.actor_updates, 50000, "actor updates")
        contract.require(self.actor_updates == self.updates // 2 and contract.same_value(
            {**self.config, "policy_delay": 2, "decision_ticks": 5}, self._metadata["config"]),
            "Inconsistent optimization counters/config")
        contract.require(all(not parameter.requires_grad and parameter.grad is None
            for model in (self.actor_target, self.critics_target) for parameter in model.parameters()),
            "Targets must remain frozen")
        state = dict(metadata=copy.deepcopy(self._metadata), updates=self.updates, actor_updates=self.actor_updates,
            models={name: model.state_dict() for name, model in self._models().items()},
            optimizers=dict(actor=self.actor_optimizer.state_dict(), critics=self.critic_optimizer.state_dict()),
            rng=dict(cpu=torch.get_rng_state(), device=None if self.device.type == "cpu" else
                     torch.cuda.get_rng_state(self.device)))
        for name, model in self._models().items():
            self._model_state(model, state["models"][name])
        for name, optimizer, count in (("actor", self.actor_optimizer, self.actor_updates),
                                       ("critics", self.critic_optimizer, self.updates)):
            self._optimizer_state(optimizer, state["optimizers"][name], count)
        return copy.deepcopy(state)

    def _model_state(self, model, state):
        expected = model.state_dict()
        contract.require(isinstance(state, dict) and set(state) == set(expected), "Exact model state required")
        for name, reference in expected.items():
            value = state[name]
            contract.require(isinstance(value, torch.Tensor) and value.shape == reference.shape and
                value.dtype == torch.float32 and value.device == self.device and
                value.layout == torch.strided, "Model tensor shape/dtype/device mismatch: " + name)
            _finite((value,), "checkpoint model")
            if name == "scales":
                contract.require(value.detach().cpu().tolist() == list(contract.SCALES), "GVF scales mismatch")

    def _optimizer_state(self, optimizer, state, count):
        expected = optimizer.state_dict()
        groups = self._optimizer_groups["actor" if optimizer is self.actor_optimizer else "critics"]
        contract.require(type(state) is dict and set(state) == {"state", "param_groups"} and
            contract.same_value(state["param_groups"], groups) and type(state["state"]) is dict,
            "Exact Adam configuration required")
        identifiers = expected["param_groups"][0]["params"]
        contract.require(set(state["state"]) == (set(identifiers) if count else set()), "Adam counters/state mismatch")
        for identifier, parameter in zip(identifiers, optimizer.param_groups[0]["params"]):
            if not count:
                continue
            item = state["state"][identifier]
            contract.require(type(item) is dict and set(item) == {"step", "exp_avg", "exp_avg_sq"},
                             "Exact Adam moments required")
            for name, value in item.items():
                contract.require(isinstance(value, torch.Tensor) and value.layout == torch.strided and
                    value.dtype == torch.float32 and value.shape == (() if name == "step" else parameter.shape) and
                    value.device == (torch.device("cpu") if name == "step" else self.device), "Adam tensor mismatch")
                _finite((value,), "Adam moments")
            contract.require(item["step"].item() == count and (item["exp_avg_sq"] >= 0).all().item(),
                             "Adam step/moments mismatch")

    def load_state_dict(self, state):
        """Strict same-device restore; validate a private copy before replacing live state."""
        contract.require(type(state) is dict and set(state) ==
            {"metadata", "updates", "actor_updates", "models", "optimizers", "rng"}, "Exact GVF TD3 state required")
        contract.require(contract.same_value(state["metadata"], self._metadata),
                         "GVF algorithm/contract/features/architecture/config/device/runtime mismatch")
        updates = contract.bounded_integer(state["updates"], 100000, "updates")
        actor_updates = contract.bounded_integer(state["actor_updates"], 50000, "actor updates")
        contract.require(actor_updates == updates // 2, "Delayed actor counter mismatch")
        state = copy.deepcopy(state)
        candidate = copy.deepcopy(self)
        models = candidate._models()
        contract.require(type(state["models"]) is dict and set(state["models"]) == set(models), "Exact models required")
        for name, model in models.items():
            candidate._model_state(model, state["models"][name])
            model.load_state_dict(state["models"][name], strict=True)
        contract.require(type(state["optimizers"]) is dict and set(state["optimizers"]) == {"actor", "critics"},
                         "Both Adam states required")
        candidate._optimizer_state(candidate.actor_optimizer, state["optimizers"]["actor"], actor_updates)
        candidate._optimizer_state(candidate.critic_optimizer, state["optimizers"]["critics"], updates)
        candidate.actor_optimizer.load_state_dict(state["optimizers"]["actor"])
        candidate.critic_optimizer.load_state_dict(state["optimizers"]["critics"])
        rng = state["rng"]
        contract.require(type(rng) is dict and set(rng) == {"cpu", "device"}, "Exact Torch RNG state required")
        contract.require(self.device.type != "cpu" or rng["device"] is None, "Unexpected device RNG")
        for device, value in ((torch.device("cpu"), rng["cpu"]), (self.device, rng["device"])):
            if device == self.device and value is None and self.device.type == "cpu":
                continue
            contract.require(isinstance(value, torch.Tensor) and value.device == torch.device("cpu") and
                value.dtype == torch.uint8 and value.ndim == 1, "Torch RNG byte vector required")
            try:
                torch.Generator(device=device).set_state(value)
            except RuntimeError as error:
                raise ValueError("Invalid Torch RNG state") from error
        candidate.actor_target.requires_grad_(False)
        candidate.critics_target.requires_grad_(False)
        candidate.updates, candidate.actor_updates = updates, actor_updates
        candidate._requires_restore = False
        self.__dict__.update(candidate.__dict__)
        torch.set_rng_state(rng["cpu"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(rng["device"], self.device)
