"""Separate GVF residual networks and validated export; no learner or promotion."""

import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from . import gvf_contract as contract


def _network(inputs, outputs, width, hidden_layers, squash=False):
    layers = []
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(inputs, width), nn.Tanh()))
        inputs = width
    layers.append(nn.Linear(width, outputs))
    if squash:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _tensor(value, size, scales, label):
    def contains_boolean(item):
        if isinstance(item, (list, tuple)):
            return any(contains_boolean(entry) for entry in item)
        return isinstance(item, bool)

    try:
        if contains_boolean(value):
            raise ValueError(label)
        value = torch.as_tensor(value, device=scales.device)
        if value.dtype == torch.bool or value.is_complex():
            raise ValueError(label)
        value = value.to(dtype=scales.dtype)
        if value.ndim < 1 or value.shape[-1] != size or not torch.isfinite(value).all():
            raise ValueError(label)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{label} requires finite tensors shaped (..., {size})") from error
    return value


class GvfActor(nn.Module):
    def __init__(self, width=64, hidden_layers=2):
        super().__init__()
        self.width, self.hidden_layers = contract.validate_architecture(width, hidden_layers)
        self.register_buffer("scales", torch.tensor(contract.SCALES, dtype=torch.float32))
        self.layers = _network(contract.FEATURE_SIZE, contract.ACTION_SIZE,
                               self.width, self.hidden_layers, squash=True)
        nn.init.zeros_(self.layers[-2].weight)
        nn.init.zeros_(self.layers[-2].bias)

    def encode_observation(self, observation):
        features = contract.encode_observation(observation)
        return torch.tensor(features, dtype=torch.float32, device="cpu").to(self.scales.device)

    def forward(self, observations, encoded=True):
        if not encoded:
            observations = self.encode_observation(observations)
        features = _tensor(observations, contract.FEATURE_SIZE, self.scales, "GVF features")
        return self.layers((features / self.scales).clamp(-10, 10))

    @torch.no_grad()
    def act(self, observation):
        actions = self.forward(observation, encoded=False)
        if not torch.isfinite(actions).all():
            raise ValueError("GVF actor produced nonfinite actions")
        return actions.detach().cpu().tolist()


class TwinQ(nn.Module):
    def __init__(self, width=64, hidden_layers=2):
        super().__init__()
        self.width, self.hidden_layers = contract.validate_architecture(width, hidden_layers)
        self.register_buffer("scales", torch.tensor(contract.SCALES, dtype=torch.float32))
        self.first = _network(contract.FEATURE_SIZE + contract.ACTION_SIZE, 1,
                              self.width, self.hidden_layers)
        self.second = _network(contract.FEATURE_SIZE + contract.ACTION_SIZE, 1,
                               self.width, self.hidden_layers)

    def forward(self, observations, actions):
        features = _tensor(observations, contract.FEATURE_SIZE, self.scales, "GVF features")
        actions = _tensor(actions, contract.ACTION_SIZE, self.scales, "GVF actions")
        if features.shape[:-1] != actions.shape[:-1] or (actions.abs() > 1).any():
            raise ValueError("GVF actions must match feature batch shape and lie in [-1, 1]")
        inputs = torch.cat(((features / self.scales).clamp(-10, 10), actions), dim=-1)
        return self.first(inputs), self.second(inputs)


def export(actor, identity, training_seed, steps, updates):
    if not isinstance(actor, GvfActor):
        raise ValueError("A separate GvfActor is required")
    if actor.scales.detach().cpu().tolist() != list(contract.SCALES):
        raise ValueError("GVF scales must match the fixed public contract")
    payload = contract.artifact_header(identity, training_seed, steps, updates,
                                       actor.width, actor.hidden_layers)
    payload["layers"] = [{"weight": layer.weight.detach().cpu().tolist(),
                          "bias": layer.bias.detach().cpu().tolist()}
                         for layer in actor.layers if isinstance(layer, nn.Linear)]
    return contract.validate_actor_artifact(payload, expected_identity=identity)


def atomic_actor_json(path, value):
    value = contract.validate_actor_artifact(value)
    limit = (64 if value["width"] == 512 else 8 if value["hiddenLayers"] == 2 else 24) * 1024 * 1024
    destination = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                             dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            size = 0
            for chunk in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(value):
                size += len(chunk)
                if size > limit:
                    raise ValueError(f"GVF actor JSON exceeds {limit} bytes")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
