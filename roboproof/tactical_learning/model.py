from __future__ import annotations

import torch
from torch import nn


FEATURE_COUNT = 23
MODEL_ID = "override-tactical-actor-critic-v2-loader"
# v3 adds seven rules-engine preview features per candidate (see roboproof/tactical_environment.js).
FEATURE_COUNT_V3 = 30
MODEL_ID_V3 = "override-tactical-mlp-v3-preview"
MAX_WIDTH = 1024
MAX_HIDDEN_LAYERS = 10


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: h + tanh(W · LayerNorm(h) + b). Keeps a deep tanh encoder trainable."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + torch.tanh(self.linear(self.norm(hidden)))


def parameter_count(feature_count: int, width: int, hidden_layers: int) -> int:
    """Parameters of TacticalActorCritic without building it (the 1024 × 10 model has about 9.5 M)."""
    count = feature_count * width + width  # input layer
    if hidden_layers == 2:
        count += width * width + width
    else:
        count += (hidden_layers - 1) * (width * width + width + 2 * width)  # residual blocks with LayerNorm
    half = width // 2
    return count + width + 1 + width * half + half + half + 1  # policy head + value head


class TacticalActorCritic(nn.Module):
    """Scores every candidate action with a shared encoder; the value head reads the mean candidate encoding.

    With two hidden layers the encoder is the original tanh MLP (state-dict keys encoder.0 / encoder.2, as in the
    exported v2 policies). With more layers, the first layer is followed by pre-norm residual blocks.
    """

    def __init__(self, feature_count: int = FEATURE_COUNT, width: int = 128, hidden_layers: int = 2):
        super().__init__()
        if (feature_count not in (FEATURE_COUNT, FEATURE_COUNT_V3) or not 16 <= width <= MAX_WIDTH
                or not 2 <= hidden_layers <= MAX_HIDDEN_LAYERS):
            raise ValueError("Unsupported tactical model dimensions")
        self.feature_count = feature_count
        self.width = width
        self.hidden_layers = hidden_layers
        self.residual = hidden_layers > 2
        if self.residual:
            self.encoder = nn.Sequential(nn.Linear(feature_count, width), nn.Tanh(),
                                         *[ResidualBlock(width) for _ in range(hidden_layers - 1)])
        else:
            self.encoder = nn.Sequential(
                nn.Linear(feature_count, width), nn.Tanh(),
                nn.Linear(width, width), nn.Tanh(),
            )
        self.policy_head = nn.Linear(width, 1)
        self.value_head = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1))

    def architecture(self) -> dict:
        return {"featureCount": self.feature_count, "width": self.width, "hiddenLayers": self.hidden_layers,
                "residual": self.residual, "parameters": sum(parameter.numel() for parameter in self.parameters())}

    def forward(self, action_features: torch.Tensor, action_mask: torch.Tensor):
        if action_features.ndim != 3 or action_features.shape[-1] != self.feature_count:
            raise ValueError("Expected batch × candidate-actions × tactical-features")
        if action_mask.shape != action_features.shape[:2] or action_mask.dtype != torch.bool:
            raise ValueError("Candidate-action mask does not match the action features")
        if torch.any(action_mask.sum(dim=1) == 0):
            raise ValueError("Every tactical state needs at least one candidate action")
        encoded = self.encoder(action_features)
        logits = self.policy_head(encoded).squeeze(-1).masked_fill(~action_mask, -1e9)
        pooled = (encoded * action_mask.unsqueeze(-1)).sum(dim=1) / action_mask.sum(dim=1, keepdim=True)
        values = self.value_head(pooled).squeeze(-1)
        return logits, values
