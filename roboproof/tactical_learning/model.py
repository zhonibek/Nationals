from __future__ import annotations

import torch
from torch import nn


FEATURE_COUNT = 23
MODEL_ID = "override-tactical-actor-critic-v2-loader"


class TacticalActorCritic(nn.Module):
    def __init__(self, feature_count: int = FEATURE_COUNT, width: int = 128):
        super().__init__()
        if feature_count != FEATURE_COUNT or width < 16 or width > 512:
            raise ValueError("Unsupported tactical model dimensions")
        self.feature_count = feature_count
        self.width = width
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
        )
        self.policy_head = nn.Linear(width, 1)
        self.value_head = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1))

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
