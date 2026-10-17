import math

import torch


FEATURE_SIZE = 145
MAX_PAIRS = 4096
ROW_KEYS = {"modelInput", "baselineAction", "alternativeAction", "returnDifference"}


def finite_vector(value, size, action=False):
    if (not isinstance(value, list) or len(value) != size or
            any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value) or
            action and any(abs(item) > 1 for item in value)):
        raise ValueError("Finite original public return-pair feature/action vectors required")


class MeasuredReturnPairs:
    def __init__(self, rows, feature_size=FEATURE_SIZE, row_weights=None):
        if type(feature_size) is not int or feature_size != FEATURE_SIZE or not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PAIRS:
            raise ValueError("Bounded original145-feature measured return pairs required")
        for row in rows:
            if not isinstance(row, dict) or set(row) != ROW_KEYS:
                raise ValueError("Exact public state, baseline/alternative actions and measured return difference required")
            finite_vector(row["modelInput"], FEATURE_SIZE)
            finite_vector(row["baselineAction"], 4, True)
            finite_vector(row["alternativeAction"], 4, True)
            difference = row["returnDifference"]
            changed = sum(first != second for first, second in zip(row["baselineAction"], row["alternativeAction"]))
            if (isinstance(difference, bool) or not isinstance(difference, (int, float)) or not math.isfinite(difference) or
                    abs(difference) > 1e6 or changed > 1 or changed == 0 and difference != 0):
                raise ValueError("Finite bounded actual single-head return difference required")
        if row_weights is None:
            row_weights = [1.0] * len(rows)
        elif isinstance(row_weights, torch.Tensor):
            if row_weights.shape != (len(rows),) or not torch.is_floating_point(row_weights):
                raise ValueError("Finite positive source row weights required")
            row_weights = row_weights.detach().cpu().tolist()
        if (not isinstance(row_weights, (list, tuple)) or len(row_weights) != len(rows) or
                any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or
                    not 0 < item <= MAX_PAIRS for item in row_weights)):
            raise ValueError("Finite positive absolute source row weights required")
        self.features = torch.tensor([row["modelInput"] for row in rows], dtype=torch.float32, device="cpu")
        self.baseline_actions = torch.tensor([row["baselineAction"] for row in rows], dtype=torch.float32, device="cpu")
        self.alternative_actions = torch.tensor([row["alternativeAction"] for row in rows], dtype=torch.float32, device="cpu")
        self.targets = torch.tensor([row["returnDifference"] for row in rows], dtype=torch.float32, device="cpu")
        self.row_weights = torch.tensor(row_weights, dtype=torch.float32, device="cpu")
        if (any(not torch.isfinite(value).all() for value in (self.features, self.baseline_actions, self.alternative_actions, self.targets, self.row_weights)) or
                torch.any(self.row_weights <= 0)):
            raise ValueError("Finite float32 measured return-pair tensors required")

    def loss(self, critics):
        scales = getattr(critics, "scales", None)
        if (not isinstance(scales, torch.Tensor) or scales.shape != (FEATURE_SIZE,) or
                not torch.is_floating_point(scales) or not torch.isfinite(scales).all() or torch.any(scales <= 0)):
            raise ValueError("Original finite matching public critic scales required")
        indices = torch.randperm(len(self.features), device="cpu")[:32]
        features = self.features[indices].to(device=scales.device, dtype=scales.dtype)
        baseline = critics(features, self.baseline_actions[indices].to(device=scales.device, dtype=scales.dtype))
        alternative = critics(features, self.alternative_actions[indices].to(device=scales.device, dtype=scales.dtype))
        for predictions in (baseline, alternative):
            if (not isinstance(predictions, (tuple, list)) or len(predictions) != 2 or
                    any(not isinstance(value, torch.Tensor) or value.shape != (len(indices),) or
                        not torch.is_floating_point(value) or not torch.isfinite(value).all() for value in predictions)):
                raise ValueError("Finite matching predictions from both original critics required")
        targets = self.targets[indices].to(device=scales.device, dtype=scales.dtype)
        weights = self.row_weights[indices].to(device=scales.device, dtype=scales.dtype)
        loss = sum((((following - previous) - targets).square() * weights).mean()
                   for previous, following in zip(baseline, alternative))
        if not torch.isfinite(loss):
            raise ValueError("Finite measured return-difference loss required")
        return loss
