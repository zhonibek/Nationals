import math
import time

import torch
from torch import nn


TEACHER_PACES = (0.85, 0.95, 1.0)
TEACHER_SELECTIONS = ("minimum-effort-v1", "pose-deadline-margin-v2")
ANCHOR_CAPACITY = 512
ANCHOR_ROWS_PER_WORLD = 16
ANCHOR_LOSS_PROFILES = ("uniform-action-v1", "pose-budget-v2")
POSE_BUDGET_SCALES = (1.0, 0.08 / (0.2 * 0.02032), 0.08 / (0.2 * 0.02032), 0.1 / (0.2 * 0.035))


def teacher_margins(metrics, task):
    if not isinstance(metrics, dict) or not isinstance(task, dict):
        raise ValueError("Finite measured teacher margins required")
    deadline = task.get("deadlineSeconds")
    names = ("positionErrorMeters", "headingErrorRadians", "elapsedSeconds")
    if (not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or not math.isfinite(deadline) or deadline <= 0 or
            any(not isinstance(metrics.get(name), (int, float)) or isinstance(metrics[name], bool) or
                not math.isfinite(metrics[name]) or metrics[name] < 0 for name in names)):
        raise ValueError("Finite measured teacher margins required")
    pose_margin = min(1 - metrics["positionErrorMeters"] / 0.02032, 1 - metrics["headingErrorRadians"] / 0.035)
    reserve = deadline - metrics["elapsedSeconds"]
    return {"poseMarginFraction": pose_margin, "settlingReserveSeconds": reserve,
            "marginScore": min(pose_margin / 0.2, reserve / 1.0)}


def select_demonstration(candidates, selection="minimum-effort-v1"):
    if selection not in TEACHER_SELECTIONS:
        raise ValueError("Unsupported teacher selection profile")
    successful = [candidate for candidate in candidates if candidate["reason"] == "success"]
    if not successful:
        return None
    if selection == "minimum-effort-v1":
        return min(successful, key=lambda candidate: (candidate["metrics"]["contactSeconds"],
                   candidate["metrics"]["effortProxyVAs"], candidate["metrics"]["elapsedSeconds"]))
    contact = min(candidate["metrics"]["contactSeconds"] for candidate in successful)
    admissible = [candidate for candidate in successful if candidate["metrics"]["contactSeconds"] == contact]
    robust = [candidate for candidate in admissible if candidate["marginScore"] >= 1]
    if robust:
        return min(robust, key=lambda candidate: (candidate["metrics"]["effortProxyVAs"], candidate["metrics"]["elapsedSeconds"]))
    return min(admissible, key=lambda candidate: (-candidate["marginScore"],
               candidate["metrics"]["effortProxyVAs"], candidate["metrics"]["elapsedSeconds"]))


def demonstration_samples(rows):
    if not rows:
        return []
    count = min(len(rows), ANCHOR_ROWS_PER_WORLD)
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)] if count > 1 else [0]
    return [rows[index] for index in indices]


class MeasuredAnchor:
    def __init__(self, feature_size):
        self.feature_size = feature_size
        self.features = torch.empty((0, feature_size), dtype=torch.float32)
        self.targets = torch.empty((0, 4), dtype=torch.float32)

    def restore(self, state):
        if not isinstance(state, dict) or state.get("schemaVersion") != 1:
            raise ValueError("Versioned measured anchor required")
        features, targets = state.get("features"), state.get("targets")
        if (not isinstance(features, torch.Tensor) or not isinstance(targets, torch.Tensor) or
                features.ndim != 2 or features.shape[1] != self.feature_size or
                targets.shape != (len(features), 4) or len(features) > ANCHOR_CAPACITY or
                features.dtype != torch.float32 or targets.dtype != torch.float32 or
                not torch.isfinite(features).all() or not torch.isfinite(targets).all() or torch.any(targets.abs() > 1)):
            raise ValueError("Bounded finite measured anchor required")
        self.features = features.detach().clone()
        self.targets = targets.detach().clone()

    def append(self, rows):
        if not rows:
            return
        features = torch.tensor([row["modelInput"] for row in rows], dtype=torch.float32)
        targets = torch.tensor([row["teacherAction"] for row in rows], dtype=torch.float32)
        if (features.shape[1:] != (self.feature_size,) or targets.shape != (len(rows), 4) or
                not torch.isfinite(features).all() or not torch.isfinite(targets).all() or torch.any(targets.abs() > 1)):
            raise ValueError("Finite collected anchor inputs and bounded actions required")
        self.features = torch.cat((self.features, features))[-ANCHOR_CAPACITY:]
        self.targets = torch.cat((self.targets, targets))[-ANCHOR_CAPACITY:]

    def state(self):
        return {"schemaVersion": 1, "features": self.features, "targets": self.targets}

    def loss(self, model, profile="uniform-action-v1"):
        if profile not in ANCHOR_LOSS_PROFILES:
            raise ValueError("Unsupported measured anchor loss profile")
        if not len(self.features):
            raise ValueError("A measured anchor needs successful training demonstrations")
        indices = torch.randperm(len(self.features))[:32]
        distribution, _ = model(self.features[indices], encoded=True)
        difference = distribution.mean * 2 - 1 - self.targets[indices]
        scales = torch.tensor(POSE_BUDGET_SCALES if profile == "pose-budget-v2" else (1, 1, 1, 1), dtype=torch.float32)
        return (difference * scales).square().mean()


def collect_demonstration(environment, model, world, deadline, step_budget, teacher_selection="minimum-effort-v1"):
    if isinstance(step_budget, bool) or not isinstance(step_budget, int) or step_budget < 0:
        raise ValueError("Bounded teacher step budget required")
    if teacher_selection not in TEACHER_SELECTIONS:
        raise ValueError("Unsupported teacher selection profile")
    candidates = []
    observed_steps = 0
    for pace in TEACHER_PACES:
        if time.monotonic() >= deadline or observed_steps >= step_budget:
            return {"interrupted": True, "observedSteps": observed_steps, "attempts": candidates}
        observation, reset_info = environment.reset(seed=world["seed"], task=world["task"], configuration=world["configuration"],
            options={"recordTransitions": False, "policyIdentity": {"kind": "scripted", "id": "training-teacher"}})
        model.reset_policy_state(reset_info["referenceDurationSeconds"])
        action = [2 * (pace - 0.25) / 0.75 - 1, 0, 0, 0]
        rows = []
        total_reward = 0
        while True:
            if time.monotonic() >= deadline or observed_steps >= step_budget:
                return {"interrupted": True, "observedSteps": observed_steps, "attempts": candidates}
            features = model.encode_observation(observation)
            rows.append({"modelInput": features.tolist(), "teacherAction": list(action)})
            observation, reward, terminated, truncated, info = environment.step(action)
            total_reward += reward
            observed_steps += 1
            if terminated or truncated:
                break
        report = environment.report()
        metrics = report["metrics"]
        if not all(math.isfinite(metrics[key]) and metrics[key] >= 0 for key in ("elapsedSeconds", "effortProxyVAs", "contactSeconds")):
            raise ValueError("Finite measured teacher metrics required")
        candidates.append({"pace": pace, "reason": report["reason"], "metrics": metrics,
                           "canonicalReward": total_reward, "rows": rows, **teacher_margins(metrics, world["task"])})
    selected = select_demonstration(candidates, teacher_selection)
    return {"interrupted": False, "observedSteps": observed_steps,
            "rows": selected["rows"] if selected else [],
            "selectedPace": selected["pace"] if selected else None,
            "teacherSelection": teacher_selection,
            "selectedMarginScore": selected["marginScore"] if selected else None,
            "canonicalReward": selected["canonicalReward"] if selected else candidates[-1]["canonicalReward"],
            "reason": "success" if selected else candidates[-1]["reason"],
            "attempts": [{key: value for key, value in candidate.items() if key != "rows"} for candidate in candidates]}


def imitation_update(model, optimizer, rows, epochs=32):
    if not rows or not isinstance(epochs, int) or not 1 <= epochs <= 64:
        raise ValueError("Bounded nonempty imitation minibatches required")
    features = torch.tensor([row["modelInput"] for row in rows], dtype=torch.float32)
    targets = torch.tensor([row["teacherAction"] for row in rows], dtype=torch.float32)
    if features.shape[1:] != (len(model.scales),) or targets.shape != (len(rows), 4) or not torch.isfinite(features).all() or \
            not torch.isfinite(targets).all() or torch.any(targets.abs() > 1):
        raise ValueError("Finite collected inputs and bounded teacher actions required")
    parameters = [*model.body.parameters(), *model.actor.parameters()]
    losses, weights, norms = [], [], []
    for _ in range(epochs):
        for indices in torch.randperm(len(rows)).tensor_split(math.ceil(len(rows) / 64)):
            distribution, _ = model(features[indices], encoded=True)
            loss = (distribution.mean * 2 - 1 - targets[indices]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite imitation loss; no checkpoint published")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            weights.append(len(indices))
            norms.append(float(norm))
    return {"phase": "measured-controller-imitation", "meanLoss": sum(loss * weight for loss, weight in zip(losses, weights)) / sum(weights),
            "maximumGradientNorm": max(norms), "imitationRows": len(rows), "imitationEpochs": epochs}
