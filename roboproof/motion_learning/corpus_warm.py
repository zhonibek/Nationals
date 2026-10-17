import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from .expert_data import FEATURE_SIZE, FEATURE_TRANSFORM, METHOD, MAX_WORLDS, iter_demonstrations, read_json
from .imitation import ANCHOR_CAPACITY, ANCHOR_ROWS_PER_WORLD, ANCHOR_LOSS_PROFILES, POSE_BUDGET_SCALES, imitation_update


METHOD_ID = "full-measured-corpus-warm-start-v1"
WORLD_LOSS_PROFILE = "pose-budget-equal-world-v1"
LOSS_PROFILES = (*ANCHOR_LOSS_PROFILES, WORLD_LOSS_PROFILE)
ANCHOR_SOURCES = ("balanced-bank-v1", "full-corpus-equal-world-v1")


def anchor_definition(source, corpus_definition):
    if source not in ANCHOR_SOURCES:
        raise ValueError("Explicit supported corpus anchor source required")
    if source == "balanced-bank-v1":
        return None
    validate_definition(corpus_definition)
    return {"method": source, "manifestSha256": corpus_definition["manifestSha256"],
            "rows": corpus_definition["rowCount"], "labelWorlds": corpus_definition["labelWorldCount"],
            "batchRows": 32, "sampling": "Uniform full-corpus row draw without replacement per actor step",
            "rowWeight": "N/(W*L), equal expected contribution per admitted world; no minibatch renormalization",
            "lossProfile": "pose-budget-v2", "data": "All original measured public inputs/actions; no new or invented labels",
            "resume": "Rebuild immutable pool only from the exact source-bound corpus; checkpoint config and global Torch RNG bind continuation",
            "legacyBank": "Existing512-row bank remains saved for original provenance; full pool supplies native retention only",
            "scope": "Soft training loss on every action head, no inference projection, evaluator data, new input or changed physics"}


class FullCorpusAnchor:
    def __init__(self, rows, row_weights, feature_size):
        if type(feature_size) is not int or feature_size not in (FEATURE_SIZE, 565) or not rows or len(rows) > MAX_WORLDS * 202:
            raise ValueError("Bounded full original public corpus anchor required")
        self.features = torch.tensor([row["modelInput"] for row in rows], dtype=torch.float32, device="cpu")
        self.targets = torch.tensor([row["teacherAction"] for row in rows], dtype=torch.float32, device="cpu")
        self.row_weights = torch.tensor(row_weights, dtype=torch.float32, device="cpu")
        if (self.features.shape != (len(rows), feature_size) or self.targets.shape != (len(rows), 4) or
                not torch.isfinite(self.features).all() or not torch.isfinite(self.targets).all() or
                torch.any(self.targets.abs() > 1) or self.row_weights.shape != (len(rows),) or
                not torch.isfinite(self.row_weights).all() or torch.any(self.row_weights <= 0) or torch.any(self.row_weights > 202)):
            raise ValueError("Finite bounded full-corpus anchor inputs/actions/weights required")

    def loss(self, model, profile="pose-budget-v2"):
        if profile != "pose-budget-v2":
            raise ValueError("Full-corpus anchor requires the declared precision loss")
        indices = torch.randperm(len(self.features), device="cpu")[:32]
        distribution, _ = model(self.features[indices], encoded=True)
        return action_loss(distribution.mean * 2 - 1, self.targets[indices], WORLD_LOSS_PROFILE, self.row_weights[indices])


def loss_definition(profile):
    if profile not in LOSS_PROFILES:
        raise ValueError("Explicit supported corpus warm loss required")
    definition = {"method": profile, "scales": list((1, 1, 1, 1) if profile == "uniform-action-v1" else POSE_BUDGET_SCALES),
            "reduction": "Mean over all four squared signed Beta-mean action errors after multiplying each axis scale",
            "scope": "Training-only action units relative to unchanged20% pose budgets; no new tolerance, inference projection or private input"}
    if profile == WORLD_LOSS_PROFILE:
        definition["reduction"] = "Mean over uniformly sampled rows of four-axis precision error times its full-corpus world weight"
        definition["worldWeighting"] = {"formula": "N/(W*L): N admitted rows, W admitted worlds, L rows in this world",
            "objective": "Equal average of each admitted world's mean four-axis precision error",
            "minibatches": "Use full-corpus weights without within-minibatch renormalization",
            "sampling": "Every admitted row once per epoch in the unchanged uniform shuffle; no row duplication or selected labels"}
    return definition


def action_loss(predicted, targets, profile, row_weights=None):
    definition = loss_definition(profile)
    if (not isinstance(predicted, torch.Tensor) or not isinstance(targets, torch.Tensor) or
            predicted.ndim != 2 or predicted.shape[1] != 4 or predicted.shape != targets.shape or not len(predicted) or
            not torch.isfinite(predicted).all() or not torch.isfinite(targets).all() or torch.any(targets.abs() > 1)):
        raise ValueError("Finite matching four-axis corpus action tensors required")
    scales = torch.tensor(definition["scales"], dtype=torch.float32, device=predicted.device)
    error = ((predicted - targets.to(predicted.device)) * scales).square()
    if profile == WORLD_LOSS_PROFILE:
        if (not isinstance(row_weights, torch.Tensor) or row_weights.shape != (len(predicted),) or
                not torch.is_floating_point(row_weights) or not torch.isfinite(row_weights).all() or
                torch.any(row_weights <= 0) or torch.any(row_weights > 202)):
            raise ValueError("Finite positive full-corpus world weights required")
        return (error.mean(dim=1) * row_weights.to(predicted.device)).mean()
    if row_weights is not None:
        raise ValueError("World weights require their explicit corpus loss profile")
    return error.mean()


def imitation(model, optimizer, rows, profile="uniform-action-v1", epochs=32, row_weights=None):
    loss_definition(profile)
    if profile != WORLD_LOSS_PROFILE and row_weights is not None:
        raise ValueError("World weights require their explicit corpus loss profile")
    if profile == "uniform-action-v1":
        return imitation_update(model, optimizer, rows, epochs)
    if not rows or isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 64:
        raise ValueError("Bounded nonempty corpus imitation minibatches required")
    features = torch.tensor([row["modelInput"] for row in rows], dtype=torch.float32, device="cpu")
    targets = torch.tensor([row["teacherAction"] for row in rows], dtype=torch.float32, device="cpu")
    world_weights = torch.tensor(row_weights, dtype=torch.float32, device="cpu") if profile == WORLD_LOSS_PROFILE and row_weights is not None else None
    if (features.shape[1:] != (len(model.scales),) or targets.shape != (len(rows), 4) or
            not torch.isfinite(features).all() or not torch.isfinite(targets).all() or torch.any(targets.abs() > 1)):
        raise ValueError("Finite original public inputs and bounded actual corpus actions required")
    if profile == WORLD_LOSS_PROFILE:
        action_loss(targets, targets, profile, world_weights)
    parameters = [*model.body.parameters(), *model.actor.parameters()]
    losses, weights, norms = [], [], []
    for _ in range(epochs):
        for indices in torch.randperm(len(rows), device="cpu").tensor_split(math.ceil(len(rows) / 64)):
            distribution, _ = model(features[indices], encoded=True)
            loss = action_loss(distribution.mean * 2 - 1, targets[indices], profile,
                               world_weights[indices] if world_weights is not None else None)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite precision corpus loss; no checkpoint published")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            weights.append(len(indices))
            norms.append(float(norm))
    return {"phase": "measured-controller-imitation", "meanLoss": sum(loss * weight for loss, weight in zip(losses, weights)) / sum(weights),
            "maximumGradientNorm": max(norms), "imitationRows": len(rows), "imitationEpochs": epochs,
            "imitationLossProfile": profile, "imitationLossScales": loss_definition(profile)["scales"],
            **({"imitationRowWeightMinimum": float(world_weights.min()), "imitationRowWeightMaximum": float(world_weights.max())}
               if world_weights is not None else {})}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_definition(value):
    keys = {"schemaVersion", "method", "manifestSha256", "worldsSha256", "recordsSha256", "producerSourcesSha256",
            "featureTransform", "featureSize", "worldCount", "labelWorldCount", "rowCount", "sourceActualQueries",
            "warmLearning", "anchorSelection", "nativeCurriculumOffset", "accounting", "scope"}
    if not isinstance(value, dict) or set(value) != keys or value["schemaVersion"] != 1 or value["method"] != METHOD_ID:
        raise ValueError("Exact measured expert-corpus definition required")
    for name in ("manifestSha256", "worldsSha256", "recordsSha256", "producerSourcesSha256"):
        text = value[name]
        if not isinstance(text, str) or len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
            raise ValueError("Actual immutable expert-corpus digests required")
    if value["featureTransform"] != FEATURE_TRANSFORM or value["featureSize"] != FEATURE_SIZE:
        raise ValueError("Original public145-input expert encoder required")
    for name, maximum in (("worldCount", MAX_WORLDS), ("labelWorldCount", MAX_WORLDS), ("rowCount", MAX_WORLDS * 202),
                          ("sourceActualQueries", 2000000)):
        count = value[name]
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= maximum:
            raise ValueError("Bounded actual expert-corpus counts required")
    if (value["labelWorldCount"] > value["worldCount"] or value["nativeCurriculumOffset"] != value["worldCount"] or
            value["labelWorldCount"] > value["rowCount"] or value["sourceActualQueries"] < value["rowCount"]):
        raise ValueError("Every source world and native curriculum offset required")
    expected = descriptions()
    if any(value[name] != text for name, text in expected.items()):
        raise ValueError("Exact expert-corpus learning/accounting declaration required")
    return value


def descriptions():
    return {"warmLearning": "Every full admitted trajectory row, uniform original32-epoch imitation at each declared warm update",
            "anchorSelection": "Existing512-row bank; equal chronological quota per admitted world, earliest declared worlds receive remainder, at most16 rows/world",
            "accounting": "Source collection cost retained separately; rereading data adds no physical queries. Native collectionSteps and episodes exclude offline repetitions; accepted learning steps include them.",
            "scope": "Measured training data only; no inference teacher, private actor input, changed physics/gate or learned-improvement claim"}


def chronological(rows, count):
    if count == 1:
        return [rows[0]]
    return [rows[round(index * (len(rows) - 1) / (count - 1))] for index in range(count)]


def anchor_rows(records):
    admitted = [record for record in records if record["rows"]]
    if not admitted:
        raise ValueError("Actually successful expert trajectories required")
    quota, remainder = divmod(ANCHOR_CAPACITY, len(admitted))
    result = []
    for index, record in enumerate(admitted):
        count = min(len(record["rows"]), ANCHOR_ROWS_PER_WORLD, quota + int(index < remainder))
        result.extend(chronological(record["rows"], count))
    return result


def world_row_weights(records):
    lengths = [len(record["rows"]) for record in records if record["rows"]]
    if not lengths or len(lengths) > MAX_WORLDS or any(length > 202 for length in lengths):
        raise ValueError("Bounded nonempty admitted expert trajectories required")
    total = sum(lengths)
    return [total / (len(lengths) * length) for length in lengths for _ in range(length)]


def load_corpus(directory, expected=None):
    directory = Path(directory)
    manifest, manifest_sha = read_json(directory / "dataset.json")
    if manifest.get("definition", {}).get("method") != METHOD:
        raise ValueError("Original measured public-motion corpus required")
    records = list(iter_demonstrations(directory))
    following, following_sha = read_json(directory / "dataset.json")
    if following != manifest or following_sha != manifest_sha:
        raise ValueError("Corpus declaration changed during admission")
    rows = [row for record in records for row in record["rows"]]
    definition = {"schemaVersion": 1, "method": METHOD_ID, "manifestSha256": manifest_sha,
                  "worldsSha256": manifest["worldsSha256"], "recordsSha256": digest_json(manifest["records"]),
                  "producerSourcesSha256": digest_json(manifest["sources"]), "featureTransform": FEATURE_TRANSFORM,
                  "featureSize": FEATURE_SIZE, "worldCount": len(records), "labelWorldCount": sum(bool(record["rows"]) for record in records),
                  "rowCount": len(rows), "sourceActualQueries": manifest["actualQueries"],
                  "nativeCurriculumOffset": len(records), **descriptions()}
    validate_definition(definition)
    if expected is not None and definition != expected:
        raise ValueError("Exact original expert-corpus bytes/definitions required")
    return {"definition": definition, "rows": rows, "anchorRows": anchor_rows(records), "rowWeights": world_row_weights(records),
            "worlds": [record["world"] for record in records]}
