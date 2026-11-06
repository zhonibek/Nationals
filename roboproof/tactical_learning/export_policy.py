"""Export the actor of a trained tactical checkpoint to files the app can run without PyTorch.

The original 2-layer v2 model becomes one JSON file. A v3 model (rules-engine preview features, up to 1024 wide and
10 hidden layers with pre-norm residual blocks) is too large for JSON: it becomes a JSON manifest plus a float32
little-endian weights file next to it, whose SHA-256 and tensor layout the manifest records.

This only reads weights; it never trains. The output belongs in a Git-ignored run folder: model weights are not committed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import torch

from .model import (FEATURE_COUNT, FEATURE_COUNT_V3, FEATURE_COUNT_V4, MAX_HIDDEN_LAYERS, MAX_WIDTH, MODEL_ID,
                    MODEL_ID_V3, MODEL_ID_V4)


POLICY_SCHEMA = "roboproof-tactical-policy-v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runs/tactical/policy.json"
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
POLICY_SCHEMA_BINARY = "roboproof-tactical-policy-v3-binary"
MAX_CHECKPOINT_BYTES_V3 = 512 * 1024 * 1024  # a 10 M-parameter model plus its Adam state
LAYER_NORM_EPSILON = 1e-5  # torch.nn.LayerNorm default, as used by model.ResidualBlock
# Training update recorded in the checkpoint (absent = the original actor-critic step) -> label shown in the app.
ALGORITHMS = {"actor-critic": "actor-critic: policy gradient with a learned critic baseline (not PPO)",
              "ppo-clip-v1": "PPO: clipped surrogate objective with GAE on the candidate-set actor-critic"}


def _finite(tensor: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
        raise RuntimeError(f"Checkpoint tensor {label} is missing or not finite")
    return tensor.detach().to(torch.float32).cpu()


def actor_layers(state: dict) -> tuple[int, dict]:
    """Pick the actor weights out of a TacticalActorCritic state dict and check every shape."""
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint has no model state")
    w1 = _finite(state.get("encoder.0.weight"), "encoder.0.weight")
    width = w1.shape[0] if w1.ndim == 2 else 0
    if not 16 <= width <= 512:
        raise RuntimeError("Unsupported tactical model width")
    expected = {"encoder.0.weight": (width, FEATURE_COUNT), "encoder.0.bias": (width,), "encoder.2.weight": (width, width),
                "encoder.2.bias": (width,), "policy_head.weight": (1, width), "policy_head.bias": (1,)}
    tensors = {}
    for key, shape in expected.items():
        tensors[key] = _finite(state.get(key), key)
        if tuple(tensors[key].shape) != shape:
            raise RuntimeError(f"Checkpoint tensor {key} has shape {tuple(tensors[key].shape)}, expected {shape}")
    layers = {"hidden1": {"weight": tensors["encoder.0.weight"].tolist(), "bias": tensors["encoder.0.bias"].tolist()},
              "hidden2": {"weight": tensors["encoder.2.weight"].tolist(), "bias": tensors["encoder.2.bias"].tolist()},
              "policy": {"weight": tensors["policy_head.weight"][0].tolist(), "bias": float(tensors["policy_head.bias"][0])}}
    return width, layers


def warm_start_summary(warm_start) -> dict | None:
    if not isinstance(warm_start, dict):
        return None
    final = (warm_start.get("pretraining") or {}).get("final") or {}
    return {"policy": warm_start.get("policy"), "demonstrationEpisodes": warm_start.get("demonstrationEpisodesCompleted"),
            "imitationAccuracy": final.get("accuracy")}


def evaluation_summary(evaluation_path: Path | None, run_id) -> dict | None:
    if evaluation_path is None or not evaluation_path.is_file():
        return None
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("runId") != run_id:
        raise RuntimeError("evaluation.json belongs to a different run than the checkpoint")
    return {"heldOutEpisodes": len(evaluation.get("episodes", [])), "meanScoreMargin": evaluation.get("meanScoreMargin"),
            "baselineMeanScoreMargin": evaluation.get("baselineMeanScoreMargin"),
            "beatsLoaderCycleOnHoldout": evaluation.get("beatsLoaderCycleOnHoldout"), "scope": evaluation.get("scope")}


def build_policy(checkpoint_bytes: bytes, evaluation_path: Path | None = None) -> dict:
    if len(checkpoint_bytes) > MAX_CHECKPOINT_BYTES:
        raise RuntimeError("Checkpoint is larger than expected for the tactical model")
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schemaVersion") != 1 or checkpoint.get("model") != MODEL_ID:
        raise RuntimeError(f"Expected a schema-1 {MODEL_ID} checkpoint")
    width, layers = actor_layers(checkpoint.get("modelState"))
    warm_start = warm_start_summary(checkpoint.get("warmStart"))
    algorithm_id = checkpoint.get("algorithm", "actor-critic")
    if algorithm_id not in ALGORITHMS:
        raise RuntimeError(f"Unknown training algorithm in checkpoint: {algorithm_id}")
    algorithm = ALGORITHMS[algorithm_id] + (f"; behavior-cloning warm start from {warm_start['policy']}" if warm_start else "")
    runtime = checkpoint.get("runtime") if isinstance(checkpoint.get("runtime"), dict) else {}
    return {"schema": POLICY_SCHEMA, "model": MODEL_ID, "featureCount": FEATURE_COUNT, "width": width, "activation": "tanh",
            "algorithmId": algorithm_id, "algorithm": algorithm, "layers": layers,
            "provenance": {"checkpointSha256": hashlib.sha256(checkpoint_bytes).hexdigest(), "runId": checkpoint.get("runId"),
                           "trainingSeed": checkpoint.get("trainingSeed"), "episodesCompleted": checkpoint.get("episodesCompleted"),
                           "gpu": runtime.get("gpu"), "sourceSha256": checkpoint.get("sourceSha256"), "warmStart": warm_start},
            "evaluation": evaluation_summary(evaluation_path, checkpoint.get("runId")),
            "scope": "abstract macro-action model scored by the Override rules engine; no robot motion or physics",
            "learnedImprovementVerified": False, "independentQualificationPerformed": False}


def actor_tensors_v3(state: dict) -> tuple[dict, list[tuple[str, torch.Tensor]]]:
    """The actor tensors of a v3 TacticalActorCritic in forward order, with every shape checked."""
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint has no model state")
    first = _finite(state.get("encoder.0.weight"), "encoder.0.weight")
    width, feature_count = tuple(first.shape) if first.ndim == 2 else (0, 0)
    if not 16 <= width <= MAX_WIDTH or feature_count not in (FEATURE_COUNT, FEATURE_COUNT_V3, FEATURE_COUNT_V4):
        raise RuntimeError("Unsupported tactical model dimensions")
    blocks = 0
    while f"encoder.{2 + blocks}.linear.weight" in state:
        blocks += 1
    residual = blocks > 0
    hidden_layers = blocks + 1 if residual else 2
    if not 2 <= hidden_layers <= MAX_HIDDEN_LAYERS:
        raise RuntimeError("Unsupported tactical model depth")
    expected = [("input.weight", "encoder.0.weight", (width, feature_count)), ("input.bias", "encoder.0.bias", (width,))]
    if residual:
        for block in range(blocks):
            key = f"encoder.{2 + block}"
            expected += [(f"blocks.{block}.norm.weight", f"{key}.norm.weight", (width,)),
                         (f"blocks.{block}.norm.bias", f"{key}.norm.bias", (width,)),
                         (f"blocks.{block}.linear.weight", f"{key}.linear.weight", (width, width)),
                         (f"blocks.{block}.linear.bias", f"{key}.linear.bias", (width,))]
    else:
        expected += [("hidden.weight", "encoder.2.weight", (width, width)), ("hidden.bias", "encoder.2.bias", (width,))]
    expected += [("policy.weight", "policy_head.weight", (1, width)), ("policy.bias", "policy_head.bias", (1,))]
    known = {key for _, key, _ in expected}
    unexpected = sorted(key for key in state if key.startswith("encoder.") and key not in known)
    if unexpected:
        raise RuntimeError(f"Checkpoint has encoder tensors this exporter does not know: {unexpected[:3]}")
    tensors = []
    for name, key, shape in expected:
        tensor = _finite(state.get(key), key)
        if tuple(tensor.shape) != shape:
            raise RuntimeError(f"Checkpoint tensor {key} has shape {tuple(tensor.shape)}, expected {shape}")
        tensors.append((name, tensor.reshape(-1) if name == "policy.weight" else tensor))
    return {"featureCount": feature_count, "width": width, "hiddenLayers": hidden_layers, "residual": residual}, tensors


def build_policy_v3(checkpoint_bytes: bytes, evaluation_path: Path | None = None,
                    weights_file: str = "policy.weights.bin") -> tuple[dict, bytes]:
    """A v3 manifest and its float32 little-endian weights; the manifest names the weights file and its SHA-256."""
    if len(checkpoint_bytes) > MAX_CHECKPOINT_BYTES_V3:
        raise RuntimeError("Checkpoint is larger than expected for the tactical model")
    if Path(weights_file).name != weights_file or not weights_file.endswith(".bin"):
        raise RuntimeError("The weights file must be a plain .bin file name next to the manifest")
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get("schemaVersion") != 1
            or checkpoint.get("model") not in (MODEL_ID_V3, MODEL_ID_V4)):
        raise RuntimeError(f"Expected a schema-1 {MODEL_ID_V3} or {MODEL_ID_V4} checkpoint")
    dimensions, tensors = actor_tensors_v3(checkpoint.get("modelState"))
    layout, chunks, offset = [], [], 0
    for name, tensor in tensors:
        values = tensor.contiguous().numpy().astype("<f4", copy=False)
        layout.append({"name": name, "shape": list(tensor.shape), "offset": offset, "length": int(values.size)})
        chunks.append(values.tobytes())
        offset += int(values.size)
    weights = b"".join(chunks)
    warm_start = warm_start_summary(checkpoint.get("warmStart"))
    algorithm_id = checkpoint.get("algorithm", "actor-critic")
    if algorithm_id not in ALGORITHMS:
        raise RuntimeError(f"Unknown training algorithm in checkpoint: {algorithm_id}")
    algorithm = ALGORITHMS[algorithm_id] + (f"; behavior-cloning warm start from {warm_start['policy']}" if warm_start else "")
    runtime = checkpoint.get("runtime") if isinstance(checkpoint.get("runtime"), dict) else {}
    feature_set = {FEATURE_COUNT: "v2", FEATURE_COUNT_V3: "v3", FEATURE_COUNT_V4: "v4"}[dimensions["featureCount"]]
    if (checkpoint["model"] == MODEL_ID_V4) != (feature_set == "v4"):
        raise RuntimeError("Only a v4 (multi-task) checkpoint has the 52 task-conditioned features")
    manifest = {"schema": POLICY_SCHEMA_BINARY, "model": checkpoint["model"], "featureSet": feature_set, **dimensions,
                "activation": "tanh", "layerNormEpsilon": LAYER_NORM_EPSILON, "actorParameters": offset,
                "dtype": "float32-le", "weightsFile": weights_file, "weightsBytes": len(weights),
                "weightsSha256": hashlib.sha256(weights).hexdigest(), "layout": layout,
                "algorithmId": algorithm_id, "algorithm": algorithm,
                "provenance": {"checkpointSha256": hashlib.sha256(checkpoint_bytes).hexdigest(), "runId": checkpoint.get("runId"),
                               "trainingSeed": checkpoint.get("trainingSeed"), "episodesCompleted": checkpoint.get("episodesCompleted"),
                               "gpu": runtime.get("gpu"), "sourceSha256": checkpoint.get("sourceSha256"), "warmStart": warm_start},
                "evaluation": evaluation_summary(evaluation_path, checkpoint.get("runId")),
                "scope": "abstract macro-action model scored by the Override rules engine; no robot motion or physics",
                "learnedImprovementVerified": False, "independentQualificationPerformed": False}
    return manifest, weights


def independent_holdout_summary(path: Path, checkpoint_sha256: str) -> dict:
    """The fresh-seed result from evaluate.py for exactly this checkpoint."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("kind") != "tactical-independent-holdout":
        raise RuntimeError("Not an independent held-out summary")
    row = next((row for row in summary.get("candidates", []) if row.get("checkpointSha256") == checkpoint_sha256), None)
    if row is None:
        raise RuntimeError("The independent held-out summary has no result for this checkpoint")
    if summary.get("metric") == "taskReturn":  # multi-task (v4) check: task return against task-greedy
        return {"episodes": summary.get("episodes"), "metric": "taskReturn",
                "seeds": {key: summary["seeds"].get(key) for key in ("first", "last")},
                "primary": summary["primary"]["index"] == row["index"], "meanTaskReturn": row["independentMeanTaskReturn"],
                "successRate": row["successRate"], "baselineMeanTaskReturn": row["baselineMeanTaskReturn"],
                "baselineSuccessRate": row["baselineSuccessRate"], "versusTaskGreedy": row["versusTaskGreedy"],
                "versusLoaderCycle": row["versusLoaderCycle"], "scope": summary.get("scope"),
                "learnedImprovementVerified": False}
    return {"episodes": summary.get("episodes"), "seeds": {key: summary["seeds"].get(key) for key in ("first", "last")},
            "primary": summary["primary"]["index"] == row["index"], "meanScoreMargin": row["independentMeanMargin"],
            "baselineMeanScoreMargin": row["baselineMeanMargin"], "versusLoaderCycle": row["versusLoaderCycle"],
            "scope": summary.get("scope"), "learnedImprovementVerified": False}


def write_bytes_atomic(path: Path, value: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a trained tactical actor for the RoboProof app (no training)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="latest.pt from a preserved tactical run folder")
    parser.add_argument("--evaluation", type=Path, help="evaluation.json of the same run (default: next to the checkpoint)")
    parser.add_argument("--independent-holdout", type=Path,
                        help="summary.json from evaluate.py; attaches this checkpoint's fresh-seed result (v3 only)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true", help="overwrite an existing exported policy")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.replace:
        parser.error(f"{args.output} exists; pass --replace to overwrite it")
    evaluation = args.evaluation if args.evaluation is not None else args.checkpoint.with_name("evaluation.json")
    raw = args.checkpoint.read_bytes()
    if len(raw) > MAX_CHECKPOINT_BYTES_V3:
        parser.error("checkpoint is larger than any tactical model")
    model_id = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True).get("model")
    if model_id in (MODEL_ID_V3, MODEL_ID_V4):
        weights_file = args.output.with_suffix(".weights.bin").name
        policy, weights = build_policy_v3(raw, evaluation, weights_file)
        if args.independent_holdout is not None:
            policy["independentHoldout"] = independent_holdout_summary(args.independent_holdout,
                                                                       policy["provenance"]["checkpointSha256"])
        write_bytes_atomic(args.output.with_name(weights_file), weights)  # weights first: the manifest names their hash
    else:
        if args.independent_holdout is not None:
            parser.error("--independent-holdout is supported for v3 policies only")
        policy = build_policy(raw, evaluation)
    write_json_atomic(args.output, policy)
    summary = {"output": str(args.output), "runId": policy["provenance"]["runId"], "width": policy["width"],
               "hiddenLayers": policy.get("hiddenLayers", 2), "weightsFile": policy.get("weightsFile"),
               "checkpointSha256": policy["provenance"]["checkpointSha256"], "algorithm": policy["algorithm"],
               "heldOutMeanScoreMargin": (policy["evaluation"] or {}).get("meanScoreMargin")}
    print(json.dumps(summary, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
