"""Export the actor of a trained tactical checkpoint to a JSON file the app can run without PyTorch.

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

from .model import FEATURE_COUNT, MODEL_ID


POLICY_SCHEMA = "roboproof-tactical-policy-v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runs/tactical/policy.json"
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
ALGORITHM = "actor-critic: policy gradient with a learned critic baseline (not PPO)"


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
    algorithm = ALGORITHM + (f"; behavior-cloning warm start from {warm_start['policy']}" if warm_start else "")
    runtime = checkpoint.get("runtime") if isinstance(checkpoint.get("runtime"), dict) else {}
    return {"schema": POLICY_SCHEMA, "model": MODEL_ID, "featureCount": FEATURE_COUNT, "width": width, "activation": "tanh",
            "algorithm": algorithm, "layers": layers,
            "provenance": {"checkpointSha256": hashlib.sha256(checkpoint_bytes).hexdigest(), "runId": checkpoint.get("runId"),
                           "trainingSeed": checkpoint.get("trainingSeed"), "episodesCompleted": checkpoint.get("episodesCompleted"),
                           "gpu": runtime.get("gpu"), "sourceSha256": checkpoint.get("sourceSha256"), "warmStart": warm_start},
            "evaluation": evaluation_summary(evaluation_path, checkpoint.get("runId")),
            "scope": "abstract macro-action model scored by the Override rules engine; no robot motion or physics",
            "learnedImprovementVerified": False, "independentQualificationPerformed": False}


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a trained tactical actor to JSON for the RoboProof app (no training)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="latest.pt from a preserved tactical run folder")
    parser.add_argument("--evaluation", type=Path, help="evaluation.json of the same run (default: next to the checkpoint)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true", help="overwrite an existing exported policy")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.replace:
        parser.error(f"{args.output} exists; pass --replace to overwrite it")
    evaluation = args.evaluation if args.evaluation is not None else args.checkpoint.with_name("evaluation.json")
    policy = build_policy(args.checkpoint.read_bytes(), evaluation)
    write_json_atomic(args.output, policy)
    summary = {"output": str(args.output), "runId": policy["provenance"]["runId"], "width": policy["width"],
               "checkpointSha256": policy["provenance"]["checkpointSha256"], "algorithm": policy["algorithm"],
               "heldOutMeanScoreMargin": (policy["evaluation"] or {}).get("meanScoreMargin")}
    print(json.dumps(summary, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
