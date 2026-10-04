"""Select new experiments, preserving exploration and excluding all existing split groups."""

import argparse
import json
import math
from pathlib import Path
import random
import sys

import torch

from .data import CONTRACT, digest, features, file_hash, group_keys, load_package, read_json, write_json
from .train import load_model


def acquire(model_path, package_path, candidates_path, output, budget=16, seed=2026,
            exploration=0.25, allow_legacy=False):
    if not allow_legacy:
        raise ValueError("Learned search requires experimental --allow-legacy")
    if not isinstance(budget, int) or not 1 <= budget <= 1000 or not isinstance(seed, int) or not 0 <= seed <= 0xffffffff:
        raise ValueError("Invalid acquisition budget/seed")
    if not math.isfinite(exploration) or not 0.2 <= exploration <= 0.3:
        raise ValueError("Exploration must be in [0.2, 0.3]")
    if Path(output).exists():
        raise ValueError("Preserve selection evidence; output already exists")
    package = load_package(package_path, allow_legacy)
    artifact, model, device = load_model(model_path)
    if artifact["package_sha256"] != package["package_sha256"]:
        raise ValueError("Model/package mismatch")
    data = read_json(candidates_path)
    candidates = data if isinstance(data, list) else data.get("scenarios")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 100000:
        raise ValueError("Expected 1-100000 candidate scenarios")
    existing = {key for row in package["rows"] for key in row["group_keys"]}
    eligible, excluded, seen, seen_ids = [], 0, set(), set()
    for scenario in candidates:
        names, inputs = features(scenario)
        if names != artifact["feature_names"]:
            raise ValueError("Candidate feature schema mismatch")
        if scenario["scenario_id"].startswith("evo-"):
            raise ValueError("Mutation candidate pools require a lineage-aware adapter, not an anonymous scenario array")
        identity = digest({name: value for name, value in scenario.items() if name != "scenario_id"})
        if identity in seen or existing.intersection(group_keys(scenario, None)):
            excluded += 1
            continue
        if scenario["scenario_id"] in seen_ids:
            raise ValueError("Distinct candidates have duplicate IDs")
        seen.add(identity)
        seen_ids.add(scenario["scenario_id"])
        eligible.append((scenario, inputs))
    if len(eligible) < budget:
        raise ValueError("Insufficient new candidates after held-out/training group and duplicate exclusions")
    preprocessing = artifact["preprocessing"]
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(eligible), 1024):
            inputs = torch.tensor([entry[1] for entry in eligible[offset:offset + 1024]], dtype=torch.float32, device=device)
            inputs = (inputs - torch.tensor(preprocessing["mean"], device=device)) / torch.tensor(preprocessing["scale"], device=device)
            probabilities = torch.sigmoid(model(inputs)[0] / artifact["temperature"]).cpu()
            if not torch.isfinite(probabilities).all():
                raise ValueError("Nonfinite prediction")
            for row in probabilities.tolist():
                entropy = sum(-(value * math.log(max(value, 1e-12)) + (1 - value) * math.log(max(1 - value, 1e-12))) for value in row) / len(row) / math.log(2)
                predictions.append(dict(category_probabilities=dict(zip(CONTRACT["classification_targets"], row)),
                                        max_category_probability=max(row), predictive_entropy=entropy,
                                        acquisition_score=0.8 * max(row) + 0.2 * entropy))
    generator = random.Random(seed)
    exploration_count = max(1, math.ceil(budget * exploration))
    random_indices = generator.sample(range(len(eligible)), exploration_count)
    random_set = set(random_indices)
    ranked = sorted((index for index in range(len(eligible)) if index not in random_set),
                    key=lambda index: (-predictions[index]["acquisition_score"], digest(eligible[index][0])))
    chosen = random_indices + ranked[:budget - exploration_count]
    selection = dict(schema_version=1, kind="ml-experimental-acquisition", canonical_ready=False,
                     engine_family=CONTRACT["engine_family"], model_sha256=file_hash(model_path),
                     package_sha256=package["package_sha256"], pool_sha256=file_hash(candidates_path),
                     expected_provenance=artifact["provenance"], model_training=artifact["training_evidence"], seed=seed, exploration_fraction_requested=exploration,
                     exploration_count=exploration_count, pool_count=len(candidates), excluded_count=excluded,
                     scenarios=[eligible[index][0] for index in chosen],
                     predictions=[dict(scenario_id=eligible[index][0]["scenario_id"], reason="random-exploration" if index in random_set else "model-ranked", **predictions[index]) for index in chosen],
                     simulation_status="NOT RUN; confirm.js must execute the source-checked controller",
                     limitations=["Legacy model only; probabilities are predictions, not observed outcomes or safety certificates.",
                                  "Entropy is a heuristic, not validated epistemic uncertainty or an ensemble.",
                                  "Existing dataset groups are excluded, including validation/calibration/final.",
                                  "No matched-budget improvement claim until repeated search comparisons."])
    write_json(output, selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "package", "candidates", "out"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--allow-legacy", action="store_true")
    options = vars(parser.parse_args())
    selection = acquire(options.pop("model"), options.pop("package"), options.pop("candidates"), options.pop("out"), **options)
    print(json.dumps(dict(selected=len(selection["scenarios"]), exploration_count=selection["exploration_count"], simulation_status=selection["simulation_status"])))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
