"""Bounded real gradient training; no cloud provisioning or automatic final evaluation."""

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch
from torch import nn

from roboproof.gpu.controller import resolve_device
from .data import CONTRACT, file_hash, load_package, prepare, source_hash, write_json
from .model import FailurePredictor, classification_metrics, loss_for


def code_identity():
    directory = Path(__file__).parent
    return {name: source_hash(directory / name) for name in ["contract.json", "data.py", "model.py", "train.py", "../gpu/controller.py"]}


def tensors(package, split, device):
    rows = [package["rows"][index] for index in package["assignments"][split]]
    preprocessing = package["preprocessing"]
    inputs = torch.tensor([row["features"] for row in rows], dtype=torch.float32, device=device)
    inputs = (inputs - torch.tensor(preprocessing["mean"], device=device)) / torch.tensor(preprocessing["scale"], device=device)
    labels = torch.tensor([row["labels"] for row in rows], dtype=torch.float32, device=device)
    regression = torch.tensor([row["regression"] for row in rows], dtype=torch.float32, device=device)
    mask = torch.tensor([row["regression_mask"] for row in rows], dtype=torch.float32, device=device)
    regression = (regression - torch.tensor(preprocessing["target_mean"], device=device)) / torch.tensor(preprocessing["target_scale"], device=device)
    return inputs, labels, regression, mask


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(path, checkpoint):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def device_metadata(device):
    return dict(device="rocm" if device.type == "cuda" else "cpu", torch=str(torch.__version__),
                hip=torch.version.hip, python=platform.python_version(), platform=platform.platform(),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                precision="float32 neural model; does not change float64 simulation", cpu_threads=torch.get_num_threads())


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_for(package, config, device):
    return FailurePredictor(len(package["feature_names"]), len(CONTRACT["classification_targets"]),
                            len(CONTRACT["regression_targets"]), config["width"], config["depth"]).to(device)


def evaluate_model(model, package, split, device, temperature=1.0, linear=None):
    inputs, labels, regression, mask = tensors(package, split, device)
    model.eval()
    with torch.no_grad():
        logits, predicted = model(inputs)
        result = dict(classification=classification_metrics(torch.sigmoid(logits / temperature), labels, CONTRACT["classification_targets"]))
        metrics = {}
        preprocessing = package["preprocessing"]
        for column, name in enumerate(CONTRACT["regression_targets"]):
            observed = mask[:, column].bool()
            if observed.any():
                scale, mean = preprocessing["target_scale"][column], preprocessing["target_mean"][column]
                predicted_log = predicted[observed, column] * scale + mean
                actual_log = regression[observed, column] * scale + mean
                metrics[name] = dict(count=int(observed.sum()), mae_log1p=float((predicted_log - actual_log).abs().mean()))
            else:
                metrics[name] = dict(count=0, mae_log1p=None)
        result["regression"] = metrics
        training_labels = tensors(package, "train", device)[1]
        frequencies = (training_labels.sum(0) + 1) / (len(training_labels) + 2)
        result["smoothed_frequency_baseline"] = classification_metrics(frequencies.expand_as(labels), labels, CONTRACT["classification_targets"])
        result["majority_baseline"] = classification_metrics((frequencies >= 0.5).float().expand_as(labels), labels, CONTRACT["classification_targets"])
        if linear is not None:
            result["logistic_baseline"] = classification_metrics(torch.sigmoid(linear(inputs)), labels, CONTRACT["classification_targets"])
    return result


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Training wall-time budget reached; checkpoint retained when available")


def train_model(package_path, output, *, allow_legacy=False, device_name="cpu", epochs=50,
                width=128, depth=3, batch_size=64, learning_rate=0.001, patience=8,
                seed=42, max_seconds=300, resume=None):
    for value, lower, upper, name in [(epochs, 1, 500, "epochs"), (batch_size, 1, 4096, "batch_size"),
                                      (patience, 1, 100, "patience"), (seed, 0, 0xffffffff, "seed")]:
        if not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"Invalid {name}")
    if not math.isfinite(max_seconds) or not 1 <= max_seconds <= 3600 or not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.1:
        raise ValueError("Invalid wall-time or learning-rate limit")
    package = load_package(package_path, allow_legacy)
    device = resolve_device(device_name)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()) and resume is None:
        raise ValueError("Output must be empty; resume explicitly or choose a new directory")
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(width=width, depth=depth, batch_size=batch_size, learning_rate=learning_rate, patience=patience, seed=seed)
    model = model_for(package, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed ^ 0x7f4a7c15)
    metadata = device_metadata(device)
    inputs, labels, regression, mask = tensors(package, "train", device)
    validation = tensors(package, "validation", device)
    best_loss, best_epoch, stale, first_epoch, history = float("inf"), 0, 0, 0, []
    best_state = cpu_state(model)
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["package_sha256"] != package["package_sha256"] or saved["config"] != config or saved["code_identity"] != code_identity() or saved["runtime"] != metadata:
            raise ValueError("Resume requires identical dataset, configuration, code and runtime/device")
        if saved.get("complete"):
            raise ValueError("Completed run is frozen; create a new experiment")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["shuffle_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["device_rng"])
        best_state, best_loss, best_epoch = saved["best_model"], saved["best_loss"], saved["best_epoch"]
        stale, first_epoch, history = saved["stale"], saved["epoch"], saved["history"]
    if first_epoch > epochs:
        raise ValueError("Resume epoch limit cannot precede saved epoch")
    initial_state = cpu_state(model)
    synchronize(device)
    started = time.monotonic()
    deadline = started + max_seconds
    checkpoint = saved if resume is not None else None
    for epoch in range(first_epoch, epochs):
        check_time(deadline)
        model.train()
        ordering = torch.randperm(len(inputs), generator=generator).to(device)
        total_loss = 0.0
        for offset in range(0, len(inputs), batch_size):
            check_time(deadline)
            selected = ordering[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits, predictions = model(inputs[selected])
            loss = loss_for(logits, predictions, labels[selected], regression[selected], mask[selected])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(selected)
        model.eval()
        with torch.no_grad():
            validation_logits, validation_predictions = model(validation[0])
            validation_loss = float(loss_for(validation_logits, validation_predictions, *validation[1:]))
        if not math.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, best_state, stale = validation_loss, epoch + 1, cpu_state(model), 0
        else:
            stale += 1
        history.append(dict(epoch=epoch + 1, train_loss=total_loss / len(inputs), validation_loss=validation_loss))
        checkpoint = dict(schema_version=1, package_sha256=package["package_sha256"], config=config,
                          code_identity=code_identity(), runtime=metadata, model=cpu_state(model),
                          optimizer=optimizer.state_dict(), best_model=best_state, best_loss=best_loss,
                          best_epoch=best_epoch, stale=stale, epoch=epoch + 1, history=history,
                          shuffle_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                          device_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [], complete=False)
        save_checkpoint(directory / "checkpoint.pt", checkpoint)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    linear = nn.Linear(len(package["feature_names"]), len(CONTRACT["classification_targets"])).to(device)
    baseline_optimizer = torch.optim.Adam(linear.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        check_time(deadline)
        baseline_optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(linear(inputs), labels)
        loss.backward()
        baseline_optimizer.step()
    calibration_inputs, calibration_labels, _, _ = tensors(package, "calibration", device)
    with torch.no_grad():
        calibration_logits = model(calibration_inputs)[0]
        candidates = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
        temperature = min(candidates, key=lambda value: float(nn.functional.binary_cross_entropy_with_logits(calibration_logits / value, calibration_labels)))
    check_time(deadline)
    validation_report = evaluate_model(model, package, "validation", device, temperature, linear)
    changed = any(not torch.equal(initial_state[name], best_state[name]) for name in initial_state)
    synchronize(device)
    final_model = dict(schema_version=1, model=best_state, linear_baseline=cpu_state(linear),
                       config=config, temperature=temperature, package_sha256=package["package_sha256"],
                       preprocessing=package["preprocessing"], feature_names=package["feature_names"],
                       feature_sha256=package["feature_sha256"], provenance=package["provenance"],
                       contract_sha256=package["contract_sha256"], scenario_spec_sha256=package["scenario_spec_sha256"],
                       code_identity=code_identity(), engine_family=package["engine_family"], canonical_ready=False,
                       training_evidence=dict(actual_gradient_training=True, weights_changed=changed, runtime=metadata,
                                              epochs_completed=checkpoint["epoch"], best_epoch=best_epoch))
    save_checkpoint(directory / "model.pt", final_model)
    checkpoint["complete"] = True
    save_checkpoint(directory / "checkpoint.pt", checkpoint)
    card = dict(schema_version=1, status="EXPERIMENTAL_LEGACY_MODEL", actual_gradient_training=True,
                weights_changed=changed, amd_training=device.type == "cuda", canonical_ready=False,
                runtime=metadata, config=config, epochs_completed=checkpoint["epoch"], best_epoch=best_epoch,
                seconds=time.monotonic() - started, model_sha256=file_hash(directory / "model.pt"),
                dataset_sha256=package["dataset_sha256"], package_sha256=package["package_sha256"],
                split_sha256=package["split_sha256"], feature_sha256=package["feature_sha256"],
                training_source_sha256=code_identity(), engine_provenance=package["provenance"],
                split_counts={name: len(indices) for name, indices in package["assignments"].items()},
                calibration=dict(method="scalar temperature selected on calibration split", temperature=temperature,
                                 count=len(calibration_inputs)), validation=validation_report,
                final_evaluation="NOT RUN; separate one-shot command required", history=history,
                limitations=["Only legacy reduced-plant labels; no original-browser or physical validity claim.",
                             "No proven advantage over search/baselines; small-corpus results are smoke evidence.",
                             "No unseen controller-build generalization; thresholds fixed at probability 0.5.",
                             "Grouped split is not a parameter-region or controller-version holdout.",
                             "Same-runtime determinism requested; cross-device/version reproducibility not guaranteed."])
    write_json(directory / "model-card.json", card)
    return card


def load_model(path, device_name="cpu"):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("code_identity") != code_identity() or artifact.get("canonical_ready") is not False or artifact.get("engine_family") != CONTRACT["engine_family"]:
        raise ValueError("Model code/engine identity mismatch")
    device = resolve_device(device_name)
    model = model_for(artifact, artifact["config"], device)
    model.load_state_dict(artifact["model"])
    model.eval()
    return artifact, model, device


def evaluate_final(package_path, model_path, output, allow_legacy=False, unlock_final=False):
    if not unlock_final:
        raise ValueError("Final evaluation requires explicit --unlock-final; it consumes the holdout")
    package = load_package(package_path, allow_legacy)
    artifact, model, device = load_model(model_path)
    if artifact["package_sha256"] != package["package_sha256"]:
        raise ValueError("Model was not trained on this package/split")
    marker = Path(package_path).with_name(Path(package_path).name + ".final-consumed.json")
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(dict(model_sha256=file_hash(model_path), package_sha256=package["package_sha256"], warning="Final set consumed; do not tune on it"), handle)
    linear = nn.Linear(len(package["feature_names"]), len(CONTRACT["classification_targets"])).to(device)
    linear.load_state_dict(artifact["linear_baseline"])
    report = dict(split="final", canonical_ready=False, model_sha256=file_hash(model_path),
                  package_sha256=package["package_sha256"],
                  metrics=evaluate_model(model, package, "final", device, artifact["temperature"], linear),
                  interpretation="One-shot legacy evaluation, not certification; if used for tuning replace the holdout")
    write_json(output, report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    packaging = commands.add_parser("prepare")
    packaging.add_argument("--input", required=True)
    packaging.add_argument("--out", required=True)
    packaging.add_argument("--seed", type=int, default=42)
    packaging.add_argument("--allow-legacy", action="store_true")
    training = commands.add_parser("train")
    training.add_argument("--package", required=True)
    training.add_argument("--out", required=True)
    training.add_argument("--device", choices=["cpu", "rocm"], default="cpu")
    for name, default in [("epochs", 50), ("width", 128), ("depth", 3), ("batch-size", 64), ("patience", 8), ("seed", 42)]:
        training.add_argument("--" + name, type=int, default=default)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--max-seconds", type=float, default=300)
    training.add_argument("--resume")
    training.add_argument("--allow-legacy", action="store_true")
    evaluation = commands.add_parser("evaluate-final")
    evaluation.add_argument("--package", required=True)
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--out", required=True)
    evaluation.add_argument("--allow-legacy", action="store_true")
    evaluation.add_argument("--unlock-final", action="store_true")
    commands.add_parser("probe")
    return result


def main():
    options = vars(parser().parse_args())
    command = options.pop("command")
    if command == "prepare":
        package = prepare(options.pop("input"), options.pop("out"), **options)
        print(json.dumps(dict(rows=len(package["rows"]), splits={name: len(indices) for name, indices in package["assignments"].items()}, canonical_ready=False)))
    elif command == "train":
        options["device_name"] = options.pop("device")
        card = train_model(options.pop("package"), options.pop("out"), **options)
        print(json.dumps({name: card[name] for name in ["status", "actual_gradient_training", "weights_changed", "amd_training", "epochs_completed", "seconds"]}))
    elif command == "evaluate-final":
        report = evaluate_final(options.pop("package"), options.pop("model"), options.pop("out"), **options)
        print(json.dumps(dict(split=report["split"], canonical_ready=False)))
    else:
        available = bool(torch.version.hip and torch.cuda.is_available())
        print(json.dumps(dict(rocm_ready=available, metadata=device_metadata(torch.device("cuda:0" if available else "cpu")))))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, TimeoutError, OSError, KeyError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
