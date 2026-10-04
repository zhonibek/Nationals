"""Versioned pre-run features, observed labels and grouped dataset packaging."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path(__file__).with_name("contract.json")
SPEC_PATH = ROOT / "roboproof/scenario-spec.json"


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()


CONTRACT = read_json(CONTRACT_PATH)
SPEC = read_json(SPEC_PATH)


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def finite(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}; expected finite [{minimum}, {maximum}]")
    return value


def features(scenario):
    if scenario.get("schema_version") != 1 or not isinstance(scenario.get("scenario_id"), str) or not scenario["scenario_id"]:
        raise ValueError("Unsupported scenario")
    seed = finite(scenario.get("random_seed"), "random_seed", 0, 0xffffffff)
    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    environment, task, controller = scenario["environment"], scenario["task"], scenario["controller"]
    if set(environment) != set(CONTRACT["environment_order"]):
        raise ValueError("Environment does not match feature schema")
    mode = task["localization"]
    modes = ["encoders", "encoders-imu", "ground-truth-baseline"]
    if mode not in modes or controller["adapter"] != "nationals-wasm":
        raise ValueError("Unsupported localization/controller")
    names, values = [], []
    for name in CONTRACT["environment_order"]:
        value = finite(environment[name], name, *SPEC["ranges"][name])
        names.append("environment." + name)
        values.append(0.0 if name.startswith("imu_") and mode != "encoders-imu" else float(value))
    for name in ["start", "goal"]:
        if not isinstance(task[name], list) or len(task[name]) != 3:
            raise ValueError("Pose must have three coordinates")
        for value in task[name]:
            finite(value, name, -100, 100)
    names.extend(["task.delta_x", "task.delta_y", "task.start_heading_sin", "task.start_heading_cos", "task.delta_heading_sin", "task.delta_heading_cos"])
    heading = task["goal"][2] - task["start"][2]
    values.extend([task["goal"][0] - task["start"][0], task["goal"][1] - task["start"][1], math.sin(task["start"][2]), math.cos(task["start"][2]), math.sin(heading), math.cos(heading)])
    names.extend(["task.dt", "task.duration"])
    values.extend([finite(task["dt"], "dt", 0.005, 0.05), finite(task["duration"], "duration", task["dt"], 60)])
    for option in modes:
        names.append("task.localization." + option)
        values.append(float(mode == option))
    for name, bounds in [("radius", (0.1, 0.3)), ("max_wheel_speed", (0.1, 2)), ("reference_time_scale", (1, 4))]:
        names.append("controller." + name)
        values.append(finite(controller[name], name, *bounds))
    for name in CONTRACT["threshold_order"]:
        names.append("task.thresholds." + name)
        values.append(finite(task["thresholds"][name], name, 0.000001, 1000))
    return names, values


def group_keys(scenario, lineage_root):
    environment = {}
    for name, bounds in SPEC["ranges"].items():
        relative = (scenario["environment"][name] - bounds[0]) / (bounds[1] - bounds[0])
        environment[name] = math.floor(relative / CONTRACT["near_duplicate_resolution"] + 0.5)
    keys = [f"seed:{scenario['random_seed']}", "world-bin:" + digest(dict(environment=environment, task=scenario["task"]))]
    if lineage_root is not None:
        if not isinstance(lineage_root, str) or not lineage_root:
            raise ValueError("Invalid lineage root")
        keys.append("lineage:" + lineage_root)
    return keys


def load_export(path, allow_legacy=False):
    data = read_json(path)
    if not allow_legacy or data.get("engine_family") != CONTRACT["engine_family"] or data.get("canonical_ready") is not False:
        raise ValueError("Only explicitly opted-in legacy experiments are implemented; canonical training is blocked")
    if data.get("schema_version") != 1 or data.get("contract_sha256") != source_hash(CONTRACT_PATH) or data.get("scenario_spec_sha256") != source_hash(SPEC_PATH):
        raise ValueError("Dataset contract/spec hash mismatch")
    if not isinstance(data.get("rows"), list) or not 1 <= len(data["rows"]) <= 100000:
        raise ValueError("Invalid dataset size")
    for name in ["simulation_version", "controller_wasm_sha256"]:
        value = data["provenance"][name]
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Missing engine/controller identity")
    rows = []
    seen = set()
    for record in data["rows"]:
        serialized = record["scenario_json"]
        if hashlib.sha256(serialized.encode()).hexdigest() != record["scenario_sha256"]:
            raise ValueError("Scenario hash mismatch")
        scenario = json.loads(serialized, parse_constant=reject_constant)
        names, inputs = features(scenario)
        if scenario["scenario_id"].startswith("evo-") and not record["lineage_root"]:
            raise ValueError("Missing evolutionary lineage")
        categories = record["categories"]
        if not isinstance(categories, list) or not categories or len(set(categories)) != len(categories) or any(name not in ["SUCCESS", *CONTRACT["classification_targets"]] for name in categories) or ("SUCCESS" in categories and len(categories) != 1):
            raise ValueError("Unknown or inconsistent robot labels")
        targets, mask = [], []
        for name in CONTRACT["regression_targets"]:
            value = record["metrics"][name]
            missing = name == "completion_time" or ("NUMERICAL_FAILURE" in categories and name in ["endpoint_error", "final_heading_error"])
            if value is None and not missing:
                raise ValueError("Unexpected missing regression target")
            if value is not None:
                finite(value, name, 0, 1e12)
            targets.append(math.log1p(value) if value is not None else 0.0)
            mask.append(float(value is not None))
        metrics, thresholds = record["metrics"], scenario["task"]["thresholds"]
        observed = dict(PATH_DIVERGENCE=metrics["max_path_deviation"] > thresholds["path"],
                        HEADING_INSTABILITY=metrics["final_heading_error"] is not None and metrics["final_heading_error"] > thresholds["heading"],
                        OSCILLATION=metrics["oscillations"] >= thresholds["oscillations"],
                        TIMEOUT=metrics["completion_time"] is None,
                        ENDPOINT_FAILURE=metrics["endpoint_error"] is not None and metrics["endpoint_error"] > thresholds["endpoint"],
                        LOCALIZATION_DIVERGENCE=metrics["max_localization_error"] > thresholds["localization"],
                        MOTOR_SATURATION=metrics["saturation_fraction"] > thresholds["saturation_fraction"])
        if any((name in categories) != failed for name, failed in observed.items()):
            raise ValueError("Category/metric label audit failed")
        row_id = record["row_id"]
        if row_id in seen:
            raise ValueError("Duplicate dataset row")
        seen.add(row_id)
        rows.append(dict(row_id=row_id, scenario=scenario, features=inputs, lineage_root=record["lineage_root"],
                         group_keys=group_keys(scenario, record["lineage_root"]),
                         labels=[float(name in categories) for name in CONTRACT["classification_targets"]],
                         regression=targets, regression_mask=mask))
    return data, rows, names


def split_groups(rows, seed):
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    key_owner = {}
    for index, row in enumerate(rows):
        for key in row["group_keys"]:
            if key in key_owner:
                parents[root(index)] = root(key_owner[key])
            key_owner[key] = index
    groups = {}
    for index in range(len(rows)):
        groups.setdefault(root(index), []).append(index)
    if len(groups) < 4:
        raise ValueError("At least four independent groups are required")
    ordered = sorted(groups.values(), key=lambda members: digest([seed, sorted(rows[index]["row_id"] for index in members)]))
    counts = [max(1, math.floor(len(ordered) * fraction)) for fraction in CONTRACT["split_fractions"]]
    while sum(counts) > len(ordered):
        largest = max(range(4), key=lambda index: counts[index])
        counts[largest] -= 1
    counts[0] += len(ordered) - sum(counts)
    assignments = {}
    cursor = 0
    for split, count in zip(CONTRACT["split_order"], counts):
        assignments[split] = sorted(index for members in ordered[cursor:cursor + count] for index in members)
        cursor += count
    return assignments


def normalization(rows, indices):
    width = len(rows[0]["features"])
    mean = [sum(rows[index]["features"][column] for index in indices) / len(indices) for column in range(width)]
    deviations = [math.sqrt(sum((rows[index]["features"][column] - mean[column]) ** 2 for index in indices) / len(indices)) for column in range(width)]
    scale = [value if value > 1e-8 else 1.0 for value in deviations]
    target_mean, target_scale = [], []
    for column in range(len(CONTRACT["regression_targets"])):
        observed = [rows[index]["regression"][column] for index in indices if rows[index]["regression_mask"][column]]
        average = sum(observed) / len(observed) if observed else 0.0
        variance = sum((value - average) ** 2 for value in observed) / len(observed) if observed else 0.0
        target_mean.append(average)
        target_scale.append(math.sqrt(variance) if variance > 1e-16 else 1.0)
    return dict(mean=mean, scale=scale, target_mean=target_mean, target_scale=target_scale)


def prepare(source, output, seed=42, allow_legacy=False):
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 0xffffffff:
        raise ValueError("Invalid split seed")
    if Path(output).exists():
        raise ValueError("Package already exists; preserve frozen splits and use a new output")
    data, rows, names = load_export(source, allow_legacy)
    assignments = split_groups(rows, seed)
    package = dict(schema_version=1, engine_family=data["engine_family"], canonical_ready=False,
                   dataset_sha256=file_hash(source), contract_sha256=data["contract_sha256"],
                   scenario_spec_sha256=data["scenario_spec_sha256"], provenance=data["provenance"],
                   sampling=data["sampling"], feature_names=names, split_seed=seed,
                   assignments=assignments, rows=rows, preprocessing=normalization(rows, assignments["train"]))
    package["split_sha256"] = digest({split: [rows[index]["row_id"] for index in indices] for split, indices in assignments.items()})
    package["feature_sha256"] = digest(names)
    package["package_sha256"] = digest(package)
    write_json(output, package)
    return package


def load_package(path, allow_legacy=False):
    package = read_json(path)
    payload = {name: value for name, value in package.items() if name != "package_sha256"}
    if package.get("package_sha256") != digest(payload):
        raise ValueError("Package integrity mismatch")
    if not allow_legacy or package.get("canonical_ready") is not False or package.get("engine_family") != CONTRACT["engine_family"]:
        raise ValueError("Legacy experimental opt-in required; canonical model is not available")
    if package.get("contract_sha256") != source_hash(CONTRACT_PATH) or package.get("scenario_spec_sha256") != source_hash(SPEC_PATH):
        raise ValueError("Package contract/spec mismatch")
    if set(package["assignments"]) != set(CONTRACT["split_order"]):
        raise ValueError("Unexpected split")
    all_indices = [index for split in CONTRACT["split_order"] for index in package["assignments"][split]]
    if any(not isinstance(index, int) or isinstance(index, bool) for index in all_indices):
        raise ValueError("Invalid row index")
    if sorted(all_indices) != list(range(len(package["rows"]))) or any(not package["assignments"][split] for split in CONTRACT["split_order"]):
        raise ValueError("Invalid, overlapping or empty split")
    owners = {}
    for split, indices in package["assignments"].items():
        for index in indices:
            row = package["rows"][index]
            names, expected = features(row["scenario"])
            if names != package["feature_names"] or expected != row["features"]:
                raise ValueError("Feature/schema mismatch")
            if group_keys(row["scenario"], row["lineage_root"]) != row["group_keys"]:
                raise ValueError("Grouping mismatch")
            if len(row["labels"]) != len(CONTRACT["classification_targets"]) or any(value not in (0, 1) for value in row["labels"]):
                raise ValueError("Invalid multi-label target")
            for name in ["regression", "regression_mask"]:
                if len(row[name]) != len(CONTRACT["regression_targets"]):
                    raise ValueError("Regression target width mismatch")
            for value, mask in zip(row["regression"], row["regression_mask"]):
                finite(value, "log1p regression target", 0, 100)
                if mask not in (0, 1):
                    raise ValueError("Invalid regression observation mask")
            for key in row["group_keys"]:
                if key in owners and owners[key] != split:
                    raise ValueError("Group leakage between splits")
                owners[key] = split
    if normalization(package["rows"], package["assignments"]["train"]) != package["preprocessing"]:
        raise ValueError("Preprocessing was not fitted on training rows")
    expected_split = digest({split: [package["rows"][index]["row_id"] for index in indices] for split, indices in package["assignments"].items()})
    if expected_split != package["split_sha256"] or digest(package["feature_names"]) != package["feature_sha256"]:
        raise ValueError("Split/feature identity mismatch")
    return package
