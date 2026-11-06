"""Build the Tactical AI training page's data from preserved campaign result bundles (no training, no GPU).

    py -3 tools/build_training_dashboard.py            # defaults: the 2026-11-06 MI300X bundles
    py -3 tools/build_training_dashboard.py --output roboproof/dashboard/training-data.json

Reads each results bundle (tar.gz of run folders, checkpoints excluded) and its independent-check archive in place,
and writes one compact JSON: every run's learning curve (moving average, at most 120 points), the best run's PPO
statistics, its warm start, and the verified results. A missing bundle is skipped and listed, never invented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "roboproof/runs/tactical/mi300x-2026-11-06"
DEFAULTS = {
    "single": RUNS / "mi300x-campaign-34.results.tar.gz",
    "single_check": RUNS / "mi300x-campaign-34-independent.tar.gz",
    "multi": RUNS / "mi300x-v4-campaign-16.results.tar.gz",
    "multi_check": RUNS / "mi300x-v4-campaign-16-independent.tar.gz",
}
MAX_CURVE_POINTS = 120
MAX_PPO_POINTS = 200


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bundle(path: Path) -> dict:
    """{relative name: bytes} for the JSON/JSONL/text members of a results bundle (read in memory, nothing extracted)."""
    files = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith((".json", ".jsonl", ".txt", ".exit-code")):
                name = member.name.split("/", 1)[1] if "/" in member.name else member.name
                files[name] = archive.extractfile(member).read()
    return files


def jsonl(data: bytes) -> list:
    rows = []
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def moving_average_curve(values, window, max_points=MAX_CURVE_POINTS):
    """[[episode, mean of the last `window` values], ...] sampled evenly, starting once the window is full (a mean of
    the first few games swings too much to show); episode numbers start at 1."""
    if not values:
        return []
    if len(values) < window:
        return [[len(values), round(sum(values) / len(values), 3)]]
    first = window - 1
    points, total, step = [], 0.0, max(1, -(-(len(values) - first) // (max_points - 1)))
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        if index >= first and ((index - first) % step == 0 or index == len(values) - 1):
            points.append([index + 1, round(total / window, 3)])
    return points


def downsample(rows, limit=MAX_PPO_POINTS):
    if len(rows) <= limit:
        return rows
    step = len(rows) / limit
    return [rows[int(index * step)] for index in range(limit - 1)] + [rows[-1]]


def rounded(value, digits=4):
    return round(value, digits) if isinstance(value, float) else value


def campaign_section(bundle: Path, check: Path | None, multi_task: bool) -> dict:
    files = read_bundle(bundle)
    campaign = json.loads(files["campaign.json"])
    summary = json.loads(files["summary.json"])
    host = files.get("host.txt", b"").decode("utf-8", "replace")
    metric = "taskReturn" if multi_task else "scoreMargin"
    window = 256 if multi_task else 64
    runs = []
    for run in campaign["runs"]:
        name = f"run-{run['index']:02d}"
        episodes = jsonl(files.get(f"{name}/episodes.jsonl", b""))
        record = json.loads(files[f"{name}/run.json"]) if f"{name}/run.json" in files else {}
        evaluation = json.loads(files[f"{name}/evaluation.json"]) if f"{name}/evaluation.json" in files else {}
        values = [row["episodeReturn"] if multi_task else row["score"]["margin"] for row in episodes]
        entry = {"index": run["index"], "seed": run["seed"], "status": record.get("status"),
                 "episodes": len(episodes), "elapsedSeconds": record.get("elapsedSeconds"),
                 "curve": moving_average_curve(values, window)}
        if multi_task:
            entry["successCurve"] = moving_average_curve(
                [1.0 if (row.get("task") or {}).get("achieved") else 0.0 for row in episodes], window)
            entry["heldOut"] = evaluation.get("meanTaskReturn")
            entry["heldOutTaskGreedy"] = (evaluation.get("baselineMeanTaskReturn") or {}).get("task-greedy")
            entry["heldOutSuccess"] = evaluation.get("successRate")
        else:
            entry["heldOut"] = evaluation.get("meanScoreMargin")
            entry["heldOutLoaderCycle"] = (evaluation.get("baselineMeanScoreMargin") or {}).get("loader-cycle")
        runs.append(entry)
    best = max((run for run in runs if isinstance(run["heldOut"], (int, float))), key=lambda run: run["heldOut"])
    name = f"run-{best['index']:02d}"
    training = jsonl(files.get(f"{name}/training.jsonl", b""))
    ppo = downsample([{key: rounded(row.get(key)) for key in ("episodes", "actorLoss", "criticLoss", "entropy",
                                                               "approximateKl", "clipFraction", "scoreMargin", "taskReturn")
                       if row.get(key) is not None} for row in training if row.get("event") == "checkpoint"])
    warm = jsonl(files.get(f"{name}/warm-start.jsonl", b""))
    record = json.loads(files[f"{name}/run.json"])
    warm_start = record.get("warmStart") or {}
    pretraining = warm_start.get("pretraining") or {}
    best_section = {**best, "ppo": ppo, "ppoUpdates": len([row for row in training if row.get("event") == "checkpoint"]),
                    "architecture": record.get("architecture"), "runtime": {key: (record.get("runtime") or {}).get(key)
                                                                            for key in ("gpu", "torch", "hip", "node")},
                    "warmStart": {"policy": warm_start.get("policy"),
                                  "demonstrations": warm_start.get("demonstrationEpisodesCompleted"),
                                  "epochs": pretraining.get("epochsCompleted"),
                                  "accuracyBefore": rounded((pretraining.get("initial") or {}).get("accuracy")),
                                  "accuracyAfter": rounded((pretraining.get("final") or {}).get("accuracy")),
                                  "lossCurve": [[row["epoch"], rounded(row["loss"])] for row in
                                                downsample([row for row in warm if row.get("event") == "pretrain-epoch"], 60)]}}
    section = {"campaign": bundle.name.replace(".results.tar.gz", ""), "bundleSha256": sha256(bundle), "metric": metric,
               "movingAverageEpisodes": window, "configuration": campaign["configuration"], "runs": runs, "best": best_section,
               "summary": {key: summary.get(key) for key in ("completedRuns", "runsBeatingLoaderCycle", "runsBeatingTaskGreedy",
                                                             "gpuBusyPercent")},
               "host": [line for line in host.splitlines() if line.startswith(("commit", "started", "finished", "cpus"))],
               "independentCheck": None}
    if check is not None and check.exists():
        checked = read_bundle(check)
        result = json.loads(checked["summary.json"])
        section["independentCheck"] = {"archiveSha256": sha256(check), "episodes": result["episodes"],
                                       "seeds": {key: result["seeds"].get(key) for key in ("first", "last")},
                                       "primary": result["primary"], "candidates": [
                                           {key: candidate.get(key) for key in (
                                               "index", "seed", "developmentMargin", "independentMeanMargin",
                                               "independentMeanTaskReturn", "successRate", "baselineMeanMargin",
                                               "baselineMeanTaskReturn", "baselineSuccessRate", "versusLoaderCycle",
                                               "versusTaskGreedy") if key in candidate}
                                           for candidate in result["candidates"]],
                                       "scope": result.get("scope")}
    return section


def verified(path: Path) -> bool:
    """True only for a complete bundle: its `.sha256` sidecar exists and matches (a partial download does not)."""
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    return expected == sha256(path)


def build(paths: dict) -> dict:
    data = {"schemaVersion": 1, "kind": "tactical-training-dashboard", "missing": [],
            "scope": "Abstract Override macro-action model scored by the rules engine; not physical motion or a field result.",
            "learnedImprovementVerified": False}
    for key, bundle, check, multi in (("singleGoal", paths["single"], paths["single_check"], False),
                                      ("multiTask", paths["multi"], paths["multi_check"], True)):
        if verified(bundle):
            data[key] = campaign_section(bundle, check if verified(check) else None, multi)
        else:
            data["missing"].append(bundle.name)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=Path, default=value)
    parser.add_argument("--output", type=Path, default=ROOT / "roboproof/dashboard/training-data.json")
    args = parser.parse_args(argv)
    data = build({key: getattr(args, key) for key in DEFAULTS})
    args.output.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size, "missing": data["missing"],
                      "sections": [key for key in ("singleGoal", "multiTask") if key in data]}))


if __name__ == "__main__":
    main()
