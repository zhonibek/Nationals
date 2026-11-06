"""Plan and summarize a parallel tactical-training campaign on one AMD Instinct MI300X (AMD Developer Cloud).

    python -m roboproof.tactical_learning.campaign plan --output-root <dir> [options]   # predeclare, check, print
    python -m roboproof.tactical_learning.campaign plan --output-root <dir> --write ... # also write campaign.json
    python -m roboproof.tactical_learning.campaign summarize <dir>                      # after the runs finish
    python -m roboproof.tactical_learning.campaign sample <file> [--interval 15]        # GPU utilization log

`plan` fixes every run's seeds, output folder and command before anything starts, and refuses seed overlaps
(between runs and with consumed_seeds.json) and a cost estimate above the approved ceiling. `summarize` reads
each run's record and the GPU utilization samples. Nothing here trains or touches a GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CONSUMED_SEEDS = Path(__file__).with_name("consumed_seeds.json")
HARD_COST_CAP_USD = 50.0  # the free AMD Developer Cloud credit; no campaign may plan beyond it
EVALUATION_RESERVE_SECONDS = 3600  # held-out evaluation and baselines run after --max-seconds of training
MAX_RUNS = 64


def consumed_ranges(path: Path = CONSUMED_SEEDS):
    record = json.loads(path.read_text(encoding="utf-8"))
    return [(row["first"], row["last"], row["note"]) for row in record["ranges"]]


def plan(*, output_root: Path, runs: int, base_seed: int, seed_stride: int, episodes: int, warm_start: int,
         max_seconds: int, evaluation_episodes: int, algorithm: str, features: str, width: int, hidden_layers: int,
         hourly_rate_usd: float, approved_max_cost_usd: float, python: str = "python3", consumed=None,
         parallel_envs: int = 1) -> dict:
    if not 1 <= runs <= MAX_RUNS:
        raise ValueError(f"runs must be 1..{MAX_RUNS}")
    if not 1 <= episodes <= 10000 or not 0 <= warm_start <= 200 or not 1 <= evaluation_episodes <= 100:
        raise ValueError("episodes, warm-start demonstrations or evaluation episodes are out of range")
    if not 1 <= max_seconds <= 6 * 3600:
        raise ValueError("max-seconds must be 1..21600")
    if not 1 <= parallel_envs <= 8:
        raise ValueError("parallel-envs must be 1..8")
    if seed_stride < episodes + warm_start:
        raise ValueError("seed-stride must be at least episodes + warm-start demonstrations, so runs never share seeds")
    if not 0 < hourly_rate_usd <= 20 or not 0 < approved_max_cost_usd <= HARD_COST_CAP_USD:
        raise ValueError(f"hourly rate must be in (0, 20] and the approved ceiling in (0, {HARD_COST_CAP_USD:g}] USD")
    wall_seconds = max_seconds + EVALUATION_RESERVE_SECONDS
    estimated_cost = wall_seconds / 3600 * hourly_rate_usd
    if estimated_cost > approved_max_cost_usd:
        raise ValueError(f"estimated worst case {estimated_cost:.2f} USD exceeds the approved ceiling {approved_max_cost_usd:.2f} USD")
    consumed = consumed_ranges() if consumed is None else consumed
    planned = []
    for index in range(runs):
        seed = base_seed + index * seed_stride
        first, last = seed, seed + episodes + warm_start - 1
        if last > 0xFFFFFFFF:
            raise ValueError("seed range exceeds uint32")
        for used_first, used_last, note in consumed:
            if first <= used_last and used_first <= last:
                raise ValueError(f"run {index:02d} seeds {first}-{last} overlap consumed seeds {used_first}-{used_last} ({note})")
        output = output_root / f"run-{index:02d}"
        command = [python, "-m", "roboproof.tactical_learning.train", "--approved-amd-mi300x", "--seed", str(seed),
                   "--episodes", str(episodes), "--max-seconds", str(max_seconds),
                   "--evaluation-episodes", str(evaluation_episodes), "--algorithm", algorithm,
                   "--features", features, "--width", str(width), "--hidden-layers", str(hidden_layers)]
        if warm_start:
            command += ["--warm-start-demonstration-episodes", str(warm_start)]
        if parallel_envs > 1:
            command += ["--parallel-envs", str(parallel_envs)]
        command += ["--output", str(output)]
        planned.append({"index": index, "seed": seed, "trainingSeeds": [first, first + episodes - 1],
                        "demonstrationSeeds": [first + episodes, last] if warm_start else None,
                        "output": str(output), "command": command})
    return {"schemaVersion": 1, "kind": "tactical-mi300x-campaign", "runs": planned,
            "configuration": {"runs": runs, "baseSeed": base_seed, "seedStride": seed_stride, "episodes": episodes,
                              "warmStartDemonstrations": warm_start, "maxSeconds": max_seconds,
                              "evaluationEpisodes": evaluation_episodes, "algorithm": algorithm, "features": features,
                              "width": width, "hiddenLayers": hidden_layers, "parallelEnvironments": parallel_envs},
            "wallClockLimitSeconds": wall_seconds, "hourlyRateUsd": hourly_rate_usd,
            "estimatedMaxCostUsd": round(estimated_cost, 2), "approvedMaxCostUsd": approved_max_cost_usd,
            "costNote": "Worst case at the hourly rate the user read in the AMD Developer Cloud console; the droplet must be destroyed afterwards to stop billing.",
            "learnedImprovementVerified": False, "independentQualificationPerformed": False}


USE_KEY = re.compile(r"use \(%\)|gpu_use|gfx_activity|utilization", re.I)


def utilization_values(sample) -> list[float]:
    """GPU-busy percentages from one rocm-smi / amd-smi JSON sample (formats differ between ROCm versions)."""
    values = []

    def walk(node, key=""):
        if isinstance(node, dict):
            for child_key, child in node.items():
                # amd-smi nests numbers as {"gfx_activity": {"value": 37, "unit": "%"}}: keep the parent's name.
                walk(child, key if child_key in ("value", "unit") else str(child_key))
        elif isinstance(node, list):
            for child in node:
                walk(child, key)
        elif USE_KEY.search(key) and "mem" not in key.lower():
            match = re.search(r"-?\d+(?:\.\d+)?", str(node))
            if match:
                values.append(float(match.group()))
    walk(sample)
    return [value for value in values if 0 <= value <= 100]


def summarize(root: Path) -> dict:
    campaign = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
    rows = []
    for run in campaign["runs"]:
        folder = Path(run["output"])
        folder = folder if folder.is_absolute() else root / folder.name
        record = json.loads((folder / "run.json").read_text(encoding="utf-8")) if (folder / "run.json").exists() else {}
        evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8")) if (folder / "evaluation.json").exists() else {}
        exit_file = root / f"run-{run['index']:02d}.exit-code"
        row = {"index": run["index"], "seed": run["seed"], "status": record.get("status", "missing"),
               "exitCode": int(exit_file.read_text().strip()) if exit_file.exists() else None,
               "episodesCompleted": record.get("episodesCompleted"),
               "learnedMargin": evaluation.get("meanScoreMargin"),
               "loaderCycleMargin": (evaluation.get("baselineMeanScoreMargin") or {}).get("loader-cycle"),
               "beatsLoaderCycleOnHoldout": evaluation.get("beatsLoaderCycleOnHoldout")}
        if evaluation.get("metric") == "taskReturn":  # multi-task (v4): task return against the per-task greedy tactic
            row.update(metric="taskReturn", learnedTaskReturn=evaluation.get("meanTaskReturn"),
                       taskGreedyTaskReturn=(evaluation.get("baselineMeanTaskReturn") or {}).get("task-greedy"),
                       successRate=evaluation.get("successRate"),
                       taskGreedySuccessRate=(evaluation.get("baselineSuccessRate") or {}).get("task-greedy"),
                       beatsTaskGreedyOnHoldout=evaluation.get("beatsTaskGreedyOnHoldout"))
        row["learned"] = row.get("learnedTaskReturn", row["learnedMargin"])
        rows.append(row)
    margins = [row["learned"] for row in rows if isinstance(row["learned"], (int, float))]
    busy = []
    samples_file = root / "gpu-utilization.jsonl"
    if samples_file.exists():
        for line in samples_file.read_text(encoding="utf-8").splitlines():
            try:
                busy.extend(utilization_values(json.loads(line).get("sample")))
            except (ValueError, AttributeError):
                continue
    best = max((row for row in rows if isinstance(row["learned"], (int, float))),
               key=lambda row: row["learned"], default=None)
    return {"schemaVersion": 1, "kind": "tactical-mi300x-campaign-summary", "runs": rows,
            "completedRuns": sum(row["status"] == "completed" for row in rows),
            "runsBeatingLoaderCycle": sum(row["beatsLoaderCycleOnHoldout"] is True for row in rows),
            "runsBeatingTaskGreedy": sum(row.get("beatsTaskGreedyOnHoldout") is True for row in rows),
            "metric": "taskReturn" if any(row.get("metric") == "taskReturn" for row in rows) else "scoreMargin",
            "learnedMargin": {"mean": statistics.fmean(margins), "median": statistics.median(margins),
                              "min": min(margins), "max": max(margins)} if margins else None,
            "bestRun": best,
            "gpuBusyPercent": {"samples": len(busy), "mean": round(statistics.fmean(busy), 1), "max": max(busy)} if busy else None,
            "learnedImprovementVerified": False, "independentQualificationPerformed": False,
            "note": "Per-run evaluations are development comparisons on held-out start poses; a run beating loader-cycle is not an independent qualification."}


def sample(path: Path, interval: float = 15.0, iterations: int | None = None):
    """Append one rocm-smi (or amd-smi) JSON sample per interval until stopped; unreadable samples are kept as null."""
    commands = ([["rocm-smi", "--showuse", "--showmemuse", "--json"]] if shutil.which("rocm-smi") else []) + \
        ([["amd-smi", "metric", "--usage", "--json"]] if shutil.which("amd-smi") else [])
    count = 0
    while iterations is None or count < iterations:
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sample": None, "tool": None}
        for command in commands:
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
                record = {**record, "sample": json.loads(completed.stdout), "tool": command[0]}
                break
            except (OSError, subprocess.TimeoutExpired, ValueError):
                continue
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
        count += 1
        if iterations is None or count < iterations:
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("--output-root", type=Path, required=True)
    planner.add_argument("--runs", type=int, default=34)  # 34 x ~5 GB peak fits one MI300X with a 20 GB margin
    planner.add_argument("--base-seed", type=int, default=2600001)
    planner.add_argument("--seed-stride", type=int, default=5000)
    planner.add_argument("--episodes", type=int, default=3000)
    planner.add_argument("--warm-start", type=int, default=32)
    planner.add_argument("--max-seconds", type=int, default=7200)
    planner.add_argument("--evaluation-episodes", type=int, default=8)
    planner.add_argument("--algorithm", choices=("actor-critic", "ppo"), default="ppo")
    planner.add_argument("--features", choices=("v2", "v3", "v4"), default="v3")
    planner.add_argument("--parallel-envs", type=int, default=1)
    planner.add_argument("--width", type=int, default=1024)
    planner.add_argument("--hidden-layers", type=int, default=10)
    planner.add_argument("--hourly-rate-usd", type=float, required=True)
    planner.add_argument("--approved-max-cost-usd", type=float, required=True)
    planner.add_argument("--python", default="python3")
    planner.add_argument("--write", action="store_true", help="write campaign.json (the output root must not exist yet)")
    summary = commands.add_parser("summarize")
    summary.add_argument("root", type=Path)
    sampler = commands.add_parser("sample")
    sampler.add_argument("file", type=Path)
    sampler.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.command == "sample":
        sample(args.file, args.interval)
        return 0
    if args.command == "summarize":
        result = summarize(args.root)
        (args.root / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("completedRuns", "runsBeatingLoaderCycle", "learnedMargin",
                                                      "bestRun", "gpuBusyPercent")}, indent=2))
        return 0
    try:
        result = plan(output_root=args.output_root, runs=args.runs, base_seed=args.base_seed,
                      seed_stride=args.seed_stride, episodes=args.episodes, warm_start=args.warm_start,
                      max_seconds=args.max_seconds, evaluation_episodes=args.evaluation_episodes,
                      algorithm=args.algorithm, features=args.features, width=args.width,
                      hidden_layers=args.hidden_layers, hourly_rate_usd=args.hourly_rate_usd,
                      approved_max_cost_usd=args.approved_max_cost_usd, python=args.python,
                      parallel_envs=args.parallel_envs)
    except ValueError as error:
        print(f"Campaign refused: {error}", file=sys.stderr)
        return 2
    if args.write:
        args.output_root.mkdir(parents=True, exist_ok=False)
        (args.output_root / "campaign.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    first, last = result["runs"][0], result["runs"][-1]
    print(f"{len(result['runs'])} runs, seeds {first['trainingSeeds'][0]}..{last['demonstrationSeeds'][1] if last['demonstrationSeeds'] else last['trainingSeeds'][1]}, "
          f"wall-clock limit {result['wallClockLimitSeconds']} s, worst case {result['estimatedMaxCostUsd']} USD "
          f"(approved {result['approvedMaxCostUsd']} USD)")
    print("first command: " + " ".join(first["command"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
