"""Independent held-out check of a finished tactical campaign on fresh start poses (approved MI300X droplet only).

    python -m roboproof.tactical_learning.evaluate --approved-amd-mi300x --campaign <campaign dir> \
        --top 5 --base-seed 2100001 --episodes 200 --output <campaign dir>-independent

Each run's own evaluation used 8 start poses, and picking the best of many runs on those poses favours lucky runs.
This check fixes the candidates first (the --top runs by development margin in <campaign>/summary.json; the primary
candidate is the best of them) and writes that plan before any episode is played. Every candidate and every
non-learning baseline then plays the same fresh seeds, alternating red and blue. The seeds are refused if they overlap
consumed_seeds.json or any campaign run's training, demonstration or evaluation seeds. Nothing is trained here, and
the result never sets learnedImprovementVerified: that still needs the user's approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

EVALUATION_SEED_MASK = 0x5BD1E995  # train.py: evaluation seed = training seed ^ mask ^ offset
BASELINE_NAMES = ("loader-cycle", "field-greedy", "greedy-all", "random")
REFERENCE = "loader-cycle"
# A multi-task (v4) campaign is judged on task return, first against the per-task greedy tactic (the stronger bar).
BASELINE_NAMES_V4 = BASELINE_NAMES + ("task-greedy",)
REFERENCE_V4 = "task-greedy"


def baseline_names(features):
    return BASELINE_NAMES_V4 if features == "v4" else BASELINE_NAMES
BOOTSTRAP_RESAMPLES = 10000
USABLE_STATUSES = ("completed", "time-budget-exhausted")


def seed_list(base_seed: int, episodes: int) -> list[int]:
    return [base_seed + index for index in range(episodes)]


def alliance_for(index: int) -> str:
    return "red" if index % 2 == 0 else "blue"


def check_seeds(seeds, campaign: dict, consumed) -> None:
    """Refuse seeds any campaign run trained, imitated or evaluated on, or that the registry marks as used."""
    low, high = min(seeds), max(seeds)
    for first, last, note in consumed:
        if low <= last and first <= high:
            raise ValueError(f"independent seeds {low}-{high} overlap consumed seeds {first}-{last} ({note})")
    evaluation_episodes = campaign["configuration"]["evaluationEpisodes"]
    for run in campaign["runs"]:
        for key in ("trainingSeeds", "demonstrationSeeds"):
            if run.get(key) and low <= run[key][1] and run[key][0] <= high:
                raise ValueError(f"independent seeds overlap run {run['index']:02d} {key}")
        seen = {(run["seed"] ^ EVALUATION_SEED_MASK ^ offset) & 0xFFFFFFFF for offset in range(evaluation_episodes)}
        if seen & set(seeds):
            raise ValueError(f"independent seeds overlap run {run['index']:02d} evaluation seeds")


def choose_candidates(root: Path, campaign: dict, summary: dict, top: int) -> list[dict]:
    """The --top usable runs by development margin (ties: lower index); fixed before any independent episode."""
    by_index = {run["index"]: run for run in campaign["runs"]}
    usable = []
    for row in summary["runs"]:
        folder = root / Path(by_index[row["index"]]["output"]).name
        development = row.get("learned", row.get("learnedMargin"))  # task return for v4, score margin before
        if (row.get("status") in USABLE_STATUSES and isinstance(development, (int, float))
                and (folder / "latest.pt").exists()):
            usable.append({"index": row["index"], "seed": row["seed"], "developmentMargin": development,
                           "developmentMetric": row.get("metric", "scoreMargin"),
                           "developmentLoaderCycleMargin": row.get("loaderCycleMargin"),
                           "checkpoint": str(folder / "latest.pt"),
                           "checkpointSha256": hashlib.sha256((folder / "latest.pt").read_bytes()).hexdigest()})
    usable.sort(key=lambda row: (-row["developmentMargin"], row["index"]))
    if not usable:
        raise ValueError("no completed run with a checkpoint and a development margin")
    return usable[:top]


def paired_statistics(learned, reference, resamples=BOOTSTRAP_RESAMPLES, seed=0) -> dict:
    """Mean paired margin difference with a seeded percentile-bootstrap 95% interval and win/tie/loss counts."""
    if len(learned) != len(reference) or len(learned) < 2:
        raise ValueError("paired statistics need two equal-length margin lists of at least 2 episodes")
    differences = [a - b for a, b in zip(learned, reference)]
    generator = random.Random(seed)
    count = len(differences)
    means = sorted(statistics.fmean(generator.choices(differences, k=count)) for _ in range(resamples))
    low, high = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    return {"episodes": count, "meanDifference": statistics.fmean(differences),
            "standardDeviation": statistics.stdev(differences),
            "bootstrap95": [low, high], "bootstrapResamples": resamples, "bootstrapSeed": seed,
            "wins": sum(value > 0 for value in differences), "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
            "passes": low > 0}


def worker(candidate: dict, features: str, seeds, output: Path) -> None:
    """Play one checkpoint deterministically on every seed and write its per-episode margins."""
    import torch
    from .model import TacticalActorCritic
    from .train import NodeTacticalEnvironment, run_episode

    device = torch.device("cuda:0")
    checkpoint = torch.load(candidate["checkpoint"], map_location=device, weights_only=True)
    architecture = checkpoint["architecture"]
    model = TacticalActorCritic(feature_count=architecture["featureCount"], width=architecture["width"],
                                hidden_layers=architecture["hiddenLayers"]).to(device)
    model.load_state_dict(checkpoint["modelState"])
    model.eval()
    environment = NodeTacticalEnvironment(features)
    episodes = []
    try:
        with (output / f"exposure-run-{candidate['index']:02d}.jsonl").open("x", encoding="utf-8") as exposure:
            for index, seed in enumerate(seeds):
                _, result = run_episode(environment, model, seed, alliance_for(index), device, deterministic=True)
                episodes.append(result)
                exposure.write(json.dumps({"seed": seed, "alliance": result["alliance"], "decisions": result["decisions"]}) + "\n")
                exposure.flush()
    finally:
        environment.close()
    record = {"candidate": candidate, "episodes": episodes, "margins": [row["score"]["margin"] for row in episodes]}
    if features == "v4":
        record.update(taskReturns=[row["episodeReturn"] for row in episodes],
                      achieved=[bool(row["task"]["achieved"]) for row in episodes],
                      tasks=[row["task"]["summary"] for row in episodes])
    (output / f"candidate-run-{candidate['index']:02d}.json").write_text(json.dumps(record, indent=2) + "\n",
                                                                         encoding="utf-8")


def baseline_worker(features: str, seeds, output: Path) -> None:
    from .train import NodeTacticalEnvironment

    environment = NodeTacticalEnvironment(features)
    names = baseline_names(features)
    fields = ("margins", "taskReturns", "achieved", "tasks") if features == "v4" else ("margins",)
    try:
        values = {field: {name: [None] * len(seeds) for name in names} for field in fields}
        for alliance in ("red", "blue"):
            positions = [index for index in range(len(seeds)) if alliance_for(index) == alliance]
            results = environment.baselines([seeds[index] for index in positions], alliance)
            for field in fields:
                for name in names:
                    for position, value in zip(positions, results[name][field]):
                        values[field][name][position] = value
    finally:
        environment.close()
    (output / "baselines.json").write_text(json.dumps({"seeds": seeds, **values}, indent=2) + "\n", encoding="utf-8")


def aggregate(plan: dict, output: Path) -> dict:
    if plan.get("features") == "v4":
        return aggregate_tasks(plan, output)
    baselines = json.loads((output / "baselines.json").read_text(encoding="utf-8"))["margins"]
    rows = []
    for candidate in plan["candidates"]:
        record = json.loads((output / f"candidate-run-{candidate['index']:02d}.json").read_text(encoding="utf-8"))
        margins = record["margins"]
        rows.append({**candidate, "independentMeanMargin": statistics.fmean(margins),
                     "baselineMeanMargin": {name: statistics.fmean(baselines[name]) for name in BASELINE_NAMES},
                     "versusLoaderCycle": paired_statistics(margins, baselines[REFERENCE], seed=plan["baseSeed"])})
    primary = next(row for row in rows if row["index"] == plan["primaryRunIndex"])
    return {"schemaVersion": 1, "kind": "tactical-independent-holdout", "campaign": plan["campaign"],
            "seeds": plan["seeds"], "episodes": len(plan["seeds"]["list"]), "candidates": rows,
            "primary": {"index": primary["index"], "seed": primary["seed"],
                        "independentMeanMargin": primary["independentMeanMargin"],
                        "loaderCycleMeanMargin": primary["baselineMeanMargin"][REFERENCE],
                        "versusLoaderCycle": primary["versusLoaderCycle"]},
            "primaryBeatsLoaderCycle": primary["versusLoaderCycle"]["passes"],
            "rule": "primary = best development margin, fixed before these episodes; it passes when the lower end of the "
                    "95% bootstrap interval of its paired margin difference against loader-cycle is above 0",
            "scope": "fresh start poses in the abstract macro-action model only; not a physical tactics or scoring qualification",
            "independentHoldoutPerformed": True, "learnedImprovementVerified": False, "humanApprovalRequired": True}


def aggregate_tasks(plan: dict, output: Path) -> dict:
    """Multi-task result: paired task-return differences against task-greedy and loader-cycle, and success rates."""
    baselines = json.loads((output / "baselines.json").read_text(encoding="utf-8"))
    rows = []
    for candidate in plan["candidates"]:
        record = json.loads((output / f"candidate-run-{candidate['index']:02d}.json").read_text(encoding="utf-8"))
        if record["tasks"] != baselines["tasks"][REFERENCE_V4]:
            raise RuntimeError("a candidate and the baselines did not play the same tasks")
        returns = record["taskReturns"]
        rows.append({**candidate, "independentMeanTaskReturn": statistics.fmean(returns),
                     "successRate": statistics.fmean(float(value) for value in record["achieved"]),
                     "independentMeanMargin": statistics.fmean(record["margins"]),
                     "baselineMeanTaskReturn": {name: statistics.fmean(baselines["taskReturns"][name]) for name in BASELINE_NAMES_V4},
                     "baselineSuccessRate": {name: statistics.fmean(float(value) for value in baselines["achieved"][name])
                                             for name in BASELINE_NAMES_V4},
                     "versusTaskGreedy": paired_statistics(returns, baselines["taskReturns"][REFERENCE_V4], seed=plan["baseSeed"]),
                     "versusLoaderCycle": paired_statistics(returns, baselines["taskReturns"][REFERENCE], seed=plan["baseSeed"])})
    primary = next(row for row in rows if row["index"] == plan["primaryRunIndex"])
    return {"schemaVersion": 1, "kind": "tactical-independent-holdout", "metric": "taskReturn", "campaign": plan["campaign"],
            "seeds": plan["seeds"], "episodes": len(plan["seeds"]["list"]), "candidates": rows,
            "primary": {"index": primary["index"], "seed": primary["seed"],
                        "independentMeanTaskReturn": primary["independentMeanTaskReturn"],
                        "successRate": primary["successRate"],
                        "taskGreedyMeanTaskReturn": primary["baselineMeanTaskReturn"][REFERENCE_V4],
                        "taskGreedySuccessRate": primary["baselineSuccessRate"][REFERENCE_V4],
                        "versusTaskGreedy": primary["versusTaskGreedy"], "versusLoaderCycle": primary["versusLoaderCycle"]},
            "primaryBeatsTaskGreedy": primary["versusTaskGreedy"]["passes"],
            "primaryBeatsLoaderCycle": primary["versusLoaderCycle"]["passes"],
            "rule": "primary = best development task return, fixed before these episodes; it passes when the lower end of "
                    "the 95% bootstrap interval of its paired task-return difference against task-greedy is above 0",
            "scope": "one random task per fresh seed in the abstract macro-action model only; not a physical tactics or scoring qualification",
            "independentHoldoutPerformed": True, "learnedImprovementVerified": False, "humanApprovalRequired": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--approved-amd-mi300x", action="store_true", required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is not None:
        plan = json.loads((args.output / "plan.json").read_text(encoding="utf-8"))
        seeds = plan["seeds"]["list"]
        if args.worker == "baselines":
            baseline_worker(plan["features"], seeds, args.output)
        else:
            candidate = next(row for row in plan["candidates"] if row["index"] == int(args.worker))
            worker(candidate, plan["features"], seeds, args.output)
        return
    if not 1 <= args.top <= 10 or not 2 <= args.episodes <= 1000 or args.episodes % 2:
        parser.error("top must be 1..10 and episodes an even number 2..1000")
    from .campaign import consumed_ranges
    from .train import ROOT, SOURCE_FILES, sha256_file, validate_amd_runtime

    runtime = validate_amd_runtime()
    campaign = json.loads((args.campaign / "campaign.json").read_text(encoding="utf-8"))
    summary = json.loads((args.campaign / "summary.json").read_text(encoding="utf-8"))
    seeds = seed_list(args.base_seed, args.episodes)
    check_seeds(seeds, campaign, consumed_ranges())
    candidates = choose_candidates(args.campaign, campaign, summary, args.top)
    args.output.mkdir(parents=True)  # refuses to reuse an earlier check's folder
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    plan = {"schemaVersion": 1, "kind": "tactical-independent-holdout-plan", "campaign": str(args.campaign),
            "commit": commit, "runtime": runtime, "features": campaign["configuration"]["features"],
            # Must match the campaign runs' run.json sourceSha256, so the rules engine is the one they trained on.
            "sourceSha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
            "baseSeed": args.base_seed, "seeds": {"first": seeds[0], "last": seeds[-1], "list": seeds,
                                                  "alliances": "red for even positions, blue for odd"},
            "candidates": candidates, "primaryRunIndex": candidates[0]["index"],
            "writtenBeforeAnyEpisode": True, "learnedImprovementVerified": False}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    names = ["baselines"] + [str(row["index"]) for row in candidates]
    processes = {name: subprocess.Popen([sys.executable, "-m", "roboproof.tactical_learning.evaluate",
                                         "--approved-amd-mi300x", "--campaign", str(args.campaign),
                                         "--output", str(args.output), "--base-seed", str(args.base_seed),
                                         "--worker", name],
                                        stdout=(args.output / f"worker-{name}.log").open("w"), stderr=subprocess.STDOUT)
                 for name in names}
    codes = {name: process.wait() for name, process in processes.items()}
    if any(codes.values()):
        raise SystemExit(f"evaluation workers failed: {codes}")
    result = aggregate(plan, args.output)
    result["elapsedSeconds"] = round(time.monotonic() - started, 1)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    keys = ("primary", "primaryBeatsTaskGreedy", "primaryBeatsLoaderCycle", "elapsedSeconds") if "primaryBeatsTaskGreedy" in result \
        else ("primary", "primaryBeatsLoaderCycle", "elapsedSeconds")
    print(json.dumps({key: result[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
