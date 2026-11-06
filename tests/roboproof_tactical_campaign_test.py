import json
import tempfile
import unittest
from pathlib import Path

from roboproof.tactical_learning import campaign

DEFAULTS = dict(runs=40, base_seed=960001, seed_stride=5000, episodes=3000, warm_start=32, max_seconds=7200,
                evaluation_episodes=8, algorithm="ppo", features="v3", width=1024, hidden_layers=10,
                hourly_rate_usd=1.99, approved_max_cost_usd=20.0)


class CampaignPlanTests(unittest.TestCase):
    def plan(self, **changes):
        return campaign.plan(output_root=Path("campaign"), **{**DEFAULTS, **changes})

    def test_forty_runs_get_disjoint_predeclared_seeds_and_the_requested_model(self):
        result = self.plan()
        runs = result["runs"]
        self.assertEqual(len(runs), 40)
        ranges = sorted((run["trainingSeeds"][0], run["demonstrationSeeds"][1]) for run in runs)
        self.assertTrue(all(previous[1] < current[0] for previous, current in zip(ranges, ranges[1:])))
        command = runs[0]["command"]
        for flag, value in (("--width", "1024"), ("--hidden-layers", "10"), ("--features", "v3"), ("--algorithm", "ppo")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--approved-amd-mi300x", command)
        self.assertEqual(result["wallClockLimitSeconds"], 7200 + campaign.EVALUATION_RESERVE_SECONDS)
        self.assertAlmostEqual(result["estimatedMaxCostUsd"], 5.97)
        self.assertIs(result["learnedImprovementVerified"], False)

    def test_consumed_seeds_cost_and_bounds_are_refused(self):
        refusals = [
            (dict(base_seed=735101), "overlap consumed seeds"),
            (dict(seed_stride=1000), "seed-stride"),
            (dict(approved_max_cost_usd=5.0), "exceeds the approved ceiling"),
            (dict(approved_max_cost_usd=60.0), "approved ceiling"),
            (dict(max_seconds=7 * 3600), "max-seconds"),
            (dict(runs=65), "runs must be"),
        ]
        for changes, message in refusals:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.plan(**changes)

    def test_the_registry_holds_every_earlier_tactical_fit_including_the_lost_one(self):
        notes = " ".join(note for _, _, note in campaign.consumed_ranges())
        for seed in ("731201", "731401", "732701", "734001", "735001", "736001"):
            self.assertIn(seed, notes)


class CampaignSummaryTests(unittest.TestCase):
    def test_summary_counts_runs_margins_and_gpu_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = campaign.plan(output_root=root, **{**DEFAULTS, "runs": 2})
            (root / "campaign.json").write_text(json.dumps(plan), encoding="utf-8")
            for index, (margin, beats) in enumerate(((210.0, False), (251.5, True))):
                run = root / f"run-{index:02d}"
                run.mkdir()
                (run / "run.json").write_text(json.dumps({"status": "completed", "episodesCompleted": 600}), encoding="utf-8")
                (run / "evaluation.json").write_text(json.dumps({"meanScoreMargin": margin, "beatsLoaderCycleOnHoldout": beats,
                                                                  "baselineMeanScoreMargin": {"loader-cycle": 244.5}}), encoding="utf-8")
                (root / f"run-{index:02d}.exit-code").write_text("0\n", encoding="utf-8")
            samples = [{"sample": {"card0": {"GPU use (%)": "62", "GPU Memory Allocated (VRAM%)": "40"}}},
                       {"sample": {"card0": {"GPU use (%)": "58"}}}, {"sample": None}]
            (root / "gpu-utilization.jsonl").write_text("\n".join(json.dumps(row) for row in samples) + "\nnot json\n",
                                                        encoding="utf-8")
            summary = campaign.summarize(root)
        self.assertEqual(summary["completedRuns"], 2)
        self.assertEqual(summary["runsBeatingLoaderCycle"], 1)
        self.assertEqual(summary["bestRun"]["learnedMargin"], 251.5)
        self.assertEqual(summary["gpuBusyPercent"], {"samples": 2, "mean": 60.0, "max": 62.0})
        self.assertIs(summary["independentQualificationPerformed"], False)

    def test_amd_smi_nested_values_are_read_and_memory_activity_is_ignored(self):
        sample = {"gpu_data": [{"gpu": 0, "usage": {"gfx_activity": {"value": 37, "unit": "%"},
                                                    "umc_activity": {"value": 90, "unit": "%"},
                                                    "mm_activity": "N/A"}}]}
        self.assertEqual(campaign.utilization_values(sample), [37.0])


if __name__ == "__main__":
    unittest.main()
