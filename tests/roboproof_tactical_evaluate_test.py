import json
import tempfile
import unittest
from pathlib import Path

from roboproof.tactical_learning import evaluate

CAMPAIGN = {"configuration": {"evaluationEpisodes": 8, "features": "v3"},
            "runs": [{"index": 0, "seed": 1400001, "trainingSeeds": [1400001, 1403000],
                      "demonstrationSeeds": [1403001, 1403032], "output": "campaign/run-00"},
                     {"index": 1, "seed": 1405001, "trainingSeeds": [1405001, 1408000],
                      "demonstrationSeeds": [1408001, 1408032], "output": "campaign/run-01"}]}


class SeedTests(unittest.TestCase):
    def test_fresh_seeds_pass_and_every_kind_of_reuse_is_refused(self):
        evaluate.check_seeds(evaluate.seed_list(2100001, 200), CAMPAIGN, consumed=[(731001, 731200, "pilot")])
        evaluation_seed = (1405001 ^ evaluate.EVALUATION_SEED_MASK ^ 3) & 0xFFFFFFFF
        refusals = [([1402990, 1402991], "trainingSeeds"), ([1403010, 1403011], "demonstrationSeeds"),
                    ([evaluation_seed, evaluation_seed + 1], "evaluation seeds"), ([731100, 731101], "consumed seeds")]
        for seeds, message in refusals:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                evaluate.check_seeds(seeds, CAMPAIGN, consumed=[(731001, 731200, "pilot")])

    def test_alliances_alternate(self):
        self.assertEqual([evaluate.alliance_for(index) for index in range(4)], ["red", "blue", "red", "blue"])


class CandidateTests(unittest.TestCase):
    def test_top_runs_by_development_margin_skip_failed_or_checkpointless_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = {**CAMPAIGN, "runs": [dict(CAMPAIGN["runs"][0]), dict(CAMPAIGN["runs"][1]),
                                             {**CAMPAIGN["runs"][1], "index": 2, "seed": 1410001, "output": "campaign/run-02"},
                                             {**CAMPAIGN["runs"][1], "index": 3, "seed": 1415001, "output": "campaign/run-03"}]}
            for index in (0, 1, 3):
                (root / f"run-{index:02d}").mkdir()
                (root / f"run-{index:02d}" / "latest.pt").write_bytes(b"weights")
            summary = {"runs": [{"index": 0, "seed": 1400001, "status": "completed", "learnedMargin": 250.0},
                                {"index": 1, "seed": 1405001, "status": "time-budget-exhausted", "learnedMargin": 255.0},
                                {"index": 2, "seed": 1410001, "status": "completed", "learnedMargin": 260.0},
                                {"index": 3, "seed": 1415001, "status": "interrupted", "learnedMargin": 270.0}]}
            chosen = evaluate.choose_candidates(root, campaign, summary, top=5)
        self.assertEqual([row["index"] for row in chosen], [1, 0])
        self.assertEqual(len(chosen[0]["checkpointSha256"]), 64)


class StatisticsTests(unittest.TestCase):
    def test_a_clear_paired_gain_passes_and_a_noisy_one_does_not(self):
        reference = [240.0, 245.0, 250.0, 238.0] * 25
        clear = evaluate.paired_statistics([value + 5 for value in reference], reference, resamples=2000, seed=1)
        self.assertEqual((clear["wins"], clear["ties"], clear["losses"]), (100, 0, 0))
        self.assertAlmostEqual(clear["meanDifference"], 5.0)
        self.assertTrue(clear["passes"])
        noisy = evaluate.paired_statistics([value + (12 if index % 2 else -11) for index, value in enumerate(reference)],
                                           reference, resamples=2000, seed=1)
        self.assertFalse(noisy["passes"])
        self.assertLess(noisy["bootstrap95"][0], 0)
        self.assertGreater(noisy["bootstrap95"][1], 0)

    def test_the_same_seed_gives_the_same_interval(self):
        learned, reference = [250.0, 241.0, 260.0, 239.0], [244.0, 244.0, 244.0, 244.0]
        self.assertEqual(evaluate.paired_statistics(learned, reference, resamples=500, seed=7)["bootstrap95"],
                         evaluate.paired_statistics(learned, reference, resamples=500, seed=7)["bootstrap95"])

    def test_aggregate_reports_the_predeclared_primary_and_never_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            seeds = [2100001, 2100002, 2100003, 2100004]
            baselines = {name: [244.0, 240.0, 250.0, 246.0] for name in evaluate.BASELINE_NAMES}
            (output / "baselines.json").write_text(json.dumps({"seeds": seeds, "margins": baselines}), encoding="utf-8")
            for index, margins in ((4, [254.0, 251.0, 259.0, 257.0]), (9, [200.0, 210.0, 205.0, 199.0])):
                (output / f"candidate-run-{index:02d}.json").write_text(json.dumps({"margins": margins}), encoding="utf-8")
            plan = {"campaign": "c", "baseSeed": 2100001, "seeds": {"list": seeds}, "primaryRunIndex": 4,
                    "candidates": [{"index": 4, "seed": 1, "developmentMargin": 262.0},
                                   {"index": 9, "seed": 2, "developmentMargin": 258.0}]}
            result = evaluate.aggregate(plan, output)
        self.assertEqual(result["primary"]["index"], 4)
        self.assertTrue(result["primaryBeatsLoaderCycle"])
        self.assertFalse(result["candidates"][1]["versusLoaderCycle"]["passes"])
        self.assertIs(result["learnedImprovementVerified"], False)
        self.assertIs(result["humanApprovalRequired"], True)


if __name__ == "__main__":
    unittest.main()
