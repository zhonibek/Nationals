import contextlib
import io
import json
import math
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from roboproof.tactical_learning import train
from roboproof.tactical_learning.model import FEATURE_COUNT, TacticalActorCritic


# Keys the trainer wrote to run.json before the warm start existed; the default path must keep exactly these.
DEFAULT_RUN_KEYS = {"schemaVersion", "runId", "model", "status", "seed", "episodesRequested", "maxSeconds",
                    "evaluationEpisodes", "featureCount", "width", "hiddenLayers", "runtime", "sourceSha256", "reward",
                    "movementAbstraction", "tacticalTiming", "trainingPerformed", "learnedImprovementVerified",
                    "independentQualificationPerformed", "promotionAllowed", "warning", "episodesCompleted",
                    "elapsedSeconds", "evaluationEpisodesCompleted", "bridgeShutdown"}
TRAINING_EXPOSURE_KEYS = {"episode", "seed", "alliance", "decisions", "runId"}
CANDIDATES = [{"id": "score-pin:red-1:pin-02:g-red-sw", "features": [0.0] * FEATURE_COUNT},
              {"id": "load-score-cup:red-1:loader-red-1:g-red-sw", "features": [1.0] * FEATURE_COUNT}]
LOAD = CANDIDATES[1]["id"]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


class FakeStdin:
    def __init__(self, process):
        self.process = process

    def write(self, text):
        for line in text.splitlines():
            self.process.receive(json.loads(line))

    def flush(self):
        pass

    def close(self):
        pass


class FakeStdout:
    def __init__(self):
        self.lines = []

    def readline(self):
        return self.lines.pop(0) if self.lines else ""

    def read(self):
        text, self.lines = "".join(self.lines), []
        return text

    def close(self):
        pass


class ScriptedBridge:
    """Answers bridge.js requests for two-decision episodes; `choose(decision)` is the demonstrated ID."""

    def __init__(self, choose=lambda decision: LOAD, clock=None, seconds_per_reset=0.0):
        self.choose, self.clock, self.seconds_per_reset = choose, clock, seconds_per_reset
        self.stdin, self.stdout, self.stderr = FakeStdin(self), FakeStdout(), io.StringIO("")
        self.returncode, self.requests, self.decisions = None, [], 0

    def observation(self):
        done = self.decisions >= 2
        return {"done": done, "actions": [] if done else CANDIDATES, "clock": 15.0 + 2 * self.decisions,
                "score": {"red": 4 * self.decisions, "blue": 0, "margin": 4 * self.decisions},
                "decisions": self.decisions, "timing": "measured"}

    def receive(self, request):
        self.requests.append(request)
        operation = request["operation"]
        if operation == "reset":
            self.decisions = 0
            if self.clock:
                self.clock.now += self.seconds_per_reset
            self.reply({"ok": True, "observation": self.observation()})
        elif operation == "demonstrate":
            self.reply({"ok": True, "policy": request["policy"], "actionId": self.choose(self.decisions), "candidate": None})
        elif operation == "step":
            self.decisions += 1
            self.reply({"ok": True, "observation": self.observation(), "reward": 4.0, "durationSeconds": 2.0,
                        "score": self.observation()["score"], "done": self.decisions >= 2})
        elif operation == "baselines":
            names = ("loader-cycle", "field-greedy", "greedy-all", "random")
            self.reply({"ok": True, "baselines": {name: {"meanMargin": 1.0, "margins": [1.0] * len(request["seeds"])}
                                                  for name in names}})
        elif operation == "close":
            self.reply({"ok": True})
            self.returncode = 0

    def reply(self, value):
        self.stdout.lines.append(json.dumps(value, separators=(",", ":")) + "\n")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("node", timeout)
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def operations(self):
        return [request["operation"] for request in self.requests]


@contextlib.contextmanager
def bridge(process):
    with mock.patch.object(train.shutil, "which", return_value="node"), \
            mock.patch.object(train.subprocess, "Popen", return_value=process):
        yield train.NodeTacticalEnvironment()


class FakeModel:
    width = 8

    def to(self, device):
        return self

    def parameters(self):
        return []

    def eval(self):
        return self

    def state_dict(self):
        return {}


def fake_pretrain(calls, clock=None):
    def pretrain(model, trajectories, device, epochs, deadline, seed):
        calls.append({"event": "pretrain", "trajectories": trajectories, "epochs": epochs, "deadline": deadline,
                      "seed": seed, "now": clock.now if clock else None})
        samples = sum(len(trajectory) for trajectory in trajectories)
        metrics = {"actorLoss": 0.25, "criticLoss": 1.5, "accuracy": 0.9}
        return {"samples": samples, "epochsCompleted": epochs, "stopReason": "all-requested-epochs",
                "initial": {**metrics, "accuracy": 0.5}, "final": metrics,
                "history": [{"epoch": 1, "loss": 1.0, "actorLoss": 0.5, "criticLoss": 1.0}]}
    return pretrain


@contextlib.contextmanager
def fake_fit(output, process, *extra, episodes=2, max_seconds=60, evaluation_episodes=2, clock=None, seconds_per_episode=0.0):
    # No real model, update, pretraining or checkpoint work runs: only the trainer's bookkeeping.
    calls, saved = [], []

    def run_episode(environment, model, seed, alliance, device, deterministic=False):
        calls.append({"event": "evaluation" if deterministic else "episode", "seed": seed})
        if clock:
            clock.now += seconds_per_episode
        return [{"features": [], "choice": 0, "reward": 1.0, "durationSeconds": 1.0}], \
            {"seed": seed, "alliance": alliance, "score": {"red": 1, "blue": 0, "margin": 1},
             "clock": 105, "decisions": 1, "episodeReturn": 1.0, "done": True}

    fake_torch = types.SimpleNamespace(device=lambda name: name, manual_seed=lambda seed: None,
                                       cuda=types.SimpleNamespace(manual_seed_all=lambda seed: None),
                                       optim=types.SimpleNamespace(Adam=lambda parameters, lr: types.SimpleNamespace(
                                           state_dict=dict)))
    argv = ["train", "--approved-free-t4", "--seed", "12345", "--episodes", str(episodes), "--max-seconds",
            str(max_seconds), "--evaluation-episodes", str(evaluation_episodes), "--output", str(output), *extra]
    time_module = types.SimpleNamespace(monotonic=clock.monotonic) if clock else train.time
    with mock.patch.object(train.sys, "argv", argv), \
            mock.patch.object(train, "validate_runtime", return_value={"device": "cuda:0", "gpu": "fake"}), \
            mock.patch.object(train, "torch", fake_torch), \
            mock.patch.object(train, "time", time_module), \
            mock.patch.object(train, "TacticalActorCritic", FakeModel), \
            mock.patch.object(train, "run_episode", side_effect=run_episode), \
            mock.patch.object(train, "update", return_value={"loss": 0.0, "samples": 1}), \
            mock.patch.object(train, "pretrain", side_effect=fake_pretrain(calls, clock)), \
            mock.patch.object(train, "atomic_torch_save", side_effect=lambda value, path: saved.append((path.name, value))), \
            mock.patch.object(train.shutil, "which", return_value="node"), \
            mock.patch.object(train.subprocess, "Popen", return_value=process), \
            contextlib.redirect_stdout(io.StringIO()) as printed:
        yield types.SimpleNamespace(calls=calls, saved=saved, printed=printed)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def fit(*extra, process=None, **options):
    """Run main() in a temporary folder and return everything it wrote."""
    process = process or ScriptedBridge()
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "run"
        with fake_fit(output, process, *extra, **options) as recorded:
            train.main()
        files = {path.name for path in output.iterdir()}
        return types.SimpleNamespace(
            process=process, files=files, calls=recorded.calls, saved=recorded.saved,
            run=json.loads((output / "run.json").read_text("utf-8")),
            evaluation=json.loads((output / "evaluation.json").read_text("utf-8")),
            exposure=read_jsonl(output / "exposure.jsonl"),
            warm_log=read_jsonl(output / "warm-start.jsonl") if "warm-start.jsonl" in files else None,
            printed=[json.loads(line) for line in recorded.printed.getvalue().splitlines()])


class WarmStartArgumentTests(unittest.TestCase):
    def parse(self, *extra):
        with mock.patch.object(train.sys, "argv", ["train", "--approved-free-t4", *extra]):
            return train.parse_args()

    def test_the_warm_start_is_off_by_default_and_bounded_when_on(self):
        args = self.parse()
        self.assertEqual((args.warm_start_demonstration_episodes, args.warm_start_epochs), (0, None))
        self.assertEqual((args.seed, args.episodes, args.max_seconds, args.evaluation_episodes), (731001, 100, 300, 12))
        args = self.parse("--warm-start-demonstration-episodes", "32")
        self.assertEqual((args.warm_start_demonstration_episodes, args.warm_start_epochs), (32, train.WARM_START_DEFAULT_EPOCHS))
        self.assertEqual(self.parse("--warm-start-demonstration-episodes", "1", "--warm-start-epochs", "50").warm_start_epochs, 50)
        for extra in (["--warm-start-epochs", "50"], ["--warm-start-demonstration-episodes", "201"],
                      ["--warm-start-demonstration-episodes", "-1"],
                      ["--warm-start-demonstration-episodes", "4", "--warm-start-epochs", "0"]):
            with self.subTest(extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(*extra)


class DemonstrationEpisodeTests(unittest.TestCase):
    def test_the_bridge_is_asked_for_the_loader_cycle_choice(self):
        process = ScriptedBridge()
        with bridge(process) as environment:
            self.assertEqual(environment.demonstrate(), LOAD)
        self.assertEqual(process.requests, [{"operation": "demonstrate", "policy": "loader-cycle"}])

    def test_demonstrated_choices_become_transitions_like_rl_episodes(self):
        process = ScriptedBridge()
        with bridge(process) as environment:
            trajectory, result = train.run_demonstration_episode(environment, 77, "blue")
        self.assertEqual(process.operations(), ["reset", "demonstrate", "step", "demonstrate", "step"])
        self.assertEqual([request.get("actionId") for request in process.requests if request["operation"] == "step"], [LOAD, LOAD])
        self.assertEqual(trajectory, [{"features": [candidate["features"] for candidate in CANDIDATES], "choice": 1,
                                       "reward": 4.0, "durationSeconds": 2.0}] * 2)
        self.assertEqual((result["seed"], result["alliance"], result["policy"], result["decisions"]), (77, "blue", "loader-cycle", 2))
        self.assertEqual((result["score"]["margin"], result["episodeReturn"], result["stopReason"]), (8, 8.0, "episode-complete"))

    def test_no_choice_stops_and_an_unknown_choice_fails_closed(self):
        with bridge(ScriptedBridge(choose=lambda decision: LOAD if decision == 0 else None)) as environment:
            trajectory, result = train.run_demonstration_episode(environment, 77, "red")
        self.assertEqual((len(trajectory), result["stopReason"], result["done"]), (1, "no-choice", False))
        with bridge(ScriptedBridge(choose=lambda decision: "park:red-9")) as environment:
            with self.assertRaisesRegex(RuntimeError, "not in the observed candidate set"):
                train.run_demonstration_episode(environment, 77, "red")

    def test_returns_are_the_rl_discounted_returns(self):
        rows = train.discounted_rows([[{"features": "a", "choice": 0, "reward": 2.0, "durationSeconds": 1.0},
                                       {"features": "b", "choice": 1, "reward": 4.0, "durationSeconds": 2.0}]])
        self.assertEqual([(row["features"], row["choice"]) for row in rows], [("b", 1), ("a", 0)])
        self.assertEqual(rows[0]["return"], 4.0)
        self.assertAlmostEqual(rows[1]["return"], 2.0 + 0.995 * 4.0)


class NoOpAdam:
    """Stands in for Adam and never changes a parameter, so this test computes losses without training anything."""

    def __init__(self, parameters, lr):
        self.parameters, self.lr, self.steps = list(parameters), lr, 0

    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        self.steps += 1


class PretrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.model = TacticalActorCritic(width=16)
        values = (row / 7.0 for row in range(10 ** 6))
        transition = lambda count, choice, reward: {"features": [[next(values) % 1.0 for _ in range(FEATURE_COUNT)] for _ in range(count)],
                                                    "choice": choice, "reward": reward, "durationSeconds": 2.0}
        self.trajectories = [[transition(3, 2, 4.0), transition(2, 0, 0.0)], [transition(4, 1, 6.0)]]
        self.optimizers = []

    def pretrain(self, epochs, deadline):
        def adam(parameters, lr):
            self.optimizers.append(NoOpAdam(parameters, lr))
            return self.optimizers[-1]
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        rng = torch.get_rng_state()
        with mock.patch.object(train.torch.optim, "Adam", adam):
            result = train.pretrain(self.model, self.trajectories, torch.device("cpu"), epochs, deadline, 12345)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in self.model.state_dict().items()),
                        "the no-op optimizer must leave every parameter unchanged")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()), "pretraining must not consume the global RNG")
        return result

    def test_losses_and_accuracy_are_measured_on_padded_demonstrations(self):
        result = self.pretrain(epochs=2, deadline=math.inf)
        self.assertEqual((result["samples"], result["epochsCompleted"], result["stopReason"]), (3, 2, "all-requested-epochs"))
        self.assertEqual((self.optimizers[0].lr, self.optimizers[0].steps), (train.WARM_START_LEARNING_RATE, 2))
        self.assertEqual([row["epoch"] for row in result["history"]], [1, 2])
        self.assertEqual(result["initial"], result["final"])
        for metrics in (result["initial"], *result["history"]):
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertTrue(0.0 <= result["final"]["accuracy"] <= 1.0)
        self.assertGreater(result["final"]["actorLoss"], 0.0)

    def test_pretraining_stops_at_its_deadline(self):
        result = self.pretrain(epochs=5, deadline=-math.inf)
        self.assertEqual((result["epochsCompleted"], result["stopReason"], result["history"]), (0, "time-budget", []))
        self.assertEqual(self.optimizers[0].steps, 0)


class WarmStartFitTests(unittest.TestCase):
    def test_the_default_path_is_unchanged(self):
        result = fit()
        self.assertEqual(set(result.run), DEFAULT_RUN_KEYS)
        self.assertEqual((result.run["status"], result.run["episodesCompleted"], result.run["evaluationEpisodesCompleted"]),
                         ("completed", 2, 2))
        self.assertEqual(result.files, {"run.json", "episodes.jsonl", "exposure.jsonl", "training.jsonl", "evaluation.json"})
        self.assertNotIn("warmStart", result.evaluation)
        self.assertEqual([set(row) for row in result.exposure], [TRAINING_EXPOSURE_KEYS] * 2)
        self.assertEqual([row["seed"] for row in result.exposure], [12345, 12346])
        self.assertEqual(result.process.operations(), ["baselines", "baselines", "close"])
        self.assertNotIn("pretrain", [call["event"] for call in result.calls])
        self.assertEqual([(name, "warmStart" in value) for name, value in result.saved], [("latest.pt", False)])
        self.assertNotIn("warm-start", [row["event"] for row in result.printed])

    def test_the_default_path_keeps_its_budget_behavior(self):
        # Unchanged known limitation: without a warm start, RL may use the whole budget and skip the evaluation.
        result = fit(episodes=10, clock=Clock(), seconds_per_episode=10.0)
        self.assertEqual((result.run["status"], result.run["episodesCompleted"], result.run["evaluationEpisodesCompleted"]),
                         ("time-budget-exhausted", 6, 0))

    def test_demonstrations_are_exposure_and_pretrain_the_model_before_rl(self):
        result = fit("--warm-start-demonstration-episodes", "2")
        operations = result.process.operations()
        self.assertEqual(operations[:10], ["reset", "demonstrate", "step", "demonstrate", "step"] * 2)
        self.assertEqual(operations[10:], ["baselines", "baselines", "close"])
        self.assertEqual({request["policy"] for request in result.process.requests if request["operation"] == "demonstrate"},
                         {"loader-cycle"})
        # Demonstration seeds follow the predeclared training block (12345, 12346).
        self.assertEqual([(row["seed"], row["alliance"]) for row in result.process.requests if row["operation"] == "reset"],
                         [(12347, "red"), (12348, "blue")])
        self.assertEqual(result.exposure[:2], [
            {"phase": "demonstration", "demonstrationEpisode": index + 1, "seed": seed, "alliance": alliance,
             "decisions": 2, "policy": "loader-cycle", "runId": result.run["runId"]}
            for index, (seed, alliance) in enumerate([(12347, "red"), (12348, "blue")])])
        self.assertEqual([set(row) for row in result.exposure[2:]], [TRAINING_EXPOSURE_KEYS] * 2)
        self.assertEqual([row["seed"] for row in result.exposure[2:]], [12345, 12346])

        events = [call["event"] for call in result.calls]
        self.assertEqual(events, ["pretrain", "episode", "episode", "evaluation", "evaluation"])
        pretrain = result.calls[0]
        self.assertEqual((pretrain["epochs"], pretrain["seed"], len(pretrain["trajectories"])), (train.WARM_START_DEFAULT_EPOCHS, 12345, 2))
        self.assertEqual({transition["choice"] for trajectory in pretrain["trajectories"] for transition in trajectory}, {1})

        warm = result.run["warmStart"]
        self.assertEqual(set(result.run) - DEFAULT_RUN_KEYS, {"warmStart"})
        self.assertEqual((warm["status"], warm["policy"], warm["demonstrationEpisodesRequested"],
                          warm["demonstrationEpisodesCompleted"], warm["demonstrationSeeds"], warm["demonstrationSamples"]),
                         ("completed", "loader-cycle", 2, 2, [12347, 12348], 4))
        self.assertEqual((warm["predeclaredDemonstrationSeeds"]["first"], warm["predeclaredDemonstrationSeeds"]["last"]), (12347, 12348))
        self.assertEqual((warm["demonstrationMeanScoreMargin"], warm["collectionStopReason"], warm["checkpoint"]),
                         (8.0, "all-requested-episodes", "warm-start.pt"))
        self.assertEqual(warm["pretraining"]["final"], {"actorLoss": 0.25, "criticLoss": 1.5, "accuracy": 0.9})
        self.assertNotIn("history", warm["pretraining"])
        self.assertEqual((warm["collectionDeadlineSeconds"], warm["deadlineSeconds"]), (24.0, 30.0))
        self.assertIs(warm["reinforcementStoppedForEvaluationReserve"], False)
        self.assertIn("evaluationReserveSeconds", warm)
        self.assertEqual((result.run["status"], result.run["episodesCompleted"], result.run["evaluationEpisodesCompleted"]),
                         ("completed", 2, 2))
        self.assertIs(result.run["learnedImprovementVerified"], False)
        self.assertIs(result.run["independentQualificationPerformed"], False)

        self.assertEqual(result.files, {"run.json", "episodes.jsonl", "exposure.jsonl", "training.jsonl", "evaluation.json",
                                        "warm-start.jsonl"})
        self.assertEqual([row["event"] for row in result.warm_log], ["demonstration", "demonstration", "pretrain-epoch"])
        self.assertEqual([row["seed"] for row in result.warm_log[:2]], [12347, 12348])
        self.assertEqual([name for name, _ in result.saved], ["warm-start.pt", "latest.pt"])
        for name, checkpoint in result.saved:
            self.assertEqual((checkpoint["warmStart"]["demonstrationSeeds"], checkpoint["learnedImprovementVerified"]),
                             ([12347, 12348], False))
        self.assertEqual(result.saved[0][1]["stage"], "behavior-cloning-warm-start")
        self.assertEqual(result.evaluation["warmStart"], {"policy": "loader-cycle", "demonstrationEpisodes": 2})
        self.assertIs(result.evaluation["learnedImprovementVerified"], False)
        printed = [row for row in result.printed if row["event"] == "warm-start"]
        self.assertEqual((printed[0]["demonstrationEpisodes"], printed[0]["samples"]), (2, 4))

    def test_the_warm_start_shares_the_budget_and_leaves_time_for_the_evaluation(self):
        clock = Clock()
        process = ScriptedBridge(clock=clock, seconds_per_reset=10.0)
        result = fit("--warm-start-demonstration-episodes", "10", process=process, episodes=10, clock=clock,
                     seconds_per_episode=10.0)
        warm = result.run["warmStart"]
        # Demonstrations start at 0, 10 and 20 s; the 40 % share (24 s) stops the fourth. Pretraining must end by 30 s.
        self.assertEqual((warm["demonstrationEpisodesCompleted"], warm["collectionStopReason"]), (3, "time-budget"))
        self.assertEqual(result.calls[0]["deadline"], 1000.0 + 30.0)
        # RL starts at 30 s; after one 10 s episode the reserve (2 x 2 evaluation episodes x 10 s) ends RL at 40 s.
        self.assertIs(warm["reinforcementStoppedForEvaluationReserve"], True)
        self.assertEqual(warm["evaluationReserveSeconds"], 40.0)
        self.assertEqual((result.run["status"], result.run["episodesCompleted"], result.run["evaluationEpisodesCompleted"]),
                         ("time-budget-exhausted", 1, 2))
        self.assertEqual([row.get("phase", "training") for row in result.exposure], ["demonstration"] * 3 + ["training"])

    def test_a_fit_without_demonstrations_skips_pretraining(self):
        result = fit("--warm-start-demonstration-episodes", "2", process=ScriptedBridge(choose=lambda decision: None))
        warm = result.run["warmStart"]
        self.assertEqual((warm["status"], warm["demonstrationEpisodesCompleted"], warm["demonstrationSamples"], warm["pretraining"]),
                         ("skipped-no-demonstrations", 2, 0, None))
        self.assertNotIn("pretrain", [call["event"] for call in result.calls])
        self.assertNotIn("checkpoint", warm)
        self.assertEqual([name for name, _ in result.saved], ["latest.pt"])
        self.assertEqual(result.run["status"], "completed")


if __name__ == "__main__":
    unittest.main()
