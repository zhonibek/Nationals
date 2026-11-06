import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from roboproof.tactical_learning import train
from roboproof.tactical_learning.model import FEATURE_COUNT, TacticalActorCritic
from tests.roboproof_tactical_warm_start_test import ScriptedBridge, fake_fit, read_jsonl


class LoggedEnvironment:
    """A bridge stand-in that records the order of sends and replies."""

    def __init__(self, name, log, candidates):
        self.name, self.log, self.candidates, self.pending = name, log, candidates, None

    def send(self, payload):
        self.log.append(("send", self.name))
        self.pending = payload

    def receive(self):
        self.log.append(("receive", self.name))
        return {"ok": True, "reward": 1.5, "durationSeconds": 2.0,
                "observation": {"done": True, "actions": [], "score": {"margin": 3}, "clock": 20, "decisions": 1}}


class LockstepStepTests(unittest.TestCase):
    def test_one_batched_forward_then_all_sends_before_any_reply(self):
        torch.manual_seed(3)
        model = TacticalActorCritic(width=16)
        log = []
        counts = (2, 5, 3)
        environments, slots = [], []
        for index, count in enumerate(counts):
            actions = [{"id": f"a{index}-{k}", "features": [float(k)] * FEATURE_COUNT} for k in range(count)]
            environments.append(LoggedEnvironment(index, log, actions))
            slots.append({"observation": {"done": False, "actions": actions}, "trajectory": [], "return": 0.0})
        train.run_parallel_step(environments, slots, [0, 1, 2], model, torch.device("cpu"))
        self.assertEqual(log, [("send", 0), ("send", 1), ("send", 2), ("receive", 0), ("receive", 1), ("receive", 2)])
        for index, count in enumerate(counts):
            transition = slots[index]["trajectory"][0]
            self.assertTrue(0 <= transition["choice"] < count)
            self.assertEqual(environments[index].pending["actionId"], f"a{index}-{transition['choice']}")
            self.assertEqual(len(transition["features"]), count)
            self.assertLessEqual(transition["logProbability"], 0.0)
            self.assertEqual(slots[index]["return"], 1.5)
            self.assertTrue(slots[index]["observation"]["done"])


def fake_parallel_step(environments, slots, active, model, device):
    for index in active:
        slot = slots[index]
        action = slot["observation"]["actions"][0]
        result = environments[index].request({"operation": "step", "actionId": action["id"]})
        slot["trajectory"].append({"features": [], "choice": 0, "reward": float(result["reward"]),
                                   "durationSeconds": float(result["durationSeconds"]), "logProbability": 0.0, "value": 0.0})
        slot["return"] += float(result["reward"])
        slot["observation"] = result["observation"]


class ParallelFitTests(unittest.TestCase):
    def test_parallel_environments_play_every_predeclared_seed_once(self):
        bridges = [ScriptedBridge() for _ in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with fake_fit(output, bridges[0], "--parallel-envs", "3", episodes=7) as recorded, \
                    mock.patch.object(train.subprocess, "Popen", side_effect=bridges), \
                    mock.patch.object(train, "run_parallel_step", side_effect=fake_parallel_step):
                train.main()
            run = json.loads((output / "run.json").read_text("utf-8"))
            exposure = read_jsonl(output / "exposure.jsonl")
            episodes = read_jsonl(output / "episodes.jsonl")
        self.assertEqual(run["parallelEnvironments"], 3)
        self.assertEqual((run["status"], run["episodesCompleted"]), ("completed", 7))
        self.assertEqual(sorted(row["seed"] for row in exposure), [12345 + episode for episode in range(7)])
        self.assertEqual([row["episode"] for row in episodes], list(range(1, 8)))
        self.assertTrue(all(row["episodeReturn"] == 8.0 for row in episodes))  # two steps of reward 4
        resets = [bridge.operations().count("reset") for bridge in bridges]
        self.assertEqual(sum(resets), 7)
        self.assertTrue(all(count >= 2 for count in resets))
        self.assertIn("baselines", bridges[0].operations())  # the held-out evaluation runs on the first bridge only
        self.assertNotIn("baselines", bridges[1].operations() + bridges[2].operations())
        self.assertEqual(len(run["parallelBridgeShutdown"]), 2)
        self.assertTrue(all(bridge.returncode == 0 for bridge in bridges))
        self.assertFalse(recorded.calls and any(call["event"] == "episode" for call in recorded.calls))


class ModelOnlyCopyTests(unittest.TestCase):
    def fit(self, *extra):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with fake_fit(output, ScriptedBridge(), *extra, episodes=8) as recorded:
                train.main()
            return recorded.saved

    def test_the_final_model_is_saved_without_optimizer_state_only_when_asked(self):
        self.assertEqual([name for name, _ in self.fit()], ["latest.pt"])
        saved = self.fit("--model-only-copy")
        self.assertEqual([name for name, _ in saved], ["latest.pt", "model-final.pt"])
        final = dict(saved)["model-final.pt"]
        self.assertNotIn("optimizerState", final)
        self.assertEqual((final["stage"], final["episodesCompleted"], final["learnedImprovementVerified"]),
                         ("final-model-only", 8, False))


if __name__ == "__main__":
    unittest.main()
