import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from roboproof.tactical_learning import train


class FakeStdin:
    def __init__(self, process):
        self.process = process
        self.closed = False

    def write(self, text):
        if self.process.broken_pipe:
            raise BrokenPipeError(32, "Broken pipe")
        for line in text.splitlines():
            self.process.receive(json.loads(line))

    def flush(self):
        pass

    def close(self):
        self.closed = True
        if self.process.stops_on == "eof":
            self.process.returncode = 0


class FakeStdout:
    def __init__(self):
        self.lines = []
        self.closed = False

    def readline(self):
        return self.lines.pop(0) if self.lines else ""

    def read(self):
        text, self.lines = "".join(self.lines), []
        return text

    def close(self):
        self.closed = True


class FakeBridgeProcess:
    """Answers bridge requests like bridge.js; `stops_on` is the first shutdown step that ends it."""

    def __init__(self, stops_on="close", returncode=None, broken_pipe=False, on_wait=None):
        self.stops_on, self.returncode, self.broken_pipe, self.on_wait = stops_on, returncode, broken_pipe, on_wait
        self.stdin, self.stdout, self.stderr = FakeStdin(self), FakeStdout(), io.StringIO("")
        self.requests, self.signals, self.waits = [], [], 0

    def receive(self, request):
        self.requests.append(request)
        if request["operation"] == "baselines":
            names = ("loader-cycle", "field-greedy", "greedy-all", "random")
            self.reply({"ok": True, "baselines": {name: {"meanMargin": 1.0, "margins": [1.0] * len(request["seeds"])}
                                                  for name in names}})
        elif request["operation"] == "close":
            self.reply({"ok": True})
            if self.stops_on == "close":
                self.returncode = 0

    def reply(self, value):
        self.stdout.lines.append(json.dumps(value, separators=(",", ":")) + "\n")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits += 1
        if self.returncode is None:
            if self.on_wait:
                self.on_wait()
            raise subprocess.TimeoutExpired("node", timeout)
        return self.returncode

    def terminate(self):
        self.signals.append("terminate")
        if self.stops_on == "terminate":
            self.returncode = -15

    def kill(self):
        self.signals.append("kill")
        if self.stops_on in ("terminate", "kill"):
            self.returncode = -9


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


def fake_episode(environment, model, seed, alliance, device, deterministic=False):
    return [{"features": [], "choice": 0, "reward": 1.0, "durationSeconds": 1.0}], \
        {"seed": seed, "alliance": alliance, "score": {"red": 1, "blue": 0, "margin": 1},
         "clock": 105, "decisions": 1, "episodeReturn": 1.0, "done": True}


@contextlib.contextmanager
def fake_fit(output, process, run_episode=fake_episode):
    # No model, update or checkpoint work runs: only the trainer's bookkeeping and shutdown path.
    fake_torch = types.SimpleNamespace(device=lambda name: name, manual_seed=lambda seed: None,
                                       cuda=types.SimpleNamespace(manual_seed_all=lambda seed: None),
                                       optim=types.SimpleNamespace(Adam=lambda parameters, lr: types.SimpleNamespace(
                                           state_dict=dict)))
    argv = ["train", "--approved-free-t4", "--seed", "12345", "--episodes", "2", "--max-seconds", "60",
            "--evaluation-episodes", "2", "--output", str(output)]
    with mock.patch.object(train.sys, "argv", argv), \
            mock.patch.object(train, "validate_runtime", return_value={"device": "cuda:0", "gpu": "fake"}), \
            mock.patch.object(train, "torch", fake_torch), \
            mock.patch.object(train, "TacticalActorCritic", FakeModel), \
            mock.patch.object(train, "run_episode", side_effect=run_episode), \
            mock.patch.object(train, "update", return_value={"loss": 0.0, "samples": 1}), \
            mock.patch.object(train, "atomic_torch_save"), \
            mock.patch.object(train.shutil, "which", return_value="node"), \
            mock.patch.object(train.subprocess, "Popen", return_value=process):
        yield


class TacticalBridgeShutdownTests(unittest.TestCase):
    def test_a_bridge_that_exits_after_close_is_a_clean_exit(self):
        process = FakeBridgeProcess(stops_on="close")
        with bridge(process) as environment:
            shutdown = environment.close()
        self.assertEqual(shutdown, {"method": "exited", "closeAcknowledged": True, "timeoutSeconds": 10, "returnCode": 0})
        self.assertEqual(process.requests, [{"operation": "close"}])
        self.assertEqual(process.signals, [])
        self.assertTrue(process.stdin.closed and process.stdout.closed)

    def test_a_bridge_that_stays_alive_after_close_is_stopped_without_raising(self):
        cases = {"eof": ("exited", 0, []), "terminate": ("terminated", -15, ["terminate"]),
                 "kill": ("killed", -9, ["terminate", "kill"]), "never": ("kill-unconfirmed", None, ["terminate", "kill"])}
        for stops_on, (method, returncode, signals) in cases.items():
            process = FakeBridgeProcess(stops_on=stops_on)
            with self.subTest(stops_on), bridge(process) as environment:
                shutdown = environment.close(timeout=0.5)
                self.assertEqual((shutdown["method"], shutdown["returnCode"], process.signals), (method, returncode, signals))
                self.assertTrue(process.stdin.closed)
                self.assertIs(shutdown["closeAcknowledged"], returncode is not None)

    def test_a_dead_or_unreachable_bridge_is_recorded(self):
        process = FakeBridgeProcess(returncode=1)
        with bridge(process) as environment:
            self.assertEqual(environment.close()["method"], "already-exited")
        self.assertEqual(process.requests, [])
        process = FakeBridgeProcess(stops_on="eof", broken_pipe=True)
        with bridge(process) as environment:
            shutdown = environment.close()
        self.assertEqual((shutdown["method"], shutdown["closeAcknowledged"]), ("exited", False))
        self.assertIn("BrokenPipeError", shutdown["closeRequestError"])

    @unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
    def test_the_real_bridge_exits_cleanly_after_close(self):
        environment = train.NodeTacticalEnvironment()
        try:
            self.assertFalse(environment.reset(17, "red")["done"])
        finally:
            shutdown = environment.close()
        self.assertEqual((shutdown["method"], shutdown["returnCode"], shutdown["closeAcknowledged"]), ("exited", 0, True))

    def test_a_finished_fit_keeps_its_final_record_when_the_bridge_hangs(self):
        # The observed T4 failure: the bridge answers close but does not exit.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            on_disk = []
            process = FakeBridgeProcess(stops_on="terminate",
                                        on_wait=lambda: on_disk.append(json.loads((output / "run.json").read_text("utf-8"))))
            printed = io.StringIO()
            with fake_fit(output, process), contextlib.redirect_stdout(printed):
                train.main()
            run = json.loads((output / "run.json").read_text("utf-8"))
            self.assertTrue((output / "evaluation.json").exists())
        self.assertEqual(on_disk[0]["status"], "completed", "final status must be written before the shutdown wait")
        self.assertEqual(on_disk[0]["episodesCompleted"], 2)
        self.assertEqual((run["status"], run["episodesCompleted"], run["evaluationEpisodesCompleted"]), ("completed", 2, 2))
        self.assertEqual((run["bridgeShutdown"]["method"], run["bridgeShutdown"]["closeAcknowledged"]), ("terminated", True))
        self.assertIs(run["learnedImprovementVerified"], False)
        self.assertIs(run["independentQualificationPerformed"], False)
        finished = json.loads(printed.getvalue().splitlines()[-1])
        self.assertEqual((finished["event"], finished["status"], finished["bridgeShutdown"]), ("finished", "completed", "terminated"))

    def test_a_failed_fit_records_interrupted_and_still_shuts_the_bridge_down(self):
        calls = []

        def failing_episode(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("engine failure")
            return fake_episode(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            process = FakeBridgeProcess(stops_on="close")
            with fake_fit(output, process, failing_episode), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, "engine failure"):
                train.main()
            run = json.loads((output / "run.json").read_text("utf-8"))
        self.assertEqual((run["status"], run["episodesCompleted"], run["bridgeShutdown"]["method"]), ("interrupted", 1, "exited"))
        self.assertEqual(process.requests[-1], {"operation": "close"})


if __name__ == "__main__":
    unittest.main()
