import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from roboproof.motion_learning.actor_artifact import atomic_actor_json
from roboproof.motion_learning.model import MotionActorCritic, export
from roboproof.motion_learning.train import atomic_json
from roboproof.motion_learning import td3_train
from roboproof.motion_env import MotionEnvironment


def payload(depth=2):
    return {"schemaVersion": 1, "policyContractVersion": 1, "algorithm": "td3-beta-mean-reference-v1",
            "observationSize": 34, "actionSize": 4, "width": 32,
            **({"hiddenLayers": depth} if depth != 2 else {}), "layers": [0.1, -0.0, 1e-20, 3.5],
            "identity": {"label": "\u0440\u043e\u0431\u043e\u0442"}}


class ActorArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_two_and_ten_layer_actor_bytes_match_original_writer(self):
        for width, depth in ((32, 2), (256, 2), (128, 10)):
            with self.subTest(width=width, depth=depth), tempfile.TemporaryDirectory() as temporary:
                torch.manual_seed(201)
                model = MotionActorCritic("broad-pace-v1", "deadline-context-v1", False, width, depth)
                value = export(model, {"label": "\u0440\u043e\u0431\u043e\u0442"}, 201, 64, 4, "td3-beta-mean-reference-v1")
                old, new = Path(temporary) / "old.json", Path(temporary) / "new.json"
                atomic_json(old, value, actor=True)
                atomic_actor_json(new, value)
                self.assertEqual(old.read_bytes(), new.read_bytes())
                self.assertEqual(json.loads(new.read_text()), value)
                self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_streaming_does_not_require_monolithic_dumps(self):
        with tempfile.TemporaryDirectory() as temporary, patch("json.dumps", side_effect=MemoryError("No full string")):
            filename = Path(temporary) / "actor.json"
            atomic_actor_json(filename, payload())
            self.assertEqual(json.loads(filename.read_text()), payload())

    def test_original_eight_and_twenty_four_mebibyte_bounds_are_exact(self):
        for depth, maximum in ((2, 8 * 1024 * 1024), (10, 24 * 1024 * 1024)):
            for excess in (0, 1):
                with self.subTest(depth=depth, excess=excess), tempfile.TemporaryDirectory() as temporary:
                    filename = Path(temporary) / "actor.json"
                    filename.write_bytes(b"previous actor")
                    prefix = json.dumps(payload(depth), allow_nan=False, separators=(",", ":"))[:-1] + ',"padding":"'
                    padding = maximum + excess - len(prefix.encode()) - len('"}\n')

                    def chunks(value):
                        yield prefix
                        for index in range(padding // 65536):
                            yield "a" * 65536
                        yield "a" * (padding % 65536)
                        yield '"}'

                    with patch("json.JSONEncoder.iterencode", side_effect=chunks):
                        if excess:
                            with self.assertRaisesRegex(ValueError, "bounded reader size"):
                                atomic_actor_json(filename, payload(depth))
                            self.assertEqual(filename.read_bytes(), b"previous actor")
                        else:
                            atomic_actor_json(filename, payload(depth))
                            self.assertEqual(filename.stat().st_size, maximum + len(os.linesep) - 1)
                    self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_encoding_and_replace_failures_preserve_existing_actor_and_foreign_temp(self):
        for failure in ("nonfinite", "memory", "write", "replace"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                filename = Path(temporary) / "actor.json"
                foreign = Path(temporary) / "actor.json.tmp"
                filename.write_bytes(b"previous actor")
                foreign.write_bytes(b"unrelated preexisting temporary")

                def interrupted(value):
                    yield '{"partial":'
                    raise MemoryError("Stream interruption")

                original_temporary = tempfile.NamedTemporaryFile

                def write_failure(**arguments):
                    output = original_temporary(**arguments)

                    def fail(chunk):
                        raise OSError("Temporary write denied")

                    output.write = fail
                    return output

                if failure == "memory":
                    with patch("json.JSONEncoder.iterencode", side_effect=interrupted), self.assertRaises(MemoryError):
                        atomic_actor_json(filename, payload())
                elif failure == "replace":
                    with patch.object(Path, "replace", side_effect=OSError("Publication denied")), self.assertRaises(OSError):
                        atomic_actor_json(filename, payload())
                elif failure == "write":
                    with patch("tempfile.NamedTemporaryFile", side_effect=write_failure), self.assertRaises(OSError):
                        atomic_actor_json(filename, payload())
                else:
                    with self.assertRaises(ValueError):
                        atomic_actor_json(filename, {**payload(), "layers": [float("nan")]})
                self.assertEqual(filename.read_bytes(), b"previous actor")
                self.assertEqual(foreign.read_bytes(), b"unrelated preexisting temporary")
                self.assertEqual(set(Path(temporary).iterdir()), {filename, foreign})

    def test_invalid_contract_is_rejected_before_any_file_creation(self):
        for field, invalid in (("schemaVersion", 2), ("policyContractVersion", 2), ("algorithm", "unknown"),
                               ("observationSize", 145), ("actionSize", 3), ("width", 1024), ("hiddenLayers", 3)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                filename = Path(temporary) / "absent/actor.json"
                with self.assertRaises(ValueError):
                    atomic_actor_json(filename, {**payload(), field: invalid})
                self.assertFalse(filename.parent.exists())

    def test_native_actor_save_failure_rejects_unsafe_resume_without_changing_evidence(self):
        calls = []

        def publish(filename, value):
            calls.append(str(filename))
            if len(calls) == 3:
                raise MemoryError("Native actor publication interrupted")
            atomic_actor_json(filename, value)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arguments = dict(directory=directory, seed=890064, seed_count=1, updates=2, max_steps=512,
                             rollout_steps=16, max_seconds=90, warm_start_updates=0, gradient_steps=2,
                             demonstration_anchor_weight=0, policy_regularization="none", deadline_pace_loss_weight=1)
            with patch.object(td3_train, "atomic_actor_json", side_effect=publish), self.assertRaises(MemoryError):
                td3_train.run(**arguments)
            self.assertEqual(len(calls), 3)
            run_id = json.loads((directory / "latest.json").read_text())["runId"]
            manifest = json.loads((directory / run_id / "run.json").read_text())
            self.assertEqual(manifest["status"], "interrupted-or-failed")
            self.assertEqual(manifest["models"][0]["updates"], 0)
            self.assertFalse((directory / run_id / "seed-890064/policy-0001.json").exists())
            before = {str(filename.relative_to(directory)): filename.read_bytes()
                      for filename in directory.rglob("*") if filename.is_file()}
            with patch.object(MotionEnvironment, "reset") as reset, patch.object(MotionEnvironment, "step") as step:
                with self.assertRaisesRegex(ValueError, "exposure.*checkpoint"):
                    td3_train.run(**arguments, resume=run_id)
                reset.assert_not_called()
                step.assert_not_called()
            self.assertEqual(before, {str(filename.relative_to(directory)): filename.read_bytes()
                                     for filename in directory.rglob("*") if filename.is_file()})


if __name__ == "__main__":
    unittest.main()
