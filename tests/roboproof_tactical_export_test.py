import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from roboproof.tactical_learning import export_policy
from roboproof.tactical_learning.model import FEATURE_COUNT, MODEL_ID, TacticalActorCritic


ROOT = Path(__file__).resolve().parents[1]


def checkpoint_bytes(network, **overrides):
    # Freshly initialized weights only; nothing is trained here.
    value = {"schemaVersion": 1, "runId": "run-1", "model": MODEL_ID, "modelState": network.state_dict(),
             "episodesCompleted": 3, "trainingSeed": 12345, "sourceSha256": {"train.py": "ab"}, "runtime": {"gpu": "fake"},
             "warmStart": {"policy": "loader-cycle", "demonstrationEpisodesCompleted": 2, "pretraining": {"final": {"accuracy": 0.5}}},
             "learnedImprovementVerified": False, "independentQualificationPerformed": False, **overrides}
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def tiny_model(width=16):
    torch.manual_seed(7)
    return TacticalActorCritic(width=width)


class TacticalPolicyExportTests(unittest.TestCase):
    def test_the_actor_is_exported_with_provenance_and_no_claims(self):
        model = tiny_model()
        raw = checkpoint_bytes(model)
        policy = export_policy.build_policy(raw)
        self.assertEqual((policy["schema"], policy["model"], policy["featureCount"], policy["width"]),
                         (export_policy.POLICY_SCHEMA, MODEL_ID, FEATURE_COUNT, 16))
        self.assertEqual(len(policy["layers"]["hidden1"]["weight"]), 16)
        self.assertEqual(len(policy["layers"]["hidden1"]["weight"][0]), FEATURE_COUNT)
        self.assertEqual(len(policy["layers"]["hidden2"]["weight"]), 16)
        self.assertEqual(len(policy["layers"]["policy"]["weight"]), 16)
        self.assertEqual(policy["layers"]["policy"]["bias"], float(model.policy_head.bias.detach()[0]))
        self.assertEqual(policy["provenance"]["checkpointSha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual((policy["provenance"]["runId"], policy["provenance"]["trainingSeed"]), ("run-1", 12345))
        self.assertEqual(policy["provenance"]["warmStart"], {"policy": "loader-cycle", "demonstrationEpisodes": 2, "imitationAccuracy": 0.5})
        self.assertIn("not PPO", policy["algorithm"])
        self.assertIn("warm start from loader-cycle", policy["algorithm"])
        self.assertIs(policy["learnedImprovementVerified"], False)
        self.assertIs(policy["independentQualificationPerformed"], False)
        self.assertNotIn("value_head", json.dumps(policy))

    def test_the_label_follows_the_training_algorithm_in_the_checkpoint(self):
        model = tiny_model()
        plain = export_policy.build_policy(checkpoint_bytes(model))
        self.assertEqual(plain["algorithmId"], "actor-critic")
        ppo = export_policy.build_policy(checkpoint_bytes(model, algorithm="ppo-clip-v1"))
        self.assertEqual(ppo["algorithmId"], "ppo-clip-v1")
        self.assertTrue(ppo["algorithm"].startswith("PPO: clipped surrogate objective"))
        self.assertNotIn("not PPO", ppo["algorithm"])
        with self.assertRaisesRegex(RuntimeError, "Unknown training algorithm"):
            export_policy.build_policy(checkpoint_bytes(model, algorithm="trpo"))

    def test_wrong_checkpoints_are_refused(self):
        model = tiny_model()
        with self.assertRaisesRegex(RuntimeError, "schema-1"):
            export_policy.build_policy(checkpoint_bytes(model, model="other-model"))
        with self.assertRaisesRegex(RuntimeError, "schema-1"):
            export_policy.build_policy(checkpoint_bytes(model, schemaVersion=2))
        state = model.state_dict()
        state["encoder.2.bias"] = torch.full_like(state["encoder.2.bias"], float("nan"))
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            export_policy.build_policy(checkpoint_bytes(model, modelState=state))
        state = model.state_dict()
        state["policy_head.weight"] = torch.zeros((2, 16))
        with self.assertRaisesRegex(RuntimeError, "policy_head.weight"):
            export_policy.build_policy(checkpoint_bytes(model, modelState=state))

    def test_the_cli_writes_once_and_checks_the_evaluation_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "latest.pt").write_bytes(checkpoint_bytes(tiny_model()))
            (run / "evaluation.json").write_text(json.dumps({"runId": "run-1", "episodes": [{}, {}], "meanScoreMargin": 191.5,
                                                             "baselineMeanScoreMargin": {"loader-cycle": 244.5},
                                                             "beatsLoaderCycleOnHoldout": False}), encoding="utf-8")
            output = run / "out" / "policy.json"
            with contextlib.redirect_stdout(io.StringIO()):
                export_policy.main(["--checkpoint", str(run / "latest.pt"), "--output", str(output)])
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["evaluation"]["meanScoreMargin"], 191.5)
            self.assertEqual(written["evaluation"]["heldOutEpisodes"], 2)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                export_policy.main(["--checkpoint", str(run / "latest.pt"), "--output", str(output)])
            (run / "evaluation.json").write_text(json.dumps({"runId": "another-run"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different run"):
                export_policy.main(["--checkpoint", str(run / "latest.pt"), "--output", str(output), "--replace"])

    @unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
    def test_the_javascript_actor_matches_the_torch_policy_head(self):
        model = tiny_model(width=32).eval()
        policy = export_policy.build_policy(checkpoint_bytes(model))
        torch.manual_seed(11)
        features = torch.randn((1, 9, FEATURE_COUNT))
        with torch.no_grad():
            logits, _ = model(features, torch.ones((1, 9), dtype=torch.bool))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            script = ("const {loadPolicy} = require('./roboproof/tactical_policy');"
                      "const policy = loadPolicy(process.argv[1]); const rows = JSON.parse(process.argv[2]);"
                      "console.log(JSON.stringify(rows.map(row => policy.score(row))));")
            output = subprocess.run(["node", "-e", script, str(path), json.dumps(features[0].tolist())], cwd=ROOT,
                                    check=True, capture_output=True, text=True).stdout
        scores = json.loads(output)
        for expected, actual in zip(logits[0].tolist(), scores):
            self.assertAlmostEqual(expected, actual, places=5)
        self.assertEqual(int(logits[0].argmax()), max(range(len(scores)), key=scores.__getitem__))


if __name__ == "__main__":
    unittest.main()
