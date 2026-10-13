import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_learning import td3_train, train
from roboproof.motion_learning.checkpoint_limits import (
    CHECKPOINT_METADATA_BYTES, CHECKPOINT_SERIALIZATION_BYTES, checkpoint_size_limit, checkpoint_tensor_bytes,
)
from roboproof.motion_learning.imitation import ANCHOR_CAPACITY, MeasuredAnchor
from roboproof.motion_learning.model import DEADLINE_CONTEXT, DEADLINE_FEATURES, FEATURE_TRANSFORMS, MotionActorCritic, export, normalization_definition
from roboproof.motion_learning.td3 import ALGORITHM, ReplayBuffer, ReservoirReplayBuffer, TD3Learner


FEATURE_SIZES = {"identity-v1": 34, "reference-frame-v1": 35, "history-context-v1": 144,
                 "deadline-context-v1": 145, "long-history-deadline-v1": 565,
                 "settling-normalized-deadline-v1": 145, "coarse-augmentation-deadline-v1": 169,
                 "multiscale-deadline-v1": 169}


def td3_config(width=128, feature_transform="long-history-deadline-v1", capacity=2048, strategy="recent-ring-v1"):
    return td3_train.configuration(190010, 1, 2, 50000, 16, 1, feature_transform,
        "finite-total-return-settle-margin-v4", "mixed-full-reach-v5", 2, capacity, 0.05, 1,
        "public-deadline-residual-v1", 4, replay_strategy=strategy, network_width=width)


def ppo_config(width=128, feature_transform="long-history-deadline-v1"):
    return {"seed": 190010, "seedCount": 1, "updates": 2, "maxSteps": 50000, "rolloutSteps": 16,
            "algorithm": "ppo-beta-reference-v1", "initialization": "broad-pace-v1", "networkWidth": width,
            "featureTransform": feature_transform, "normalizationDefinition": normalization_definition(feature_transform),
            "deadlineContextDefinition": dict(DEADLINE_CONTEXT) if feature_transform in DEADLINE_FEATURES else None,
            "warmStartUpdates": 1, "demonstrationAnchorWeight": 1}


def full_replay(feature_size, capacity, strategy="recent-ring-v1"):
    replay_type = ReservoirReplayBuffer if strategy == "uniform-reservoir-v1" else ReplayBuffer
    replay = replay_type(feature_size, capacity)
    saved = replay.state()
    saved.update(count=capacity, cursor=0)
    saved["observations"].fill_(0.125)
    saved["next_observations"].fill_(0.25)
    saved["rewards"].fill_(0.1)
    saved["required_paces"].fill_(0.75)
    saved["horizons"].fill_(1)
    saved["terminals"][-1] = True
    if strategy == "uniform-reservoir-v1":
        saved["seenRows"] = capacity
    replay.restore(saved)
    return replay


def full_anchor(feature_size):
    anchor = MeasuredAnchor(feature_size)
    anchor.restore({"schemaVersion": 1, "features": torch.full((ANCHOR_CAPACITY, feature_size), 0.125),
                    "targets": torch.full((ANCHOR_CAPACITY, 4), 0.25)})
    return anchor


def initialized_td3(config):
    actor = MotionActorCritic("broad-pace-v1", config["featureTransform"], False, config["networkWidth"])
    learner = TD3Learner(actor)
    replay = full_replay(len(actor.scales), config["replayCapacity"], config["replayStrategy"])
    for _ in range(2):
        learner.update(replay.sample(8))
    return {"model": actor.state_dict(), "learner": learner.state(), "replay": replay.state(),
            "measuredAnchor": full_anchor(len(actor.scales)).state()}


def initialized_ppo(config):
    model = MotionActorCritic("broad-pace-v1", config["featureTransform"], True, config["networkWidth"])
    optimizer = torch.optim.Adam([
        {"params": [*model.body.parameters(), *model.actor.parameters()], "lr": 0.001},
        {"params": [*model.critic_body.parameters(), *model.critic.parameters()], "lr": 0.0003}])
    distribution, values = model(torch.full((8, len(model.scales)), 0.125), encoded=True)
    loss = distribution.mean.square().mean() + values.square().mean()
    loss.backward()
    optimizer.step()
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "measuredAnchor": full_anchor(len(model.scales)).state()}


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise AssertionError("Checkpoint fixture must remain on CPU")
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(entry) for entry in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(entry) for entry in value)
    return 0


class MotionCheckpointCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)

    def assert_state_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.device.type, "cpu")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for name in expected:
                self.assert_state_equal(actual[name], expected[name])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for actual_entry, expected_entry in zip(actual, expected):
                self.assert_state_equal(actual_entry, expected_entry)
        else:
            self.assertEqual(actual, expected)

    def checkpoint_call(self, trainer, directory, config, sources, resume=False):
        environment = SimpleNamespace(contract={"identity": {"checkpointCapacityFixture": True}})
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(trainer, "MotionEnvironment", side_effect=AssertionError("No robot worlds permitted")), \
                patch.object(trainer, "training_world", side_effect=AssertionError("No robot worlds permitted")), \
                patch.object(torch.cuda, "is_available", side_effect=AssertionError("CPU only")):
            return trainer.train_seed(environment, directory, config["seed"], config, sources, -math.inf, resume=resume)

    def test_analytical_tensor_counts_and_real_serialization_cover_every_supported_shape(self):
        self.assertEqual(set(FEATURE_SIZES), set(FEATURE_TRANSFORMS))
        maxima = {}
        for width in (32, 64, 128, 256):
            for profile in FEATURE_TRANSFORMS:
                config = td3_config(width, profile, 64)
                state = initialized_td3(config)
                self.assertEqual(len(state["model"]["scales"]), FEATURE_SIZES[profile])
                self.assertEqual(len(state["learner"]["actorOptimizer"]["state"]), 6)
                self.assertEqual(len(state["learner"]["criticOptimizer"]["state"]), 12)
                for capacity in (64, 1024, 2048):
                    for strategy in ("recent-ring-v1", "uniform-reservoir-v1"):
                        with self.subTest(algorithm="td3", width=width, profile=profile, capacity=capacity, strategy=strategy):
                            config = td3_config(width, profile, capacity, strategy)
                            state["replay"] = full_replay(FEATURE_SIZES[profile], capacity, strategy).state()
                            self.assertEqual(tensor_bytes(state), checkpoint_tensor_bytes(config))
                            checkpoint = {**state, "config": config, "worldRng": random.Random(190010).getstate(),
                                          "torchRng": torch.get_rng_state(), "history": []}
                            stream = io.BytesIO()
                            torch.save(checkpoint, stream)
                            measured = stream.tell()
                            self.assertLessEqual(measured, checkpoint_size_limit(config))
                            stream.seek(0)
                            saved = torch.load(stream, map_location="cpu", weights_only=True)
                            self.assert_state_equal(saved, checkpoint)
                            maxima[("td3", width)] = max(maxima.get(("td3", width), 0), measured)
                with self.subTest(algorithm="ppo", width=width, profile=profile):
                    config = ppo_config(width, profile)
                    state = initialized_ppo(config)
                    self.assertEqual(len(state["optimizer"]["state"]), 12)
                    self.assertEqual(tensor_bytes(state), checkpoint_tensor_bytes(config))
                    stream = io.BytesIO()
                    torch.save({**state, "config": config, "history": [], "torchRng": torch.get_rng_state()}, stream)
                    self.assertLessEqual(stream.tell(), checkpoint_size_limit(config))
                    maxima[("ppo", width)] = max(maxima.get(("ppo", width), 0), stream.tell())
        for (algorithm, width), measured in sorted(maxima.items()):
            print(json.dumps({"checkpointCapacity": algorithm, "width": width, "maximumMeasuredBytes": measured}))

    def test_default_dimensions_anchor_accounting_and_budget_extension_do_not_change_limits(self):
        config = {"algorithm": "ppo-beta-reference-v1"}
        self.assertEqual(checkpoint_size_limit(config), checkpoint_size_limit({**config, "networkWidth": 32, "featureTransform": "identity-v1"}))
        for current in (ppo_config(), td3_config()):
            baseline = {**current, "demonstrationAnchorWeight": 0}
            self.assertEqual(checkpoint_tensor_bytes(current) - checkpoint_tensor_bytes(baseline), ANCHOR_CAPACITY * (565 + 4) * 4)
            self.assertEqual(checkpoint_size_limit(current), checkpoint_size_limit({**current, "updates": 100, "maxSteps": 50000}))
            self.assertEqual(checkpoint_size_limit(current) - checkpoint_tensor_bytes(current), CHECKPOINT_METADATA_BYTES + CHECKPOINT_SERIALIZATION_BYTES)
            self.assertEqual(checkpoint_size_limit(current), checkpoint_size_limit({**current, "checkpointMaxBytes": 2 ** 60}))

    def test_invalid_admission_dimensions_fail_without_allocations_or_rng_draws(self):
        config = td3_config()
        invalid = [{**config, "networkWidth": value} for value in (True, 32.0, 512, "128")]
        invalid += [{**config, "replayCapacity": value} for value in (True, 63, 2049, 2048.0)]
        invalid += [{**config, "demonstrationAnchorWeight": value} for value in (True, -1, 101, math.nan, math.inf)]
        invalid += [{**config, "featureTransform": "unknown"}, {**config, "algorithm": "unknown"}]
        saved_rng = torch.get_rng_state().clone()
        with patch("torch.tensor", side_effect=AssertionError("No tensor allocation")), \
                patch("torch.nn.Linear", side_effect=AssertionError("No network allocation")):
            for current in invalid:
                with self.subTest(config=current), self.assertRaises(ValueError):
                    checkpoint_size_limit(current)
        self.assert_state_equal(torch.get_rng_state(), saved_rng)

    def test_largest_states_with_bounded_history_use_real_writer_and_resume_without_worlds(self):
        for trainer, config, build_state in ((td3_train, td3_config(width=256), initialized_td3),
                                            (td3_train, td3_config(width=256, strategy="uniform-reservoir-v1"), initialized_td3),
                                            (train, ppo_config(width=256), initialized_ppo)):
            with self.subTest(algorithm=config["algorithm"], strategy=config.get("replayStrategy")), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                sources = trainer.source_identity()
                initial = self.checkpoint_call(trainer, directory, config, sources)
                self.assertFalse(initial["trained"])
                checkpoint_path = directory / "checkpoint.pt"
                checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
                checkpoint.update(build_state(config))
                checkpoint["torchRng"] = torch.get_rng_state()
                checkpoint["history"] = [{"checkpointCapacityFixture": "h" * (CHECKPOINT_METADATA_BYTES - CHECKPOINT_SERIALIZATION_BYTES)}]
                torch.save(checkpoint, checkpoint_path)
                self.assertGreater(checkpoint_path.stat().st_size, 4 * 1024 * 1024)
                self.assertLessEqual(checkpoint_path.stat().st_size, checkpoint_size_limit(config))
                with patch.object(trainer.torch, "load", wraps=torch.load) as loader:
                    resumed = self.checkpoint_call(trainer, directory, config, sources, resume=True)
                    self.assertEqual(loader.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
                self.assertFalse(resumed["trained"])
                self.assertEqual((resumed["steps"], resumed["episodes"], resumed["updates"]), (0, 0, 0))
                saved = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
                self.assert_state_equal(saved, checkpoint)
                print(json.dumps({"checkpointCapacity": config["algorithm"], "strategy": config.get("replayStrategy"),
                                  "largestHistoryMeasuredBytes": checkpoint_path.stat().st_size,
                                  "analyticalLimitBytes": checkpoint_size_limit(config)}))

    def test_reader_rejects_missing_and_one_byte_oversized_files_before_deserialization(self):
        for trainer, config in ((train, ppo_config()), (td3_train, td3_config())):
            with self.subTest(algorithm=config["algorithm"]), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                with patch.object(trainer.torch, "load", side_effect=AssertionError("Premature deserialization")):
                    with self.assertRaisesRegex(ValueError, "Bounded.*checkpoint"):
                        self.checkpoint_call(trainer, directory, config, {}, resume=True)
                    with (directory / "checkpoint.pt").open("wb") as checkpoint:
                        checkpoint.seek(checkpoint_size_limit(config))
                        checkpoint.write(b"\0")
                    with self.assertRaisesRegex(ValueError, "Bounded.*checkpoint"):
                        self.checkpoint_call(trainer, directory, config, {}, resume=True)

    def test_writer_uses_same_limit_and_never_replaces_committed_checkpoint_when_oversized(self):
        for trainer, config in ((train, ppo_config()), (td3_train, td3_config())):
            with self.subTest(algorithm=config["algorithm"]), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                sources = trainer.source_identity()
                self.checkpoint_call(trainer, directory, config, sources)
                checkpoint_path = directory / "checkpoint.pt"
                before = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
                def oversized_save(value, filename):
                    with Path(filename).open("wb") as checkpoint:
                        checkpoint.seek(checkpoint_size_limit(config))
                        checkpoint.write(b"\0")
                with patch.object(trainer.torch, "save", side_effect=oversized_save):
                    with self.assertRaisesRegex(ValueError, "exceeds bounded reader size"):
                        self.checkpoint_call(trainer, directory, config, sources, resume=True)
                self.assertEqual(hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), before)

    def test_source_runtime_identity_config_and_scale_gates_survive_larger_admission(self):
        for trainer, config in ((train, ppo_config(feature_transform="settling-normalized-deadline-v1")),
                                (td3_train, td3_config(feature_transform="settling-normalized-deadline-v1"))):
            with self.subTest(algorithm=config["algorithm"]), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                sources = trainer.source_identity()
                self.assertIn("roboproof/motion_learning/checkpoint_limits.py", sources)
                self.checkpoint_call(trainer, directory, config, sources)
                checkpoint_path = directory / "checkpoint.pt"
                checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
                changes = ({"sources": {}}, {"identity": {}}, {"seed": config["seed"] + 1},
                           {"torchVersion": "different"}, {"schemaVersion": 2},
                           {"config": {**config, "networkWidth": 64}},
                           {"config": {**config, "featureTransform": "deadline-context-v1"}},
                           {"config": {**config, "normalizationDefinition": None}})
                if trainer is td3_train:
                    changes += ({"execution": {}}, {"algorithm": "ppo-beta-reference-v1"},
                                {"config": {**config, "replayCapacity": 64}})
                for change in changes:
                    torch.save({**checkpoint, **change}, checkpoint_path)
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        self.checkpoint_call(trainer, directory, config, sources, resume=True)
                corrupted = copy.deepcopy(checkpoint)
                corrupted["model"]["scales"][3] *= 2
                torch.save(corrupted, checkpoint_path)
                with self.assertRaisesRegex(ValueError, "normalization mismatch|scale mismatch"):
                    self.checkpoint_call(trainer, directory, config, sources, resume=True)
                torch.save(checkpoint, checkpoint_path)
                with self.assertRaisesRegex(ValueError, "resume required|explicit resume"):
                    self.checkpoint_call(trainer, directory, config, sources)
                bad_config = {**config, "normalizationDefinition": None}
                with patch.object(trainer.torch, "load", side_effect=AssertionError("Invalid config must fail before load")):
                    with self.assertRaises(ValueError):
                        self.checkpoint_call(trainer, directory, bad_config, sources, resume=True)

    def test_json_artifact_budget_remains_four_mib(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            with self.assertRaisesRegex(ValueError, "artifact exceeds bounded reader size"):
                train.atomic_json(path, {"value": "h" * CHECKPOINT_METADATA_BYTES})
            self.assertFalse(path.exists())

    def test_large_actor_publication_has_separate_eight_mib_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy-0000.json"
            actor = MotionActorCritic("broad-pace-v1", "long-history-deadline-v1", False, 256)
            document = export(actor, {}, 190010, 0, 0, ALGORITHM)
            with self.assertRaisesRegex(ValueError, "artifact exceeds bounded reader size"):
                train.atomic_json(path, document)
            train.atomic_json(path, document, actor=True)
            self.assertGreater(path.stat().st_size, CHECKPOINT_METADATA_BYTES)
            self.assertLess(path.stat().st_size, 2 * CHECKPOINT_METADATA_BYTES)
            self.assertEqual(json.loads(path.read_text()), document)
            committed = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "actor contract required"):
                train.atomic_json(path, {"value": "h" * CHECKPOINT_METADATA_BYTES}, actor=True)
            with self.assertRaisesRegex(ValueError, "artifact exceeds bounded reader size"):
                train.atomic_json(path, {**document, "extra": "h" * (2 * CHECKPOINT_METADATA_BYTES)}, actor=True)
            self.assertEqual(path.read_bytes(), committed)


if __name__ == "__main__":
    unittest.main()
