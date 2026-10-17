import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import call, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.motion_learning.episode_replay import (
    STRATEGY, EpisodeBalancedReplayBuffer, tag_complete_episodes,
)
from roboproof.motion_learning.td3 import (
    ReplayBuffer, ReservoirReplayBuffer, complete_episode_returns,
)


REPLAY_FIELDS = ("observations", "next_observations", "actions", "rewards", "terminals",
                 "required_paces", "horizons")


def episode_rows(lengths, start=0):
    rows = []
    for length in lengths:
        for offset in range(length):
            index = start + len(rows)
            rows.append({"modelInput": [float(index)] + [index / 1000] * 33,
                         "nextModelInput": [float(index + 1)] + [(index + 1) / 1000] * 33,
                         "action": [(index % 9 - 4) / 8, 0.1, -0.2, 0.05],
                         "reward": (index % 17 - 8) / 4,
                         "requiredPace": 0.5 + (index % 5) / 10,
                         "terminal": offset == length - 1})
    return rows


def packed_rows(lengths=(8, 24, 32), start=0):
    return complete_episode_returns(tag_complete_episodes(episode_rows(lengths, start)), 32)


class ReplayAssertions(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state().clone())

    def assert_state_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
            else:
                self.assertEqual(actual[name], value)

    def assert_rejected_atomically(self, replay, operation):
        before = replay.state()
        rng = torch.get_rng_state().clone()
        with self.assertRaises(ValueError):
            operation()
        self.assert_state_equal(replay.state(), before)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def assert_batch_indices(self, replay, batch, indices):
        self.assertEqual(set(batch), set(REPLAY_FIELDS))
        for name in REPLAY_FIELDS:
            torch.testing.assert_close(batch[name], getattr(replay, name)[indices], rtol=0, atol=0)


class EpisodeTaggingTests(ReplayAssertions):
    def test_tagging_clones_rows_and_splits_only_on_original_boolean_terminals(self):
        lengths = (3, 1, 1, 5)
        rows = episode_rows(lengths)
        untouched = copy.deepcopy(rows)
        rng = torch.get_rng_state().clone()
        tagged = tag_complete_episodes(rows)
        self.assertIsNot(tagged, rows)
        self.assertEqual(rows, untouched)
        expected = [length for length in lengths for _ in range(length)]
        self.assertEqual([row["episodeLength"] for row in tagged], expected)
        for original, cloned, length in zip(rows, tagged, expected):
            self.assertIsNot(cloned, original)
            self.assertIs(type(cloned["episodeLength"]), int)
            self.assertEqual(cloned, {**original, "episodeLength": length})
        tagged[0]["reward"] = 999
        tagged[0]["episodeLength"] = 1
        self.assertEqual(rows, untouched)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_length_bounds_apply_per_episode_not_to_the_whole_collection(self):
        lengths = (1, 202, 1, 202)
        tagged = tag_complete_episodes(episode_rows(lengths))
        self.assertEqual([row["episodeLength"] for row in tagged],
                         [length for length in lengths for _ in range(length)])
        for lengths in ((203,), (1, 203), (202, 203)):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                tag_complete_episodes(episode_rows(lengths))

    def test_empty_incomplete_invalid_terminal_and_non_raw_inputs_are_rejected(self):
        invalid = [[], episode_rows((4,))[:-1], episode_rows((1, 4))[:-1],
                   tuple(episode_rows((2,))), iter(episode_rows((2,))),
                   tag_complete_episodes(episode_rows((2,))), [None], [{}]]
        for terminal in (0, 1, None, "false", "true", 0.0, float("nan")):
            for position in (0, 2, 3):
                rows = episode_rows((3, 1))
                rows[position]["terminal"] = terminal
                invalid.append(rows)
        missing = episode_rows((3,))
        del missing[1]["terminal"]
        invalid.append(missing)
        rng = torch.get_rng_state().clone()
        for index, rows in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                tag_complete_episodes(rows)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_tag_before_thirty_two_step_returns_preserves_lengths_through_terminal_tails(self):
        lengths = (8, 24, 32, 40, 202)
        rows = episode_rows(lengths)
        untouched = copy.deepcopy(rows)
        tagged = tag_complete_episodes(rows)
        tagged_before = copy.deepcopy(tagged)
        packed = complete_episode_returns(tagged, 32)
        position = 0
        for length in lengths:
            for offset in range(length):
                result = packed[position + offset]
                horizon = min(32, length - offset)
                self.assertEqual(result["episodeLength"], length)
                self.assertEqual(result["horizon"], horizon)
                self.assertEqual(result["terminal"], length - offset <= 32)
                self.assertEqual(result["nextModelInput"], rows[position + offset + horizon - 1]["nextModelInput"])
                self.assertEqual(result["modelInput"], rows[position + offset]["modelInput"])
            position += length
        self.assertEqual(sum(row["terminal"] for row in rows), len(lengths))
        self.assertEqual(sum(row["terminal"] for row in packed), sum(min(32, length) for length in lengths))
        self.assertEqual(rows, untouched)
        self.assertEqual(tagged, tagged_before)


class EpisodeBalancedReplayTests(ReplayAssertions):
    def test_subclass_default_capacity_and_versioned_empty_metadata(self):
        rng = torch.get_rng_state().clone()
        replay = EpisodeBalancedReplayBuffer(34)
        self.assertIsInstance(replay, ReservoirReplayBuffer)
        self.assertEqual(STRATEGY, "episode-balanced-reservoir-v1")
        self.assertEqual((replay.capacity, replay.count, replay.cursor, replay.seen_rows), (1024, 0, 0, 0))
        self.assertEqual(replay.episode_lengths.shape, (1024,))
        self.assertEqual(replay.episode_lengths.dtype, torch.int64)
        self.assertEqual(int(torch.count_nonzero(replay.episode_lengths)), 0)
        saved = replay.state()
        original = ReservoirReplayBuffer(34).state()
        self.assertEqual(set(saved), set(original) | {"episodeLengths"})
        self.assertEqual((saved["schemaVersion"], saved["strategy"], saved["seenRows"]), (4, STRATEGY, 0))
        torch.testing.assert_close(saved["episodeLengths"], replay.episode_lengths, rtol=0, atol=0)
        self.assertEqual(original["schemaVersion"], 3)
        self.assertEqual(original["strategy"], "uniform-reservoir-v1")
        self.assertEqual(ReplayBuffer(34).state()["schemaVersion"], 2)
        for invalid in (True, 63, 2049, 64.0, "64", None):
            with self.subTest(capacity=invalid), self.assertRaises(ValueError):
                EpisodeBalancedReplayBuffer(34, invalid)
        self.assertEqual(EpisodeBalancedReplayBuffer(34, 2048).capacity, 2048)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_full_pool_equal_episode_mass_for_eight_twenty_four_and_thirty_two_rows(self):
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows())
        self.assertTrue(bool(replay.terminals.all()))
        self.assertEqual(replay.episode_lengths.tolist(), [8] * 8 + [24] * 24 + [32] * 32)
        before, rng = replay.state(), torch.get_rng_state().clone()
        probabilities = replay.sampling_probabilities()
        expected = torch.tensor([1 / (3 * length) for length in (8, 24, 32) for _ in range(length)],
                                dtype=torch.float64)
        torch.testing.assert_close(probabilities, expected, rtol=0, atol=1e-15)
        self.assertAlmostEqual(float(probabilities.sum()), 1, places=14)
        for start, stop in ((0, 8), (8, 32), (32, 64)):
            self.assertAlmostEqual(float(probabilities[start:stop].sum()), 1 / 3, places=14)
        self.assertNotAlmostEqual(float(probabilities[0]), 1 / replay.count)
        self.assert_state_equal(replay.state(), before)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_probabilities_normalize_only_occupied_slots_and_ignore_terminal_flags(self):
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows((2, 5)))
        expected = torch.tensor([1 / 4] * 2 + [1 / 10] * 5, dtype=torch.float64)
        self.assertEqual(replay.episode_lengths[7:].tolist(), [0] * 57)
        for terminal in (False, True):
            replay.terminals[:replay.count].fill_(terminal)
            actual = replay.sampling_probabilities(0)
            self.assertEqual(actual.shape, (7,))
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)

    def test_controlled_multinomial_indices_return_only_normal_fields_without_objective_correction(self):
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows())
        indices = torch.tensor([63, 0, 8, 7, 31, 32, 63, 8])
        expected = replay.sampling_probabilities()
        for fraction in (0, 0.0):
            with self.subTest(terminal_fraction=fraction):
                with patch("torch.multinomial", return_value=indices) as draw, patch("torch.randint") as uniform:
                    batch = replay.sample(len(indices), fraction)
                draw.assert_called_once()
                uniform.assert_not_called()
                arguments, keywords = draw.call_args
                probabilities = arguments[0] if arguments else keywords["input"]
                batch_size = arguments[1] if len(arguments) > 1 else keywords["num_samples"]
                replacement = arguments[2] if len(arguments) > 2 else keywords.get("replacement", False)
                torch.testing.assert_close(probabilities, expected, rtol=0, atol=0)
                self.assertEqual(batch_size, len(indices))
                self.assertIs(replacement, True)
                self.assert_batch_indices(replay, batch, indices)

    def test_sample_matches_exact_multinomial_rng_for_default_and_explicit_zero(self):
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows())
        before = replay.state()
        torch.manual_seed(108001)
        rng = torch.get_rng_state().clone()
        indices = torch.multinomial(replay.sampling_probabilities(), 32, replacement=True)
        expected_rng = torch.get_rng_state().clone()
        for arguments in ((), (0,), (0.0,)):
            torch.set_rng_state(rng)
            batch = replay.sample(32, *arguments)
            self.assert_batch_indices(replay, batch, indices)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
            self.assert_state_equal(replay.state(), before)

    def test_reservoir_membership_all_fields_and_rng_match_original_above_capacity(self):
        rows = packed_rows((8, 24, 32, 1, 202, 17, 7, 45))
        untouched = copy.deepcopy(rows)
        for seed in (108002, 108003):
            torch.manual_seed(seed)
            original = ReservoirReplayBuffer(34, 64)
            balanced = EpisodeBalancedReplayBuffer(34, 64)
            start = 0
            for stop in (1, 64, 65, 122, 243, len(rows)):
                rng = torch.get_rng_state().clone()
                original.append(rows[start:stop])
                expected_rng = torch.get_rng_state().clone()
                torch.set_rng_state(rng)
                balanced.append(rows[start:stop])
                with self.subTest(seed=seed, accepted=stop):
                    self.assertEqual((balanced.count, balanced.cursor, balanced.seen_rows),
                                     (original.count, original.cursor, original.seen_rows))
                    for name in REPLAY_FIELDS:
                        torch.testing.assert_close(getattr(balanced, name), getattr(original, name), rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                    expected_lengths = torch.tensor(
                        [rows[int(index)]["episodeLength"] for index in balanced.observations[:balanced.count, 0]],
                        dtype=torch.int64)
                    torch.testing.assert_close(balanced.episode_lengths[:balanced.count], expected_lengths, rtol=0, atol=0)
                    self.assertEqual(int(torch.count_nonzero(balanced.episode_lengths[balanced.count:])), 0)
                start = stop
        self.assertEqual(rows, untouched)

    def test_controlled_eviction_updates_metadata_only_in_the_selected_retained_slots(self):
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows())
        saved = replay.state()
        expected = copy.deepcopy(saved)
        incoming = packed_rows((1, 2, 3), start=64)
        selections = (5, 64, 0, 65, 63, 6)
        with patch("torch.randint", side_effect=[torch.tensor([slot]) for slot in selections]) as draw:
            replay.append(incoming)
        self.assertEqual(draw.call_args_list, [call(seen, (1,)) for seen in range(65, 71)])
        row_fields = ("modelInput", "nextModelInput", "action", "reward", "terminal", "requiredPace", "horizon")
        for row, slot in zip(incoming, selections):
            if slot < replay.capacity:
                for name, source in zip(REPLAY_FIELDS, row_fields):
                    expected[name][slot] = torch.as_tensor(row[source], dtype=expected[name].dtype)
                expected["episodeLengths"][slot] = row["episodeLength"]
        expected.update(count=64, cursor=6, seenRows=70)
        self.assert_state_equal(replay.state(), expected)
        self.assertEqual(saved["episodeLengths"].tolist(), [8] * 8 + [24] * 24 + [32] * 32)

    def test_checkpoint_continuation_preserves_evicted_metadata_membership_and_sample_rng(self):
        rows = packed_rows((8, 24, 32, 1, 202, 17, 7, 45))
        torch.manual_seed(108004)
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(rows[:130])
        checkpoint, rng = replay.state(), torch.get_rng_state().clone()
        replay.append(rows[130:])
        expected_state, retained_rng = replay.state(), torch.get_rng_state().clone()
        expected_sample = replay.sample(32)
        sampled_rng = torch.get_rng_state().clone()
        restored = EpisodeBalancedReplayBuffer(34, 64)
        before_restore = torch.get_rng_state().clone()
        restored.restore(checkpoint)
        torch.testing.assert_close(torch.get_rng_state(), before_restore, rtol=0, atol=0)
        self.assert_state_equal(restored.state(), checkpoint)
        torch.set_rng_state(rng)
        restored.append(rows[130:])
        self.assert_state_equal(restored.state(), expected_state)
        torch.testing.assert_close(torch.get_rng_state(), retained_rng, rtol=0, atol=0)
        self.assert_state_equal(restored.sample(32), expected_sample)
        torch.testing.assert_close(torch.get_rng_state(), sampled_rng, rtol=0, atol=0)

    def test_valid_empty_partial_and_full_checkpoints_are_cloned_and_restore_without_rng_changes(self):
        for lengths in ((), (2, 5), (1, 202)):
            with self.subTest(lengths=lengths):
                replay = EpisodeBalancedReplayBuffer(34, 64)
                if lengths:
                    replay.append(packed_rows(lengths))
                saved = replay.state()
                restored = EpisodeBalancedReplayBuffer(34, 64)
                rng = torch.get_rng_state().clone()
                restored.restore(saved)
                self.assert_state_equal(restored.state(), saved)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                saved["episodeLengths"].fill_(999)
                saved["observations"].fill_(999)
                self.assert_state_equal(restored.state(), replay.state())

    def test_append_rejects_empty_non_list_and_exhausted_iterators_atomically(self):
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows())
        incoming = packed_rows((2,), start=64)
        exhausted = iter(incoming)
        list(exhausted)
        invalid = ([], tuple(incoming), iter(incoming), (row for row in incoming), exhausted, None, incoming[0])
        with patch("torch.randint") as draw:
            for index, rows in enumerate(invalid):
                with self.subTest(case=index):
                    self.assert_rejected_atomically(replay, lambda: replay.append(rows))
        draw.assert_not_called()

    def test_missing_invalid_lengths_and_horizon_greater_than_length_reject_entire_append(self):
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows())
        valid = packed_rows((2,), start=64)[0]
        missing = dict(valid)
        del missing["episodeLength"]
        invalid = [missing, {**valid, "horizon": 3}]
        invalid.extend({**valid, "episodeLength": length}
                       for length in (True, False, 0, -1, 203, 2.0, "2", None, float("nan"), torch.tensor(2)))
        with patch("torch.randint") as draw:
            for index, row in enumerate(invalid):
                with self.subTest(case=index):
                    self.assert_rejected_atomically(replay, lambda: replay.append([valid, row]))
        draw.assert_not_called()

    def test_invalid_base_transition_fields_reject_entire_append_before_reservoir_draw(self):
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows())
        valid = packed_rows((32,), start=64)[0]
        changes = ({"modelInput": [0.0] * 33}, {"modelInput": [float("nan")] * 34},
                   {"nextModelInput": [float("inf")] * 34}, {"nextModelInput": [0.0] * 35},
                   {"action": [2.0, 0, 0, 0]}, {"action": [0.0] * 3},
                   {"reward": True}, {"reward": float("nan")}, {"reward": 1000001},
                   {"requiredPace": 0.24}, {"requiredPace": 1.01}, {"requiredPace": True},
                   {"terminal": 1}, {"horizon": 0}, {"horizon": 33}, {"horizon": True})
        with patch("torch.randint") as draw:
            for change in changes:
                with self.subTest(change=change):
                    self.assert_rejected_atomically(replay, lambda: replay.append([valid, {**valid, **change}]))
        draw.assert_not_called()

    def test_accepted_row_limit_rejects_append_without_state_or_rng_mutation(self):
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows())
        saved = replay.state()
        saved.update(seenRows=50000, cursor=50000 % 64)
        replay.restore(saved)
        with patch("torch.randint") as draw:
            self.assert_rejected_atomically(replay, lambda: replay.append(packed_rows((1,), start=64)))
        draw.assert_not_called()

    def test_empty_sampling_batch_bounds_and_nonzero_terminal_mixtures_reject_atomically(self):
        empty = EpisodeBalancedReplayBuffer(34, 64)
        self.assert_rejected_atomically(empty, empty.sampling_probabilities)
        self.assert_rejected_atomically(empty, lambda: empty.sample(1))
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows((2, 5)))
        with patch("torch.multinomial") as draw:
            for batch_size in (0, -1, 8, True, False, 1.0, "1", None):
                with self.subTest(batch_size=batch_size):
                    self.assert_rejected_atomically(replay, lambda: replay.sample(batch_size))
            for fraction in (0.001, 0.25, 0.5, 1, -0.1, True, False,
                             float("nan"), float("inf"), -float("inf"), "0", None):
                with self.subTest(terminal_fraction=fraction):
                    self.assert_rejected_atomically(replay, lambda: replay.sampling_probabilities(fraction))
                    self.assert_rejected_atomically(replay, lambda: replay.sample(1, fraction))
        draw.assert_not_called()
        self.assertEqual(replay.sample(1)["observations"].shape, (1, 34))
        self.assertEqual(replay.sample(replay.count)["observations"].shape, (7, 34))

    def test_restore_rejects_invalid_schema_strategy_counts_cursor_and_missing_keys_atomically(self):
        source = EpisodeBalancedReplayBuffer(34, 64)
        source.append(packed_rows((2, 5)))
        saved = source.state()
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows((8, 24, 32, 7)))
        changes = [{"schemaVersion": 3}, {"schemaVersion": 5}, {"strategy": "uniform-reservoir-v1"},
                   {"strategy": "recent-ring-v1"}, {"capacity": 128}, {"featureSize": 35}]
        changes.extend({"seenRows": value} for value in (True, -1, 50001, 7.0, "7", None, 6, 8))
        changes.extend({"count": value} for value in (True, -1, 65, 7.0, "7", None, 6, 8))
        changes.extend({"cursor": value} for value in (True, -1, 64, 7.0, "7", None, 6))
        for change in changes:
            with self.subTest(change=change):
                self.assert_rejected_atomically(replay, lambda: replay.restore({**saved, **change}))
        for name in saved:
            malformed = {key: value for key, value in saved.items() if key != name}
            with self.subTest(missing=name):
                self.assert_rejected_atomically(replay, lambda: replay.restore(malformed))
        for malformed in (None, [], {}):
            with self.subTest(state=malformed):
                self.assert_rejected_atomically(replay, lambda: replay.restore(malformed))
        full = replay.state()
        for change in ({"seenRows": 70}, {"cursor": 0}, {"count": 63}):
            with self.subTest(full_change=change):
                self.assert_rejected_atomically(replay, lambda: replay.restore({**full, **change}))

    def test_restore_rejects_metadata_dtype_shape_active_bounds_and_nonzero_unused_slots_atomically(self):
        source = EpisodeBalancedReplayBuffer(34, 64)
        source.append(packed_rows((2, 5)))
        saved = source.state()
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows())
        lengths = saved["episodeLengths"]
        invalid = [None, lengths.tolist(), lengths.float(), lengths.to(torch.int32), lengths.bool(),
                   lengths[:-1], lengths.reshape(64, 1)]
        for position, value in ((0, 0), (0, -1), (0, 203), (6, 0), (7, 1), (63, -1)):
            malformed = lengths.clone()
            malformed[position] = value
            invalid.append(malformed)
        for index, metadata in enumerate(invalid):
            with self.subTest(case=index):
                self.assert_rejected_atomically(replay, lambda: replay.restore({**saved, "episodeLengths": metadata}))

    def test_restore_retains_all_base_tensor_validation_and_is_atomic(self):
        source = EpisodeBalancedReplayBuffer(34, 64)
        source.append(packed_rows((2, 5)))
        saved = source.state()
        replay = EpisodeBalancedReplayBuffer(34, 64)
        replay.append(packed_rows())
        for name in REPLAY_FIELDS:
            value = saved[name]
            invalid = [None, value.tolist(), value[:-1], value.to(torch.float64)]
            if value.is_floating_point():
                for nonfinite in (float("nan"), float("inf")):
                    malformed = value.clone()
                    malformed.reshape(-1)[0] = nonfinite
                    invalid.append(malformed)
            for index, malformed in enumerate(invalid):
                with self.subTest(field=name, case=index):
                    self.assert_rejected_atomically(replay, lambda: replay.restore({**saved, name: malformed}))
        for name, value in (("actions", 2), ("required_paces", 0.24), ("required_paces", 1.01),
                            ("horizons", 0), ("horizons", 33)):
            malformed = copy.deepcopy(saved)
            malformed[name].reshape(-1)[0] = value
            with self.subTest(field=name, value=value):
                self.assert_rejected_atomically(replay, lambda: replay.restore(malformed))

    def test_strategies_cannot_restore_each_others_checkpoints_and_original_sampling_is_unchanged(self):
        balanced = EpisodeBalancedReplayBuffer(34, 64)
        balanced.append(packed_rows())
        for original in (ReplayBuffer(34, 64), ReservoirReplayBuffer(34, 64)):
            original.append(packed_rows())
            self.assert_rejected_atomically(balanced, lambda: balanced.restore(original.state()))
            self.assert_rejected_atomically(original, lambda: original.restore(balanced.state()))
            torch.manual_seed(108005)
            indices = torch.randint(original.count, (32,))
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(108005)
            self.assert_batch_indices(original, original.sample(32), indices)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
