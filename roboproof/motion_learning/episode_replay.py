import math

import torch

from .td3 import ReservoirReplayBuffer


STRATEGY = "episode-balanced-reservoir-v1"
MAX_EPISODE_ROWS = 202


def definition():
    return {"method": STRATEGY, "retention": "Unchanged uniform accepted-row reservoir and actual row accounting",
            "length": "Actual complete raw episode row count, attached before n-step horizon packing; all outcomes retained",
            "probability": "Normalize1/episodeLength over occupied reservoir slots; draw with replacement",
            "objective": "Both critics, actor Q and its public priors use the same episode-length-balanced row distribution",
            "weight": "No inverse-probability correction: intentionally different training objective, not uniform-row equivalence",
            "limitation": "Equal mass per episode only when every episode row is retained; a finite row reservoir approximates that balance",
            "terminalMixture": "Must be zero; no compounded terminal selection",
            "scope": "Training-only sampling metadata; no actor input, outcome filtering, new query, physics, reward or gate change"}


def tag_complete_episodes(rows):
    if not isinstance(rows, list) or not rows or len(rows) > 50000:
        raise ValueError("Bounded complete raw episode rows required")
    prepared = []
    episode_start = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("terminal"), bool) or "episodeLength" in row:
            raise ValueError("Original boolean terminal rows without inferred episode lengths required")
        length = index - episode_start + 1
        if length > MAX_EPISODE_ROWS:
            raise ValueError("Actual episode length exceeds original deadline budget")
        if row["terminal"]:
            prepared.extend({**item, "episodeLength": length} for item in rows[episode_start:index + 1])
            episode_start = index + 1
    if episode_start != len(rows):
        raise ValueError("Complete intrinsic-terminal episodes required before horizon packing")
    return prepared


class EpisodeBalancedReplayBuffer(ReservoirReplayBuffer):
    def __init__(self, feature_size, capacity=1024):
        super().__init__(feature_size, capacity)
        self.episode_lengths = torch.zeros(capacity, dtype=torch.int64)

    def append(self, rows):
        if not isinstance(rows, list):
            raise ValueError("Bounded complete episode row list required")
        prepared = self._prepare(rows)
        lengths = [row.get("episodeLength") for row in rows]
        if (not prepared or self.seen_rows + len(prepared) > 50000 or
                any(type(length) is not int or not 1 <= length <= MAX_EPISODE_ROWS for length in lengths) or
                any(values[-1] > length for values, length in zip(prepared, lengths))):
            raise ValueError("Bounded actual complete episode lengths and accepted row count required")
        for values, length in zip(prepared, lengths):
            self.seen_rows += 1
            if self.count < self.capacity:
                selected = self.count
                self.count += 1
            else:
                selected = int(torch.randint(self.seen_rows, (1,)))
            self.cursor = self.seen_rows % self.capacity
            if selected >= self.capacity:
                continue
            for name, value in zip(("observations", "next_observations", "actions", "rewards", "required_paces", "terminals", "horizons"), values):
                getattr(self, name)[selected] = value
            self.episode_lengths[selected] = length

    def sampling_probabilities(self, terminal_fraction=0):
        if (isinstance(terminal_fraction, bool) or not isinstance(terminal_fraction, (int, float)) or
                not math.isfinite(terminal_fraction) or terminal_fraction != 0 or not self.count):
            raise ValueError("Occupied episode-balanced replay requires zero terminal mixture")
        lengths = self.episode_lengths[:self.count]
        if torch.any(lengths < 1) or torch.any(lengths > MAX_EPISODE_ROWS):
            raise ValueError("Actual bounded retained episode lengths required")
        probabilities = lengths.to(torch.float64).reciprocal()
        return probabilities / probabilities.sum()

    def sample(self, batch_size, terminal_fraction=0):
        if type(batch_size) is not int or not 1 <= batch_size <= self.count:
            raise ValueError("Bounded occupied episode-balanced replay batch required")
        probabilities = self.sampling_probabilities(terminal_fraction)
        indices = torch.multinomial(probabilities, batch_size, replacement=True)
        return {name: getattr(self, name)[indices] for name in
                ("observations", "next_observations", "actions", "rewards", "terminals", "required_paces", "horizons")}

    def state(self):
        return {**super().state(), "schemaVersion": 4, "strategy": STRATEGY, "episodeLengths": self.episode_lengths.clone()}

    def restore(self, state):
        count = state.get("count") if isinstance(state, dict) else None
        lengths = state.get("episodeLengths") if isinstance(state, dict) else None
        horizons = state.get("horizons") if isinstance(state, dict) else None
        if (not isinstance(state, dict) or state.get("schemaVersion") != 4 or state.get("strategy") != STRATEGY or
                type(count) is not int or not 0 <= count <= self.capacity or not isinstance(lengths, torch.Tensor) or
                lengths.shape != self.episode_lengths.shape or lengths.dtype != torch.int64 or
                not isinstance(horizons, torch.Tensor) or horizons.shape != lengths.shape or horizons.dtype != torch.int64 or
                torch.any(lengths[:count] < 1) or torch.any(lengths[:count] > MAX_EPISODE_ROWS) or torch.any(lengths[count:] != 0) or
                torch.any(lengths[:count] < horizons[:count])):
            raise ValueError("Versioned bounded episode replay lengths required")
        super().restore({**state, "schemaVersion": 3, "strategy": "uniform-reservoir-v1"})
        self.episode_lengths.copy_(lengths)
