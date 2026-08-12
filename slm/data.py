"""Memory-mapped loader over the packed uint16 token windows.

The whole token set is 6.6 GB against 251 GB of RAM, so after the first pass it
lives in page cache. A memmap plus a shuffled window index is both simpler and
faster than a prefetch pipeline here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import config as C


@dataclass
class WindowStore:
    """All windows of one split, addressable by a single global index."""

    shards: list[np.ndarray]
    sources: list[str]
    offsets: np.ndarray  # cumulative window counts, len == len(shards) + 1
    seq_len: int

    @property
    def n_windows(self) -> int:
        return int(self.offsets[-1])

    @classmethod
    def load(cls, split_dir: Path) -> "WindowStore":
        index = json.loads((split_dir / "index.json").read_text())
        seq_len = index["seq_len"]
        shards, sources, counts = [], [], []
        for entry in index["shards"]:
            arr = np.memmap(split_dir / entry["name"], dtype=np.uint16, mode="r")
            n = arr.size // seq_len
            shards.append(arr[: n * seq_len].reshape(n, seq_len))
            sources.append(entry["source"])
            counts.append(n)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return cls(shards=shards, sources=sources, offsets=offsets, seq_len=seq_len)

    def gather(self, idx: np.ndarray) -> torch.Tensor:
        """Fetch windows by global index -> int64 tensor (B, seq_len)."""
        shard_of = np.searchsorted(self.offsets, idx, side="right") - 1
        out = np.empty((idx.size, self.seq_len), dtype=np.int64)
        for s in np.unique(shard_of):
            mask = shard_of == s
            local = idx[mask] - self.offsets[s]
            out[mask] = self.shards[s][local].astype(np.int64)
        return torch.from_numpy(out)

    def indices_for_sources(self, wanted: tuple[str, ...]) -> np.ndarray:
        """Global window indices belonging to the given sources."""
        parts = [
            np.arange(self.offsets[i], self.offsets[i + 1], dtype=np.int64)
            for i, src in enumerate(self.sources)
            if src in wanted
        ]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def source_names(self) -> list[str]:
        seen = []
        for s in self.sources:
            if s not in seen:
                seen.append(s)
        return seen


class EpochSampler:
    """Deterministic per-epoch permutation, sharded across DDP ranks.

    Reshuffling every epoch (seed = DATA_SEED + epoch) matters for a 5-epoch run:
    an identical order teaches the model sequence position as well as content,
    and produces periodic artefacts in the loss curve.
    """

    def __init__(self, n_windows: int, rank: int, world_size: int,
                 windows_per_step: int):
        self.n_windows = n_windows
        self.rank = rank
        self.world_size = world_size
        self.windows_per_step = windows_per_step
        self.steps_per_epoch = n_windows // windows_per_step

    def epoch_order(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(C.DATA_SEED + epoch)
        order = rng.permutation(self.n_windows)
        usable = self.steps_per_epoch * self.windows_per_step
        return order[:usable]

    def rank_batches(self, epoch: int, start_step: int = 0):
        """Yield (step, micro_batch_indices_for_this_rank) for one epoch."""
        order = self.epoch_order(epoch)
        per_step = self.windows_per_step
        per_rank = per_step // self.world_size
        for step in range(start_step, self.steps_per_epoch):
            chunk = order[step * per_step : (step + 1) * per_step]
            mine = chunk[self.rank * per_rank : (self.rank + 1) * per_rank]
            yield step, mine
