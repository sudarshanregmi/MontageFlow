"""EEG windows stored by scripts/prepare_data.py.

Each split directory holds X.npy (N, C, L) float32 in microvolts, memory-mapped, and
y.npy (N,) with integer labels. Models see x * scale (0.01, i.e. microvolts / 100).
"""
from __future__ import annotations

import os

import numpy as np
import torch


class Windows:
    def __init__(self, path: str, scale: float = 0.01):
        self.path = path
        self.x = np.load(os.path.join(path, "X.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(path, "y.npy")).astype(np.int64)
        self.scale = scale

    def __len__(self):
        return len(self.y)

    @property
    def shape(self):
        return self.x.shape[1:]

    @property
    def num_classes(self):
        return int(self.y.max()) + 1

    def get(self, idx: np.ndarray):
        """Windows and labels at `idx` as tensors (read in sorted order for fast memmap access)."""
        order = np.argsort(idx)
        x = np.empty((len(idx), *self.shape), dtype=np.float32)
        x[order] = self.x[np.asarray(idx)[order]]
        return torch.from_numpy(x * self.scale), torch.from_numpy(self.y[idx])


class BatchSampler:
    """Random batches drawn with replacement, optionally inversely to class frequency."""

    def __init__(self, labels: np.ndarray, balance_classes: bool = True, seed: int = 0):
        self.n = len(labels)
        self.gen = torch.Generator().manual_seed(seed)
        self.weights = None
        if balance_classes:
            counts = np.bincount(labels).astype(np.float64)
            w = 1.0 / np.maximum(counts, 1.0)[labels]
            self.weights = torch.from_numpy(w / w.sum())

    def __call__(self, batch_size: int) -> np.ndarray:
        if self.weights is not None:
            idx = torch.multinomial(self.weights, batch_size, replacement=True, generator=self.gen)
        else:
            idx = torch.randint(0, self.n, (batch_size,), generator=self.gen)
        return np.sort(idx.numpy())
