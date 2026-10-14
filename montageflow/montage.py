"""The 16-channel bipolar (TCP) montage and the linear constraints it imposes.

Every channel is a difference of two electrode potentials, so summing channels along
a chain of electrodes telescopes to the difference between its endpoints. The montage
has two pairs of chains with shared endpoints (Fp1 -> O1 on the left, Fp2 -> O2 on the
right), which gives two exact equations A x = 0 at every time point.
"""
from __future__ import annotations

import math

import numpy as np
import torch

CHANNELS = [
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]
CHANNEL_NAMES = [f"{a}-{b}" for a, b in CHANNELS]

# value of the relative residual for independent Gaussian channels: sqrt(2m/pi), m = 8
INDEPENDENT_RESIDUAL = math.sqrt(2 * 8 / math.pi)

# (chain_a, chain_b): both chains connect the same two electrodes
CONSTRAINTS = [
    ([0, 1, 2, 3], [8, 9, 10, 11]),      # Fp1 -> O1
    ([4, 5, 6, 7], [12, 13, 14, 15]),    # Fp2 -> O2
]


def constraint_matrix() -> np.ndarray:
    """A in R^{2x16}: +1 on one chain, -1 on the other."""
    A = np.zeros((len(CONSTRAINTS), len(CHANNELS)))
    for i, (a, b) in enumerate(CONSTRAINTS):
        A[i, a] = 1.0
        A[i, b] = -1.0
    return A


def projector() -> np.ndarray:
    """Orthogonal projector onto ker A. Here A A^T = 8 I, so P = I - A^T A / 8."""
    A = constraint_matrix()
    return np.eye(A.shape[1]) - A.T @ np.linalg.solve(A @ A.T, A)


def project(x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Project the channel axis (dim 1) of x, shape (B, C, ...), onto ker A."""
    return torch.einsum("ij,bj...->bi...", P.to(x.dtype), x)


def montage_index(x: np.ndarray) -> float:
    """Mean relative residual of the two constraints, scaled so independent channels score 1.

    For each window and constraint: mean_t |sum(chain_a) - sum(chain_b)| divided by the RMS of
    the 8 channels involved. Averaged over windows and constraints, then divided by
    sqrt(2m/pi) (m = 8), the value for independent Gaussian channels. Real recordings score 0.
    """
    return float(montage_residuals(x).mean() / INDEPENDENT_RESIDUAL)


def montage_residuals(x: np.ndarray) -> np.ndarray:
    """Per-window relative residual (averaged over the two constraints), shape (N,)."""
    x = np.asarray(x, dtype=np.float64)
    rel = []
    for a, b in CONSTRAINTS:
        resid = x[:, a].sum(1) - x[:, b].sum(1)
        rms = np.sqrt((x[:, a + b] ** 2).mean(axis=(1, 2))) + 1e-12
        rel.append(np.abs(resid).mean(1) / rms)
    return np.mean(rel, axis=0)
