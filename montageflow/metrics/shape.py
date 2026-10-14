"""Shape correlation: do generated windows couple the same channel pairs as real EEG?

The 16x16 Pearson correlation matrix is computed per window and averaged over windows.
The score is the Pearson correlation between the off-diagonal entries of the generated and
the real average matrix.
"""
from __future__ import annotations

import numpy as np


def correlation_sum(x: np.ndarray) -> np.ndarray:
    """Sum over windows of the per-window channel correlation matrix. x: (N, C, L)."""
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean(2, keepdims=True)
    sd = xc.std(2) + 1e-12
    r = np.einsum("ncl,nkl->nck", xc, xc) / x.shape[2] / (sd[:, :, None] * sd[:, None, :])
    return r.sum(0)


def shape_correlation(mean_gen: np.ndarray, mean_real: np.ndarray) -> float:
    off = ~np.eye(len(mean_gen), dtype=bool)
    return float(np.corrcoef(mean_gen[off], mean_real[off])[0, 1])
