"""TS-FID: Frechet distance between Gaussians fitted to spectral features of real and
generated windows. The features follow JET (https://github.com/Y-Research-SBU/JET):
FFT magnitude along time (lower half of the bins), average-pooled to at most 256 frequency
bins and 4 channel groups, log(1 + .), then standardized within each window.

Statistics are accumulated in chunks (`Moments`) so a whole split never has to be in memory.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


def features(x, freq_bins: int = 256, channel_groups: int = 4) -> np.ndarray:
    """x: (N, C, L) -> (N, D) float64 feature vectors."""
    x = torch.as_tensor(np.asarray(x)).to(torch.float64)
    mag = torch.fft.rfft(x, dim=-1).abs()
    mag = mag[..., :max(1, int(mag.shape[-1] * 0.5))]
    mag = F.adaptive_avg_pool1d(mag, min(freq_bins, mag.shape[-1]))
    if mag.shape[1] > channel_groups:
        mag = F.adaptive_avg_pool1d(mag.permute(0, 2, 1), channel_groups).permute(0, 2, 1)
    f = torch.log1p(mag.reshape(mag.shape[0], -1))
    f = (f - f.mean(dim=1, keepdim=True)) / f.std(dim=1, keepdim=True).clamp_min(1e-6)
    return f.numpy()


class Moments:
    """Running count, sum and sum of outer products of feature vectors."""

    def __init__(self):
        self.n, self.s, self.ss = 0, None, None

    def update(self, f: np.ndarray):
        f = np.asarray(f, dtype=np.float64)
        if self.s is None:
            self.s, self.ss = np.zeros(f.shape[1]), np.zeros((f.shape[1], f.shape[1]))
        self.n += len(f)
        self.s += f.sum(0)
        self.ss += f.T @ f
        return self

    def mean_cov(self):
        mu = self.s / self.n
        return mu, (self.ss - self.n * np.outer(mu, mu)) / (self.n - 1)


def frechet(a: Moments, b: Moments) -> float:
    mu_a, cov_a = a.mean_cov()
    mu_b, cov_b = b.mean_cov()
    covmean, _ = linalg.sqrtm(cov_a.dot(cov_b), disp=False)
    if not np.isfinite(covmean).all():
        eye = np.eye(len(mu_a)) * 1e-6
        covmean, _ = linalg.sqrtm((cov_a + eye).dot(cov_b + eye), disp=False)
    covmean = covmean.real
    diff = mu_a - mu_b
    value = float(diff.dot(diff) + np.trace(cov_a + cov_b - 2.0 * covmean))
    return value if math.isfinite(value) else math.nan


def ts_fid(generated, real) -> float:
    return frechet(Moments().update(features(generated)), Moments().update(features(real)))
