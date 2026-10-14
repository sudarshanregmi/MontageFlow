"""Per-window physiological features and the distance used to compare their distributions.

Features (fs = 200 Hz):
  psd_slope        exponent chi of a least-squares 1/f^chi fit to each channel's Welch
                   spectrum over 1-40 Hz (2 s Hann windows), averaged over channels
  envelope         log10 RMS over time of e(t) = sqrt(mean_c x_c(t)^2)
  hjorth_activity  mean over channels of log10 var(x_c)
  hjorth_mobility  mean over channels of sqrt(var(dx_c) / var(x_c)), dx the first difference
  drift            least-squares slope of e(t) across the window, times L, over mean e(t)

A feature is scored by the 1-Wasserstein distance between its generated and real
distributions, divided by the interquartile range of the real values.
"""
from __future__ import annotations

import numpy as np

FEATURES = ("psd_slope", "envelope", "hjorth_activity", "hjorth_mobility", "drift")
EPS = 1e-12


def welch(x: np.ndarray, fs: float = 200.0, nperseg: int = 400):
    L = x.shape[-1]
    nperseg = min(nperseg, L)
    starts = list(range(0, L - nperseg + 1, max(1, nperseg // 2))) or [0]
    w = np.hanning(nperseg)
    norm = 1.0 / (fs * (w ** 2).sum())
    psd = 0.0
    for s in starts:
        seg = x[..., s:s + nperseg].astype(np.float64)
        seg = seg - seg.mean(-1, keepdims=True)
        psd = psd + np.abs(np.fft.rfft(seg * w, axis=-1)) ** 2 * norm
    return np.fft.rfftfreq(nperseg, d=1.0 / fs), psd / len(starts)


def envelope(x: np.ndarray) -> np.ndarray:
    return np.sqrt((x.astype(np.float64) ** 2).mean(1) + EPS)


def window_features(x: np.ndarray, fs: float = 200.0) -> np.ndarray:
    """x: (N, C, L) -> (N, 5), columns in FEATURES order."""
    x = np.asarray(x, dtype=np.float64)

    freqs, psd = welch(x, fs)
    band = (freqs >= 1.0) & (freqs <= 40.0)
    lf = np.log10(freqs[band])
    lp = np.log10(psd[..., band] + EPS)
    lf, lp = lf - lf.mean(), lp - lp.mean(-1, keepdims=True)
    psd_slope = -((lp * lf).sum(-1) / (lf ** 2).sum()).mean(1)

    e = envelope(x)
    env = np.log10(np.sqrt((e ** 2).mean(-1)) + EPS)

    d1 = np.diff(x, axis=-1)
    v0, v1 = x.var(-1) + EPS, d1.var(-1) + EPS
    activity = np.log10(v0).mean(1)
    mobility = np.sqrt(v1 / v0).mean(1)

    t = np.arange(e.shape[-1], dtype=np.float64)
    t -= t.mean()
    drift = ((e - e.mean(-1, keepdims=True)) * t).sum(-1) / (t ** 2).sum() * e.shape[-1] / (e.mean(-1) + EPS)

    return np.stack([psd_slope, env, activity, mobility, drift], axis=1)


def w1(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == len(b):
        return float(np.abs(np.sort(a) - np.sort(b)).mean())
    q = np.linspace(0.0, 1.0, max(len(a), len(b), 512))
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def iqr(a: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    lo, hi = np.percentile(a[np.isfinite(a)], [25, 75])
    return float(hi - lo)


def distances(gen_features: np.ndarray, real_features: np.ndarray) -> dict:
    """W1 / IQR(real) for each feature."""
    return {name: w1(gen_features[:, j], real_features[:, j]) / iqr(real_features[:, j])
            for j, name in enumerate(FEATURES)}
