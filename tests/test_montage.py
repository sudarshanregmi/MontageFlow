import numpy as np
import torch

from montageflow import montage


def test_projector_is_orthogonal_projection_onto_kernel():
    A, P = montage.constraint_matrix(), montage.projector()
    assert np.allclose(A @ A.T, 8 * np.eye(2))
    assert np.allclose(P, np.eye(16) - A.T @ A / 8)
    assert np.allclose(A @ P, 0)
    assert np.allclose(P @ P, P) and np.allclose(P, P.T)
    assert np.linalg.matrix_rank(P) == 14


def test_bipolar_channels_satisfy_constraints():
    rng = np.random.default_rng(0)
    electrodes = {e for pair in montage.CHANNELS for e in pair}
    v = {e: rng.normal(size=(8, 500)) for e in electrodes}           # potentials, 8 windows
    x = np.stack([v[a] - v[b] for a, b in montage.CHANNELS], axis=1)  # (8, 16, 500)
    assert montage.montage_index(x) < 1e-12


def test_independent_channels_score_about_one():
    x = np.random.default_rng(1).normal(size=(2000, 16, 200))
    assert abs(montage.montage_index(x) - 1.0) < 0.02


def test_project_makes_any_signal_valid():
    P = torch.from_numpy(montage.projector()).float()
    x = torch.randn(4, 16, 10, 100)
    y = montage.project(x, P).reshape(4, 16, -1).numpy()
    assert montage.montage_index(y) < 1e-5
