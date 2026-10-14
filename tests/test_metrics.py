import numpy as np

from montageflow.metrics import Moments, frechet, physiology, shape, ts_fid


def test_ts_fid_streaming_matches_one_shot():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(600, 16, 400)), rng.normal(size=(600, 16, 400)) * 1.5
    streamed_a, streamed_b = Moments(), Moments()
    for i in range(0, 600, 150):
        streamed_a.update(ts_fid.features(a[i:i + 150]))
        streamed_b.update(ts_fid.features(b[i:i + 150]))
    assert np.isclose(frechet(streamed_a, streamed_b), ts_fid.ts_fid(a, b))
    assert ts_fid.ts_fid(a, a) < 1e-6


def test_shape_correlation_of_identical_sets_is_one():
    x = np.random.default_rng(1).normal(size=(50, 16, 300))
    x[:, 1] += x[:, 0]                                  # give the channels some structure
    r = shape.correlation_sum(x) / len(x)
    assert np.isclose(shape.shape_correlation(r, r), 1.0)


def test_physiology_distance_is_zero_for_identical_sets():
    x = np.random.default_rng(2).normal(size=(40, 16, 1000))
    f = physiology.window_features(x)
    assert f.shape == (40, len(physiology.FEATURES))
    assert all(d == 0 for d in physiology.distances(f, f).values())
