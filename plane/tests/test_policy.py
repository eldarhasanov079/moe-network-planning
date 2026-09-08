import numpy as np

from moe_feeder.policy import collapse_history, score_heldout, slots_to_bytes


def test_collapse_mean_and_worst():
    a = np.array([[0, 1], [2, 0]], dtype=float)
    b = np.array([[0, 3], [4, 0]], dtype=float)
    assert np.allclose(collapse_history([a, b], "mean"), [[0, 2], [3, 0]])
    assert np.allclose(collapse_history([a, b], "worst"), [[0, 3], [4, 0]])


def test_envelope_metrics_and_bytes():
    M = np.array([[0, 10], [5, 0]], dtype=float)
    E = np.array([[0, 8], [8, 0]], dtype=float)
    m = score_heldout(M, E)
    assert m["overflow_ratio"] > 0
    assert m["coverage_ratio"] < 1
    assert slots_to_bytes(M, 2)[0, 1] == 20


def test_p95_and_volume():
    a = np.array([[0, 1], [1, 0]], dtype=float)
    b = np.array([[0, 10], [2, 0]], dtype=float)
    p95 = collapse_history([a, b], "p95")
    assert p95[0, 1] >= 1
    vol = collapse_history([a, b], "volume")
    assert vol.shape == (2, 2)


def test_uniform_zero_diagonal():
    hist = [np.ones((4, 4)), np.ones((4, 4))]
    U = collapse_history(hist, "uniform")
    assert np.all(np.diag(U) == 0)
    assert U.sum() > 0
