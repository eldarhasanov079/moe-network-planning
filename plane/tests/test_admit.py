import numpy as np

from moe_feeder.admit import admit_slots, token_drop_stats


def test_admit_slots_clips_overflow():
    actual = np.array([[0, 10], [5, 0]], dtype=float)
    reserved = np.array([[0, 8], [8, 0]], dtype=float)
    sent = admit_slots(actual, reserved)
    assert sent[0, 1] == 8
    assert sent[1, 0] == 5


def test_token_drop_stats_counts_overflow():
    indices = np.array([[0], [1]], dtype=np.int64)
    place = np.array([0, 1], dtype=np.int64)
    src = np.array([0, 0], dtype=np.int64)
    reserved = np.array([[0, 1], [0, 0]], dtype=float)
    stats = token_drop_stats(indices, place, src, reserved)
    assert stats["n_tokens"] == 2
    assert stats["frac_tokens_dropped"] > 0
