import numpy as np
import pytest

from moe_feeder.admit import admit_by_slot, admit_slots, token_drop_stats


def _case(seed=0, n=4000, k=6, D=8, experts=64):
    rng = np.random.default_rng(seed)
    indices = np.stack([rng.choice(experts, size=k, replace=False) for _ in range(n)]).astype(np.int64)
    scores = np.sort(rng.random((n, k)), axis=1)[:, ::-1]
    place = np.repeat(np.arange(D), experts // D)
    per = int(np.ceil(n / D))
    src = (np.arange(n) // per).astype(np.int64)
    dest = place[indices]
    M = np.zeros((D, D))
    np.add.at(M, (np.repeat(src, k), dest.ravel()), 1.0)
    E = M * rng.uniform(0.85, 1.05, size=M.shape) + rng.uniform(0, 0.9, size=M.shape)  # fractional
    return indices, scores, place, src, M, E


def test_arrival_matches_token_drop_stats():
    indices, scores, place, src, M, E = _case()
    ref = token_drop_stats(indices, place, src, E)
    out = admit_by_slot(indices, place, src, E, order="arrival", overflow="drop")
    assert out.stats["frac_tokens_dropped"] == pytest.approx(ref["frac_tokens_dropped"], abs=0)
    assert out.stats["frac_tokens_fully_dropped"] == pytest.approx(ref["frac_tokens_fully_dropped"], abs=0)
    assert out.stats["frac_slots_dropped"] == pytest.approx(ref["frac_slots_dropped"], abs=0)


@pytest.mark.parametrize("order", ["arrival", "score", "rank"])
def test_reserved_counts_are_min_M_floorE(order):
    indices, scores, place, src, M, E = _case(seed=1)
    out = admit_by_slot(indices, place, src, E, scores=scores, order=order, overflow="tail")
    expect = np.minimum(M, np.floor(E + 1e-9))
    assert np.array_equal(out.sent_reserved, expect)
    assert np.array_equal(out.sent_reserved + out.sent_tail + out.sent_dropped, M)
    n, k = indices.shape
    assert np.all(out.reserved.sum(1) + out.tail.sum(1) + out.dropped.sum(1) == k)
    assert out.sent_dropped.sum() == 0


def test_score_order_keeps_heavier_slots():
    indices, scores, place, src, M, E = _case(seed=2)
    out = admit_by_slot(indices, place, src, E, scores=scores, order="score", overflow="tail")
    w = scores / scores.sum(1, keepdims=True)
    dest = place[indices]
    D = E.shape[0]
    cell = src[:, None] * D + dest
    for c in np.unique(cell[out.tail]):
        res_w = w[(cell == c) & out.reserved]
        tail_w = w[(cell == c) & out.tail]
        if len(res_w) and len(tail_w):
            assert res_w.min() >= tail_w.max() - 1e-12
    arrival = admit_by_slot(indices, place, src, E, scores=scores, order="arrival", overflow="tail")
    assert out.stats["weight_mass_tail_per_token"] <= arrival.stats["weight_mass_tail_per_token"]


def test_rank_order_prefers_low_columns():
    indices, scores, place, src, M, E = _case(seed=3)
    out = admit_by_slot(indices, place, src, E, order="rank", overflow="tail")
    hist = np.array(out.stats["rank_hist_tail"])
    assert hist[-1] >= hist[0]


def test_gates_and_errors():
    indices, scores, place, src, M, E = _case(seed=4)
    gated = admit_by_slot(indices, place, src, E, scores=scores, order="arrival", overflow="tail", weight_gate=1.0)
    assert gated.sent_tail.sum() == 0 and gated.sent_dropped.sum() > 0
    cut = admit_by_slot(indices, place, src, E, order="arrival", overflow="tail", rank_cut=0)
    assert cut.sent_tail.sum() == 0
    with pytest.raises(ValueError):
        admit_by_slot(indices, place, src, E, order="score")
    with pytest.raises(ValueError):
        admit_by_slot(indices, place, src, E, order="arrival", weight_gate=0.1)


def test_admit_slots_consistency():
    indices, scores, place, src, M, E = _case(seed=5)
    out = admit_by_slot(indices, place, src, np.floor(E), order="arrival", overflow="tail")
    assert np.array_equal(out.sent_reserved, admit_slots(M, np.floor(E)))
