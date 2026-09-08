import numpy as np
import pytest

from moe_feeder.deflect_spare import POLICIES, deflect_step, deflect_step_causal
from moe_feeder.reconfig import robust_collapse


def _case(seed=0, n=3000, k=6, D=8):
    rng = np.random.default_rng(seed)
    indices = np.stack([rng.choice(64, size=k, replace=False) for _ in range(n)]).astype(np.int64)
    scores = np.sort(rng.random((n, k)), axis=1)[:, ::-1]
    place = np.repeat(np.arange(D), 8)
    src = (np.arange(n) // int(np.ceil(n / D))).astype(np.int64)
    dest = place[indices[:, :k - 1]]
    M = np.zeros((D, D))
    np.add.at(M, (np.repeat(src, k - 1), dest.ravel()), 1.0)
    E = M * rng.uniform(0.9, 1.05, size=M.shape) + 1.0
    return indices, scores, place, src, E


@pytest.mark.parametrize("policy", POLICIES)
def test_deflect_conserves_and_fits(policy):
    indices, scores, place, src, E = _case()
    r = deflect_step(indices, scores, place, src, E, policy=policy)
    s = r.stats
    assert np.all(r.sent <= np.floor(E + 1e-9) + 1e-9)          # never over the reservation
    assert abs(s["resolved_spare_frac"] + s["resolved_any_frac"] + s["dropped_frac_of_over"] - 1.0) < 1e-9 or s["n_over"] == 0
    assert s["cost_pref_per_token"] <= s["cost_pess_per_token"] + 1e-12
    assert s["cost_pess_per_token"] <= s["cost_drop_per_token"] + 1e-12
    if policy == "drop":
        assert s["dropped_frac_of_over"] == pytest.approx(1.0) or s["n_over"] == 0
    if policy in ("any", "spare_any"):
        assert s["resolved_any_frac"] + s["resolved_spare_frac"] >= 0.0


def test_causal_matches_batch_on_resolution():
    rng = np.random.default_rng(1)
    indices, scores, place, src, E = _case(1)
    M = E.copy()
    E = np.floor(M * rng.uniform(0.985, 1.03, size=M.shape)) + 1.0
    for policy in ("spare", "spare_any"):
        b = deflect_step(indices, scores, place, src, E, policy=policy, order="arrival").stats
        c = deflect_step_causal(indices, scores, place, src, E, policy=policy).stats
        assert c["frac_slots_over"] >= b["frac_slots_over"] - 1e-12
        assert c["frac_slots_over"] <= 2.0 * b["frac_slots_over"] + 1e-12
        assert abs(b["resolved_spare_frac"] - c["resolved_spare_frac"]) < 0.25
        assert abs(b["cost_pref_per_token"] - c["cost_pref_per_token"]) < 0.5 * max(b["cost_drop_per_token"], 1e-9)


def test_no_scores_path():
    indices, scores, place, src, E = _case(2)
    r = deflect_step(indices, None, place, src, E, policy="spare_any", order="rank")
    assert r.stats["cost_pref_per_token"] == 0.0
    assert 0.0 <= r.stats["resolved_spare_frac"] <= 1.0


def test_robust_collapse_excludes_spike():
    rng = np.random.default_rng(0)
    base = [rng.uniform(900, 1100, size=(8, 8)) for _ in range(3)]
    spike = base[0].copy()
    spike[:, 3] *= 6.0
    E_all, n0 = robust_collapse(base, "p95")
    E_sp, n1 = robust_collapse(base + [spike], "p95")
    assert n0 == 0 and n1 == 1
    assert np.allclose(E_sp, E_all)


def test_multi_spare_resolves_at_least_single():
    indices, scores, place, src, E = _case(3)
    one = deflect_step(indices, scores, place, src, E, policy="spare", n_spares=1).stats
    two = deflect_step(indices, scores, place, src, E, policy="spare", n_spares=2).stats
    assert two["n_spares"] == 2 and one["n_spares"] == 1
    assert 0.0 <= two["resolved_spare_frac"] <= 1.0
