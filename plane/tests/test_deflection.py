import numpy as np

from moe_feeder.config import FeederConfig
from moe_feeder.deflection import apply_deflection
from moe_feeder.runtime import replay_against_plan


def _place_identity(n=8):
    return np.arange(n, dtype=np.int64)


def test_none_does_not_move():
    n = 20
    indices = np.stack([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)], axis=1)
    scores = np.stack([np.full(n, 0.6), np.full(n, 0.4)], axis=1)
    place = _place_identity()
    src = np.zeros(n, dtype=np.int64)
    cfg = FeederConfig(deflection="none", num_devices=8, num_experts=8)
    out = apply_deflection(cfg, indices, scores, place, src)
    assert out.deflected == 0
    assert np.all(out.chosen_device == 0)


def test_online_global_moves_low_margin_off_hot_device():
    n = 10
    indices = np.stack([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)], axis=1)
    scores = np.stack([np.full(n, 0.51), np.full(n, 0.49)], axis=1)
    place = _place_identity()
    src = np.zeros(n, dtype=np.int64)
    cfg = FeederConfig(deflection="online_global", tau=0.05, num_devices=8, num_experts=8)
    out = apply_deflection(cfg, indices, scores, place, src)
    assert out.eligible == n
    assert out.deflected > 0
    assert out.frac_deflected > 0


def test_replay_deflects_only_when_overflow_and_low_margin():
    n = 6
    indices = np.stack([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)], axis=1)
    scores = np.stack([np.full(n, 0.51), np.full(n, 0.49)], axis=1)
    place = _place_identity()
    src = np.zeros(n, dtype=np.int64)
    reservation = np.zeros((8, 8))
    reservation[0, 0] = 2
    reservation[0, 1] = 10
    cfg = FeederConfig(tau=0.05, num_devices=8, num_experts=8)
    stats = replay_against_plan(cfg, indices, scores, place, src, reservation, deflect=True)
    assert stats.would_overflow >= 1
    assert stats.deflected_to_fit >= 1
