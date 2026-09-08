import sys
import numpy as np

from moe_feeder.config import EXPERIMENTS
from moe_feeder.stage2_blocks import block_matrix, offset_blocks, scale_reservation, step_blocks, step_reservation

sys.path.insert(0, str(EXPERIMENTS))
from common.placement import token_source_ranks  # noqa: E402


def test_full_block_reproduces_stage1_sharding():
    n = 250_000
    blocks = step_blocks(n, n, 8)
    assert len(blocks) == 1
    idx, src = blocks[0]
    assert np.array_equal(idx, np.arange(n))
    assert np.array_equal(src, token_source_ranks(n, 8, "contiguous"))


def test_blocks_partition_each_rank_shard():
    n, B = 205_000, 32_768
    blocks = step_blocks(n, B, 8)
    per = int(np.ceil(n / 8))
    m = B // 8
    assert len(blocks) == per // m
    seen = np.zeros(n, dtype=int)
    for idx, src in blocks:
        assert len(idx) == B and np.all(np.bincount(src, minlength=8) == m)
        assert np.all(idx // per == src)
        seen[idx] += 1
    assert seen.max() == 1


def test_offset_blocks_and_scaling():
    blocks = offset_blocks(1000, 100, 200, 8)
    assert len(blocks) == (1000 - 100) // 200
    assert blocks[0][0][0] == 100
    E = np.full((8, 8), 10.0)
    assert np.allclose(scale_reservation(E, 25, 100), 2.5)


def test_step_reservation_at_full_B_is_stage1():
    rng = np.random.default_rng(0)
    n, k = 8000, 6
    place = np.repeat(np.arange(8), 8)
    window = [rng.integers(0, 64, size=(n, k)) for _ in range(3)]
    src = token_source_ranks(n, 8, "contiguous")
    E_full, ns = step_reservation(window, n, "mean", place, 8)
    assert ns == 3
    ref = np.mean([block_matrix(w, place, src, 8) for w in window], axis=0)
    assert np.allclose(E_full, ref)
