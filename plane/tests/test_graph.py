import numpy as np

from moe_feeder.graph import graph_from_matrix
from puppeteer.ir import NodeKind


def test_graph_from_matrix_skips_diagonal_and_zeros():
    M = np.array([
        [10, 5, 0],
        [1, 0, 2],
        [0, 3, 7],
    ])
    graph = graph_from_matrix(M)
    transfers = [n for n in graph.nodes.values() if n.kind is NodeKind.TRANSFER]
    pairs = {(t.src_rank, t.dst_rank, t.size_bytes) for t in transfers}
    assert pairs == {(0, 1, 5), (1, 0, 1), (1, 2, 2), (2, 1, 3)}
    assert graph.num_ranks == 3
    assert all(t.origin.matrix_entry == (t.src_rank, t.dst_rank) for t in transfers)


def test_plan_matrix_runs_puppeteer():
    from moe_feeder.config import DEFAULT_TOPOLOGY
    from moe_feeder.planner import plan_matrix

    rng = np.random.default_rng(0)
    M = rng.integers(0, 20, size=(8, 8))
    np.fill_diagonal(M, 0)
    result = plan_matrix(M, str(DEFAULT_TOPOLOGY), bytes_per_slot=64)
    assert result["n_flows"] > 0
    assert result["iteration_time_s"] is not None
    assert result["flow_bytes"] > 0


def test_envelope_sizes_reuse_mean_routes():
    from moe_feeder.config import DEFAULT_TOPOLOGY
    from moe_feeder.planner import plan_matrix

    rng = np.random.default_rng(1)
    mean = rng.integers(1, 8, size=(8, 8)).astype(float)
    np.fill_diagonal(mean, 0)
    envelope = mean * 2.0
    mean_run = plan_matrix(mean, str(DEFAULT_TOPOLOGY), bytes_per_slot=64, envelope_slot="mean")
    env_run = plan_matrix(
        envelope, str(DEFAULT_TOPOLOGY), bytes_per_slot=64,
        envelope_slot="p95", route_slots=mean,
    )
    mean_paths = {(f.src_rank, f.dst_rank): tuple(f.path) for f in mean_run["plan"].flows}
    env_paths = {(f.src_rank, f.dst_rank): tuple(f.path) for f in env_run["plan"].flows}
    shared = set(mean_paths) & set(env_paths)
    assert shared
    for pair in shared:
        assert mean_paths[pair] == env_paths[pair]
    assert env_run["flow_bytes"] > mean_run["flow_bytes"]
