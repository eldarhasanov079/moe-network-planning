import numpy as np

from moe_feeder.config import DEFAULT_TOPOLOGY
from moe_feeder.graph import graph_from_matrix
from moe_feeder.planner import plan_matrix
from puppeteer.config import RunConfig
from puppeteer.pipeline import plan_graph


def test_plan_matrix_ecmp_and_least_loaded_run():
    matrix = np.full((8, 8), 16.0)
    np.fill_diagonal(matrix, 0.0)
    planned = plan_matrix(matrix, str(DEFAULT_TOPOLOGY), router_kind="least-loaded")
    hashed = plan_matrix(matrix, str(DEFAULT_TOPOLOGY), router_kind="ecmp", router_seed=4)
    assert planned["iteration_time_s"] > 0
    assert hashed["iteration_time_s"] > 0


def test_plan_matrix_new_baselines_run():
    matrix = np.full((8, 8), 16.0)
    np.fill_diagonal(matrix, 0.0)
    for kind in ("first-fit", "round-robin", "spray", "random-path"):
        planned = plan_matrix(matrix, str(DEFAULT_TOPOLOGY), router_kind=kind, router_seed=2)
        assert planned["iteration_time_s"] > 0
        if kind == "spray":
            assert planned["n_flows"] > 8 * 7


def test_plan_graph_router_kind_on_feeder_graph():
    config = RunConfig.load(str(DEFAULT_TOPOLOGY))
    matrix = np.full((8, 8), 1024, dtype=np.int64)
    np.fill_diagonal(matrix, 0)
    graph = graph_from_matrix(matrix)
    run = plan_graph(graph, config, router_kind="ecmp", router_seed=1)
    assert run.plan.metrics["iteration_time_s"] > 0
