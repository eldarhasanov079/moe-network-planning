"""Two byte classes in one Puppeteer event loop on the 8-GPU Clos."""

import numpy as np
import pytest

from moe_feeder.config import DEFAULT_TOPOLOGY
from moe_feeder.graph import graph_from_classes, graph_from_matrix
from moe_feeder.planner import freeze_paths, plan_matrix, plan_two_class
from moe_feeder.policy import slots_to_bytes
from puppeteer.config import RunConfig
from puppeteer.ir import NodeKind, Origin, TransferNode
from puppeteer.planner.router import (
    ClassRouter,
    ClosEcmpRouter,
    ClosLeastLoadedRouter,
    FrozenPathRouter,
    enumerate_clos_routes,
)
from puppeteer.planner.state import NetworkState

TOPO = str(DEFAULT_TOPOLOGY)
BYTES_PER_SLOT = 2048
TAIL_SEED = 3


def _slots(seed: int):
    rng = np.random.default_rng(seed)
    reserved = rng.integers(1, 20, size=(8, 8))
    tail = rng.integers(1, 12, size=(8, 8))
    np.fill_diagonal(reserved, 0)
    np.fill_diagonal(tail, 0)
    return reserved, tail


def _flows_of(plan, cls):
    return [f for f in plan.flows if f.origin.get("traffic_class") == cls]


@pytest.fixture(scope="module")
def slots():
    return _slots(0)


@pytest.fixture(scope="module")
def frozen(slots):
    reserved, _ = slots
    return freeze_paths(reserved, TOPO, bytes_per_slot=BYTES_PER_SLOT)


@pytest.fixture(scope="module")
def ecmp_result(slots, frozen):
    reserved, tail = slots
    return plan_two_class(
        reserved, tail, TOPO,
        frozen_paths=frozen, tail_router_kind="ecmp", tail_seed=TAIL_SEED,
        bytes_per_slot=BYTES_PER_SLOT,
    )


@pytest.fixture(scope="module")
def spray_result(slots, frozen):
    reserved, tail = slots
    return plan_two_class(
        reserved, tail, TOPO,
        frozen_paths=frozen, tail_router_kind="spray", bytes_per_slot=BYTES_PER_SLOT,
    )


# ------------------------------------------------------------------ (a)


def test_reserved_class_is_unchanged_by_the_tail(ecmp_result):
    classes = ecmp_result["classes"]
    assert classes["reserved"]["makespan_s"] == pytest.approx(
        ecmp_result["reserved_alone_makespan_s"], rel=1e-9
    )
    alone = {f.flow_id: f.finish_s for f in ecmp_result["reserved_plan"].flows}
    merged = {f.flow_id: f.finish_s for f in _flows_of(ecmp_result["plan"], "reserved")}
    assert set(alone) == set(merged)
    for uid, finish in alone.items():
        assert merged[uid] == pytest.approx(finish, rel=1e-9, abs=1e-15)


# ------------------------------------------------------------------ (b)


def test_tail_bytes_conserved_and_routed_by_ecmp(slots, frozen, ecmp_result):
    _, tail = slots
    tail_bytes = slots_to_bytes(tail, BYTES_PER_SLOT)
    off_diagonal = int(tail_bytes.sum() - np.trace(tail_bytes))

    tail_flows = _flows_of(ecmp_result["plan"], "tail")
    assert tail_flows
    assert sum(f.size_bytes for f in tail_flows) == off_diagonal
    assert ecmp_result["classes"]["tail"]["bytes"] == off_diagonal
    assert all(f.origin["priority"] == 1 for f in tail_flows)

    config = RunConfig.load(TOPO)
    ecmp = ClosEcmpRouter(config.topology, seed=TAIL_SEED)
    state = NetworkState(config.topology)
    differs = 0
    for f in tail_flows:
        expected = ecmp.route(f.src_rank, f.dst_rank, state).links
        assert f.path == expected
        frozen_path = frozen[(f.src_rank, f.dst_rank)].links
        if expected != frozen_path:
            differs += 1
            assert f.path != frozen_path
    assert differs > 0, "ECMP never disagreed with the frozen paths; test is vacuous"

    for f in _flows_of(ecmp_result["plan"], "reserved"):
        assert f.path == frozen[(f.src_rank, f.dst_rank)].links
        assert f.origin.get("priority", 0) == 0


def test_tail_is_served_from_the_residual(ecmp_result):
    """On every link, at every instant, tail rate <= capacity - reserved rate."""
    plan = ecmp_result["plan"]
    config = RunConfig.load(TOPO)
    per_link = {}
    for f in plan.flows:
        is_tail = f.origin.get("traffic_class") == "tail"
        for link in f.path:
            for start, until, bps in plan.rate_intervals(f):
                per_link.setdefault(link, []).append((start, until, bps, is_tail))

    checked = 0
    for link, intervals in per_link.items():
        if not any(t for *_, t in intervals) or all(t for *_, t in intervals):
            continue
        cap = config.topology.links[link].capacity_bps
        for t in sorted({start for start, *_ in intervals}):
            reserved = sum(b for s, u, b, tail in intervals if not tail and s <= t < u)
            tail = sum(b for s, u, b, tail in intervals if tail and s <= t < u)
            checked += 1
            assert tail <= cap - reserved + 1e-6 * cap, (link, t, reserved, tail, cap)
    assert checked > 0
    assert ecmp_result["tail_extension_s"] >= 0.0
    assert ecmp_result["total_makespan_s"] >= ecmp_result["reserved_alone_makespan_s"]


# ------------------------------------------------------------------ (c)


def test_merged_plan_is_zero_queue(ecmp_result, spray_result):
    for result in (ecmp_result, spray_result):
        checks = result["metrics"]["verification"]
        assert checks["zero_queue"] == "ok"
        assert checks["liveness"] == "ok"


# ------------------------------------------------------------------ (d)


def test_spray_tail_splits_inter_pod_pairs_across_all_paths(spray_result):
    config = RunConfig.load(TOPO)
    n_paths = len(enumerate_clos_routes(config.topology, 0, 7))
    assert n_paths == 8
    tail_07 = [
        f for f in _flows_of(spray_result["plan"], "tail")
        if tuple(f.origin["matrix_entry"]) == (0, 7)
    ]
    assert len(tail_07) == n_paths
    assert len({tuple(f.path) for f in tail_07}) == n_paths
    reserved_07 = [
        f for f in _flows_of(spray_result["plan"], "reserved")
        if tuple(f.origin["matrix_entry"]) == (0, 7)
    ]
    assert len(reserved_07) == 1
    assert spray_result["classes"]["reserved"]["makespan_s"] == pytest.approx(
        spray_result["reserved_alone_makespan_s"], rel=1e-9
    )


# ------------------------------------------------------------------ (e)


def test_freeze_paths_covers_every_off_host_pair_and_feeds_plan_matrix(slots, frozen):
    reserved, _ = slots
    config = RunConfig.load(TOPO)
    for src in range(8):
        for dst in range(8):
            if src == dst or config.topology.host_of(src) == config.topology.host_of(dst):
                continue
            assert (src, dst) in frozen
            assert frozen[(src, dst)].links
    result = plan_matrix(reserved, TOPO, bytes_per_slot=BYTES_PER_SLOT, frozen_paths=frozen)
    assert result["iteration_time_s"] > 0
    for f in result["plan"].flows:
        assert f.path == frozen[(f.src_rank, f.dst_rank)].links


def test_plan_two_class_rejects_a_rate_floor(slots, frozen):
    reserved, tail = slots
    config = RunConfig.load(TOPO)
    config.planner.min_rate_bps = 1e9
    with pytest.raises(ValueError, match="min_rate"):
        plan_two_class(reserved, tail, config, frozen_paths=frozen)


# ------------------------------------------------------- graph and router


def test_graph_from_classes_single_class_matches_graph_from_matrix(slots):
    reserved, _ = slots
    M = slots_to_bytes(reserved, BYTES_PER_SLOT)
    a = graph_from_matrix(M)
    b = graph_from_classes(
        [{"name": "reserved", "priority": 0, "M_bytes": M, "spray": False}]
    )

    def flows(graph):
        return {
            (n.src_rank, n.dst_rank, n.size_bytes, n.origin.matrix_entry, n.origin.chunk)
            for n in graph.nodes.values()
            if n.kind is NodeKind.TRANSFER
        }

    assert flows(a) == flows(b)
    assert len(a.nodes) == len(b.nodes)
    assert sum(len(v) for v in a.children.values()) == sum(
        len(v) for v in b.children.values()
    )
    for n in b.nodes.values():
        if n.kind is NodeKind.TRANSFER:
            assert n.uid.startswith("xfer:reserved:")
            assert n.origin.traffic_class == "reserved"
            assert n.origin.priority == 0
    for n in a.nodes.values():
        if n.kind is NodeKind.TRANSFER:
            emitted = n.origin.to_dict()
            assert "traffic_class" not in emitted
            assert "priority" not in emitted


def test_class_router_dispatches_on_traffic_class_and_prepares_once(frozen):
    config = RunConfig.load(TOPO)
    reserved_router = FrozenPathRouter(ClosLeastLoadedRouter(config.topology), frozen)
    tail_router = ClosEcmpRouter(config.topology, seed=TAIL_SEED)
    router = ClassRouter(
        {"reserved": reserved_router, "tail": tail_router}, default=reserved_router
    )
    state = NetworkState(config.topology)
    router.prepare(state)

    def node(cls):
        return TransferNode(
            uid="x", name="x", src_rank=0, dst_rank=7, size_bytes=1,
            origin=Origin(source_node="t", traffic_class=cls),
        )

    assert router.route_node(node("reserved"), state).links == frozen[(0, 7)].links
    assert router.route_node(node("tail"), state).links == tail_router.route(0, 7, state).links
    assert router.route_node(node(None), state).links == frozen[(0, 7)].links
    assert router.route_node(node("unknown"), state).links == frozen[(0, 7)].links
    assert router.route(0, 7, state).links == frozen[(0, 7)].links
