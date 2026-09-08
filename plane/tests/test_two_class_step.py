import numpy as np
import pytest

from moe_feeder.config import DEFAULT_TOPOLOGY
from moe_feeder.graph import graph_two_class_step
from moe_feeder.planner import WIRE_S_PER_SLOT, delivered_by, freeze_paths, plan_two_class_step

TOPO = str(DEFAULT_TOPOLOGY)


def _case(seed=0):
    rng = np.random.default_rng(seed)
    M = rng.uniform(4000, 9000, size=(8, 8))
    np.fill_diagonal(M, 12000)
    E = M * rng.uniform(0.95, 1.05, size=M.shape)
    R = np.minimum(M, np.floor(E))
    return R, M - R, freeze_paths(E, TOPO)


def test_graph_structure_counts():
    R, T, _ = _case()
    from puppeteer.config import RunConfig

    topo = RunConfig.load(TOPO).topology
    g = graph_two_class_step(R * 2048, T * 2048, R.sum(0), T.sum(0), compute_s_per_slot=1e-7, structure="chunked", topology=topo)
    kinds = [n.uid.split(":")[1] for n in g.nodes.values() if n.uid.startswith("r")]
    assert kinds.count("comp_res") == 8 and kinds.count("comp_tail") == 8
    for d in range(8):
        parents = g.parents["r{}:comp_tail".format(d)]
        assert "r{}:comp_res".format(d) in parents
        assert all(("tail:disp" in p) or p.endswith("comp_res") for p in parents)
    g2 = graph_two_class_step(R * 2048, T * 2048, R.sum(0), T.sum(0), compute_s_per_slot=1e-7, structure="monolithic", topology=topo)
    off = ~np.eye(8, dtype=bool)
    for semantics in ("nccl", "rdma"):
        g2 = graph_two_class_step(R * 2048, T * 2048, R.sum(0), T.sum(0), compute_s_per_slot=1e-7,
                                  structure="monolithic", topology=topo, semantics=semantics)
        for d in range(8):
            parents = g2.parents["r{}:comp".format(d)]
            assert any("reserved:disp" in p for p in parents)
            has_in = bool((T[:, d] * off[:, d]).sum() > 0)
            has_out = bool((T[d, :] * off[d, :]).sum() > 0)
            expect = (has_in or has_out) if semantics == "nccl" else has_in
            assert any("tail:disp" in p for p in parents) == expect
            for uid in g2.children["r{}:comp".format(d)]:
                if uid.startswith("xfer:") and ":comb:" in uid:
                    src, dst = int(uid.split(":")[3]), int(uid.split(":")[4])
                    if semantics == "nccl":
                        assert "r{}:comp".format(dst) in g2.parents[uid]
    g3 = graph_two_class_step(R * 2048, np.zeros_like(T), R.sum(0), np.zeros(8), compute_s_per_slot=1e-9,
                              structure="chunked", topology=topo, compute_floor_s=3e-5)
    assert g3.nodes["r0:comp_tail"].duration_hint == 0.0
    assert g3.nodes["r0:comp_res"].duration_hint == pytest.approx(max(3e-5, 1e-9 * R.sum(0)[0]))


@pytest.mark.parametrize("allocator", ["tte-priority", "max-min"])
def test_chunked_reserved_schedule_unchanged(allocator):
    R, T, paths = _case(1)
    for r in (0.0, 1.0):
        base = plan_two_class_step(R, T, TOPO, frozen_paths=paths, compute_s_per_slot=r * WIRE_S_PER_SLOT, structure="chunked", reserved_only=True, allocator=allocator)
        two = plan_two_class_step(R, T, TOPO, frozen_paths=paths, compute_s_per_slot=r * WIRE_S_PER_SLOT, structure="chunked", allocator=allocator)
        assert two["reserved_dispatch_finish_s"] == pytest.approx(base["reserved_dispatch_finish_s"], rel=1e-9)
        assert two["reserved_combine_finish_s"] == pytest.approx(base["reserved_combine_finish_s"], rel=1e-9)
        assert two["step_makespan_s"] >= base["step_makespan_s"] - 1e-12
        assert two["allocator"] == allocator


def test_monolithic_waits_for_tail_and_deadline_readout():
    R, T, paths = _case(2)
    base = plan_two_class_step(R, T, TOPO, frozen_paths=paths, compute_s_per_slot=0.5 * WIRE_S_PER_SLOT, structure="chunked", reserved_only=True)
    mono = plan_two_class_step(R, T, TOPO, frozen_paths=paths, compute_s_per_slot=0.5 * WIRE_S_PER_SLOT, structure="monolithic")
    chunk = plan_two_class_step(R, T, TOPO, frozen_paths=paths, compute_s_per_slot=0.5 * WIRE_S_PER_SLOT, structure="chunked")
    assert mono["step_makespan_s"] >= base["step_makespan_s"] - 1e-12
    assert chunk["step_makespan_s"] >= base["step_makespan_s"] - 1e-12
    assert abs(mono["step_makespan_s"] - chunk["step_makespan_s"]) / base["step_makespan_s"] < 0.05
    dl = chunk["deadline"]
    tot = sum(dl["tail_dispatch_bytes"].values())
    assert tot == chunk["tail_bytes"]
    fr = [sum(v["dispatch_delivered"].values()) / tot for _, v in sorted(dl["by_delta"].items())]
    assert all(0.0 <= f <= 1.0 + 1e-9 for f in fr)
    assert all(fr[i] <= fr[i + 1] + 1e-9 for i in range(len(fr) - 1))  # more slack, more delivered
    nat = sum(dl["natural_dispatch_delivered"].values()) / tot
    assert 0.0 <= nat <= 1.0 + 1e-9


def test_delivered_by_integrates_schedule():
    R, T, paths = _case(3)
    two = plan_two_class_step(R, T, TOPO, frozen_paths=paths, structure="chunked")
    for f in two["plan"].flows:
        assert delivered_by(f, f.start_s) == 0.0
        assert delivered_by(f, f.finish_s + 1e-9) == pytest.approx(f.size_bytes)
        mid = 0.5 * (f.start_s + f.finish_s)
        assert 0.0 <= delivered_by(f, mid) <= f.size_bytes + 1e-6


def test_quota_replan_matches_one_pass_integration():
    from moe_feeder.planner import deadline_quota_replan

    R, T, paths = _case(4)
    base = plan_two_class_step(R, T, TOPO, frozen_paths=paths, compute_s_per_slot=0.1 * WIRE_S_PER_SLOT, structure="chunked")
    for L in (0.0, 1e-4, 5e-4):
        cut = deadline_quota_replan(R, T, base, L, TOPO, frozen_paths=paths,
                                    compute_s_per_slot=0.1 * WIRE_S_PER_SLOT, structure="chunked")
        assert cut["dispatch_overshoot_s"] <= 1e-6 + 1e-3 * base["reserved_dispatch_finish_s"]
        assert 0.0 <= cut["dispatch_quota_bytes"].sum() <= cut["tail_bytes"].sum() + 1e-6
        assert cut["cut_step_makespan_s"] >= base["reserved_combine_finish_s"] - 1e-12
