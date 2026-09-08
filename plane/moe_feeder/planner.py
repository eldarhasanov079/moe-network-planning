"""Hand a byte matrix to Puppeteer and collect the Clos plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from puppeteer.config import RunConfig
from puppeteer.pipeline import plan_graph
from puppeteer.plan.model import Plan
from puppeteer.plan.writers import write_plan, write_summary, write_timeline
from puppeteer.planner.router import (
    ClassRouter,
    ClosEcmpRouter,
    ClosLeastLoadedRouter,
    ClosSprayRouter,
    FrozenPathRouter,
    Router,
)
from puppeteer.topology import Route

from .graph import graph_from_classes, graph_from_matrix, graph_two_class_step
from .policy import slots_to_bytes


def _config(topology: Union[str, RunConfig]) -> RunConfig:
    """Accept a topology yaml path (the usual case) or an already-loaded RunConfig."""
    if isinstance(topology, RunConfig):
        return topology
    return RunConfig.load(str(topology))


def paths_from_plan(plan: Plan, topology) -> Dict[Tuple[int, int], Route]:
    """Turn a planned Clos path set into a frozen ``(src, dst) -> Route`` map."""
    out: Dict[Tuple[int, int], Route] = {}
    for flow in plan.flows:
        cap = (
            topology.path_capacity(flow.path)
            if flow.path
            else topology.scale_up_bps
        )
        out[(flow.src_rank, flow.dst_rank)] = Route(
            links=list(flow.path),
            scope=flow.scope,
            capacity_bps=cap,
        )
    return out


def _payload(run, M_slots: np.ndarray, M_bytes: np.ndarray) -> Dict[str, Any]:
    return {
        "metrics": run.plan.metrics,
        "meta": run.plan.meta,
        "n_flows": len(run.plan.flows),
        "flow_bytes": int(sum(f.size_bytes for f in run.plan.flows)),
        "iteration_time_s": run.plan.metrics.get("iteration_time_s"),
        "exposed_comm_s": run.plan.metrics.get("exposed_comm_s"),
        "ideal_iteration_time_s": run.plan.metrics.get("ideal_iteration_time_s"),
        "plan_wall_s": run.plan.metrics.get("plan_wall_s"),
        "plan": run.plan,
    }


def plan_matrix(
    M_slots: np.ndarray,
    topology: str,
    bytes_per_slot: int = 2048,
    envelope_slot: Optional[str] = None,
    out_dir: Optional[str] = None,
    route_slots: Optional[np.ndarray] = None,
    router_kind: str = "least-loaded",
    router_seed: int = 0,
    frozen_paths: Optional[Dict[Tuple[int, int], Route]] = None,
) -> Dict[str, Any]:
    """Plan ``M_slots`` as P2P sizes."""
    config = _config(topology)
    M_bytes = slots_to_bytes(M_slots, bytes_per_slot)
    frozen = frozen_paths
    if frozen is None and route_slots is not None:
        frozen = freeze_paths(route_slots, config, bytes_per_slot=bytes_per_slot)

    graph = graph_from_matrix(
        M_bytes,
        envelope_slot=envelope_slot,
        topology=config.topology,
        spray=(router_kind.replace("_", "-") in ("spray", "rps")),
    )
    run = plan_graph(
        graph,
        config,
        workload_path="moe-feeder:{}".format(envelope_slot or "matrix"),
        frozen_paths=frozen,
        router_kind=router_kind,
        router_seed=router_seed,
    )
    payload = _payload(run, M_slots, M_bytes)
    if out_dir is not None:
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        write_plan(run.plan, str(dest / "plan.json"))
        write_summary(run.plan, str(dest / "summary.txt"))
        write_timeline(run.plan, str(dest / "timeline.json"))
        np.save(dest / "matrix_slots.npy", np.asarray(M_slots))
        np.save(dest / "matrix_bytes.npy", M_bytes)
        if route_slots is not None:
            np.save(dest / "route_slots.npy", np.asarray(route_slots))
        payload["out_dir"] = str(dest)
    return payload


def freeze_paths(
    route_slots: np.ndarray,
    topology: Union[str, RunConfig],
    bytes_per_slot: int = 2048,
) -> Dict[Tuple[int, int], Route]:
    """Plan ``route_slots`` least-loaded once and return its ``(src, dst) -> Route`` map."""
    config = _config(topology)
    route_bytes = slots_to_bytes(route_slots, bytes_per_slot)
    route_graph = graph_from_matrix(route_bytes, envelope_slot="routes")
    route_run = plan_graph(
        route_graph,
        config,
        workload_path="moe-feeder:routes",
        router_kind="least-loaded",
    )
    return paths_from_plan(route_run.plan, config.topology)


def _class_summary(plan: Plan) -> Dict[str, Dict[str, Any]]:
    """Per-class makespan / start / bytes / bytes per link, read off the flows' origins."""
    out: Dict[str, Dict[str, Any]] = {}
    for flow in plan.flows:
        cls = flow.origin.get("traffic_class")
        if cls is None:
            continue
        entry = out.setdefault(
            cls,
            {
                "makespan_s": 0.0,
                "start_s": float("inf"),
                "bytes": 0,
                "n_flows": 0,
                "link_bytes": {},
            },
        )
        entry["makespan_s"] = max(entry["makespan_s"], flow.finish_s)
        entry["start_s"] = min(entry["start_s"], flow.start_s)
        entry["bytes"] += int(flow.size_bytes)
        entry["n_flows"] += 1
        for link in flow.path:
            entry["link_bytes"][link] = entry["link_bytes"].get(link, 0) + int(
                flow.size_bytes
            )
    return out


def plan_two_class(
    M_reserved_slots: np.ndarray,
    M_tail_slots: np.ndarray,
    topology: Union[str, RunConfig],
    *,
    frozen_paths: Dict[Tuple[int, int], Route],
    tail_router_kind: str = "ecmp",
    tail_seed: int = 0,
    bytes_per_slot: int = 2048,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Plan a reserved class and a tail class in one Puppeteer event loop."""
    config = _config(topology)
    if config.planner.min_rate_bps != 0:
        raise ValueError(
            "plan_two_class needs planner.min_rate == 0 (got {} bps): the rate floor "
            "is handed out before priority, so the tail class would steal capacity "
            "from the reserved class".format(config.planner.min_rate_bps)
        )
    if frozen_paths is None:
        raise ValueError("plan_two_class needs frozen_paths; see freeze_paths()")
    kind = (tail_router_kind or "ecmp").replace("_", "-")
    if kind not in ("ecmp", "spray", "rps"):
        raise ValueError(
            "tail_router_kind must be 'ecmp' or 'spray'/'rps', got {!r}".format(
                tail_router_kind
            )
        )
    spray = kind in ("spray", "rps")

    reserved_bytes = slots_to_bytes(M_reserved_slots, bytes_per_slot)
    tail_bytes = slots_to_bytes(M_tail_slots, bytes_per_slot)
    reserved_cls = {
        "name": "reserved",
        "priority": 0,
        "M_bytes": reserved_bytes,
        "spray": False,
    }
    tail_cls = {"name": "tail", "priority": 1, "M_bytes": tail_bytes, "spray": spray}

    reserved_graph = graph_from_classes([reserved_cls], topology=config.topology)
    reserved_run = plan_graph(
        reserved_graph,
        config,
        workload_path="moe-feeder:reserved",
        frozen_paths=frozen_paths,
        router_kind="least-loaded",
    )

    def router_factory() -> Router:
        reserved_router = FrozenPathRouter(
            ClosLeastLoadedRouter(config.topology), frozen_paths
        )
        tail_router: Router
        if spray:
            tail_router = ClosSprayRouter(config.topology)
        else:
            tail_router = ClosEcmpRouter(config.topology, seed=tail_seed)
        return ClassRouter(
            {"reserved": reserved_router, "tail": tail_router}, default=reserved_router
        )

    merged_graph = graph_from_classes(
        [reserved_cls, tail_cls], topology=config.topology
    )
    run = plan_graph(
        merged_graph,
        config,
        workload_path="moe-feeder:two-class:{}".format(kind),
        router_factory=router_factory,
        tte_horizon=reserved_run.ideal.makespan,
    )

    classes = _class_summary(run.plan)
    reserved_makespan = classes.get("reserved", {}).get("makespan_s", 0.0)
    tail_makespan = classes.get("tail", {}).get("makespan_s", 0.0)
    payload: Dict[str, Any] = {
        "classes": classes,
        "reserved_alone_makespan_s": reserved_run.result.makespan,
        "total_makespan_s": run.result.makespan,
        "tail_extension_s": max(0.0, tail_makespan - reserved_makespan),
        "tail_router_kind": kind,
        "tail_seed": tail_seed,
        "plan": run.plan,
        "reserved_plan": reserved_run.plan,
        "metrics": run.plan.metrics,
    }
    if out_dir is not None:
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        write_plan(run.plan, str(dest / "plan.json"))
        write_summary(run.plan, str(dest / "summary.txt"))
        write_timeline(run.plan, str(dest / "timeline.json"))
        write_plan(reserved_run.plan, str(dest / "reserved_plan.json"))
        np.save(dest / "reserved_slots.npy", np.asarray(M_reserved_slots))
        np.save(dest / "tail_slots.npy", np.asarray(M_tail_slots))
        payload["out_dir"] = str(dest)
    return payload


WIRE_S_PER_SLOT = 2048 * 8.0 / 100e9


def delivered_by(flow, T: float) -> float:
    """Bytes of ``flow`` delivered by time ``T``, integrating its rate schedule."""
    if T <= flow.start_s:
        return 0.0
    if T >= flow.finish_s:
        return float(flow.size_bytes)
    segs = flow.rate_schedule
    total = 0.0
    for i, seg in enumerate(segs):
        t0 = seg.t_s
        t1 = segs[i + 1].t_s if i + 1 < len(segs) else flow.finish_s
        if t0 >= T:
            break
        total += seg.bps * (min(t1, T) - t0) / 8.0
    return float(min(total, flow.size_bytes))


def _flows_by_cell(plan: Plan, cls: str, phase: str):
    out: Dict[Tuple[int, int], list] = {}
    for flow in plan.flows:
        o = flow.origin
        if o.get("traffic_class") != cls or o.get("phase") != phase:
            continue
        cell = tuple(o.get("matrix_entry") or (flow.src_rank, flow.dst_rank))
        out.setdefault(cell, []).append(flow)
    return out


def plan_two_class_step(
    M_reserved_slots: np.ndarray,
    M_tail_slots: np.ndarray,
    topology: Union[str, RunConfig],
    *,
    frozen_paths: Dict[Tuple[int, int], Route],
    compute_s_per_slot: float = 0.0,
    structure: str = "chunked",
    tail_router_kind: str = "ecmp",
    tail_seed: int = 0,
    bytes_per_slot: int = 2048,
    deadline_deltas: Sequence[float] = (0.0, 0.0025, 0.005, 0.01, 0.02),
    reserved_only: bool = False,
    allocator: Optional[str] = None,
    semantics: str = "nccl",
    compute_floor_s: float = 0.0,
) -> Dict[str, Any]:
    """Plan one layer step with two classes and expert compute in ONE event loop."""
    config = _config(topology)
    if allocator is not None:
        import copy

        config = copy.deepcopy(config)
        config.planner.allocator = allocator
    if config.planner.min_rate_bps != 0:
        raise ValueError("plan_two_class_step needs planner.min_rate == 0")
    kind = (tail_router_kind or "ecmp").replace("_", "-")
    spray = kind in ("spray", "rps")
    R_slots = np.asarray(M_reserved_slots, dtype=np.float64)
    T_slots = np.zeros_like(R_slots) if reserved_only else np.asarray(M_tail_slots, dtype=np.float64)
    R_bytes = slots_to_bytes(R_slots, bytes_per_slot)
    T_bytes = slots_to_bytes(T_slots, bytes_per_slot)
    res_in = R_slots.sum(axis=0)
    tail_in = T_slots.sum(axis=0)

    graph = graph_two_class_step(
        R_bytes, T_bytes, res_in, tail_in,
        compute_s_per_slot=compute_s_per_slot, structure=structure,
        topology=config.topology, spray_tail=spray,
        semantics=semantics, compute_floor_s=compute_floor_s,
    )

    def router_factory() -> Router:
        reserved_router = FrozenPathRouter(ClosLeastLoadedRouter(config.topology), frozen_paths)
        tail_router: Router = ClosSprayRouter(config.topology) if spray else ClosEcmpRouter(config.topology, seed=tail_seed)
        return ClassRouter({"reserved": reserved_router, "tail": tail_router}, default=reserved_router)

    from puppeteer.analysis import ideal_pass

    if reserved_only:
        horizon = None
    else:
        alone = graph_two_class_step(
            R_bytes, np.zeros_like(T_bytes), res_in, np.zeros_like(tail_in),
            compute_s_per_slot=compute_s_per_slot, structure=structure,
            topology=config.topology, spray_tail=False,
            semantics=semantics, compute_floor_s=compute_floor_s,
        )
        horizon = ideal_pass(alone, config.compute, config.topology).makespan
    run = plan_graph(
        graph, config, workload_path="moe-feeder:step:{}".format(structure),
        router_factory=router_factory, tte_horizon=horizon,
    )
    plan, result = run.plan, run.result

    def _finish(cls: str, phase: str) -> float:
        vals = [f.finish_s for f in plan.flows if f.origin.get("traffic_class") == cls and f.origin.get("phase") == phase]
        return max(vals) if vals else 0.0

    n = int(R_slots.shape[0])
    res_disp, res_comb = _finish("reserved", "dispatch"), _finish("reserved", "combine")
    tail_disp, tail_comb = _finish("tail", "dispatch"), _finish("tail", "combine")
    comp_finish = {}
    for d in range(n):
        key = "r{}:comp".format(d) if structure == "monolithic" else "r{}:comp_res".format(d)
        comp_finish[d] = float(result.finish.get(key, 0.0))
    comp_tail_finish = {d: float(result.finish.get("r{}:comp_tail".format(d), 0.0)) for d in range(n)} if structure == "chunked" else {}

    payload: Dict[str, Any] = {
        "structure": structure,
        "compute_s_per_slot": float(compute_s_per_slot),
        "step_makespan_s": float(result.makespan),
        "reserved_dispatch_finish_s": res_disp,
        "reserved_combine_finish_s": res_comb,
        "tail_dispatch_finish_s": tail_disp,
        "tail_combine_finish_s": tail_comb,
        "expert_reserved_finish_s": comp_finish,
        "expert_tail_finish_s": comp_tail_finish,
        "reserved_bytes": int(R_bytes.sum() - np.trace(R_bytes)),
        "tail_bytes": int(T_bytes.sum() - np.trace(T_bytes)),
        "reserved_slots_total": float(R_slots.sum()),
        "tail_slots_total": float(T_slots.sum()),
        "tail_router_kind": kind,
        "tail_seed": tail_seed,
        "allocator": config.planner.allocator,
        "semantics": semantics,
        "compute_floor_s": float(compute_floor_s),
        "plan": plan,
    }
    if reserved_only or T_bytes.sum() <= 0:
        payload["deadline"] = {}
        return payload

    disp_cells = _flows_by_cell(plan, "tail", "dispatch")
    comb_cells = _flows_by_cell(plan, "tail", "combine")
    tail_disp_bytes = {cell: sum(f.size_bytes for f in fl) for cell, fl in disp_cells.items()}
    tail_comb_bytes = {cell: sum(f.size_bytes for f in fl) for cell, fl in comb_cells.items()}
    deadline: Dict[str, Any] = {"tail_dispatch_bytes": tail_disp_bytes, "tail_combine_bytes": tail_comb_bytes, "by_delta": {}}
    for delta in deadline_deltas:
        Td = res_disp * (1.0 + delta)
        Tc = res_comb * (1.0 + delta)
        deadline["by_delta"][float(delta)] = {
            "T_dispatch_s": Td,
            "T_combine_s": Tc,
            "dispatch_delivered": {cell: sum(delivered_by(f, Td) for f in fl) for cell, fl in disp_cells.items()},
            "combine_delivered": {cell: sum(delivered_by(f, Tc) for f in fl) for cell, fl in comb_cells.items()},
        }
    if structure == "chunked":
        deadline["natural_dispatch_delivered"] = {
            cell: sum(delivered_by(f, comp_finish[cell[1]]) for f in fl) for cell, fl in disp_cells.items()
        }
    payload["deadline"] = deadline
    return payload


def deadline_quota_replan(
    M_reserved_slots: np.ndarray,
    M_tail_slots: np.ndarray,
    base: Dict[str, Any],
    lateness_s: float,
    topology: Union[str, RunConfig],
    *,
    frozen_paths: Dict[Tuple[int, int], Route],
    bytes_per_slot: int = 2048,
    margin_slots: int = 1,
    iterations: int = 2,
    **plan_kwargs,
) -> Dict[str, Any]:
    """The deadline as a mechanism a dispatcher can execute: a sender-side byte quota."""
    T_d = base["reserved_dispatch_finish_s"] + lateness_s
    T_c = base["reserved_combine_finish_s"] + lateness_s
    plan = base["plan"]
    n = int(np.asarray(M_reserved_slots).shape[0])
    quota = np.zeros((n, n), dtype=np.float64)
    tail_bytes = np.zeros((n, n), dtype=np.float64)
    for cell, flows in _flows_by_cell(plan, "tail", "dispatch").items():
        quota[cell] = sum(delivered_by(f, T_d) for f in flows)
        tail_bytes[cell] = sum(f.size_bytes for f in flows)
    def _to_slots(q: np.ndarray) -> np.ndarray:
        return np.maximum(np.floor(q / float(bytes_per_slot) + 1e-9) - margin_slots, 0.0) * (q > 0)

    quota_slots = _to_slots(quota)
    cut = None
    for _ in range(max(1, iterations)):
        cut = plan_two_class_step(
            M_reserved_slots, quota_slots, topology, frozen_paths=frozen_paths,
            bytes_per_slot=bytes_per_slot, deadline_deltas=(), **plan_kwargs,
        )
        short = False
        for cell, flows in _flows_by_cell(cut["plan"], "tail", "dispatch").items():
            got = sum(delivered_by(f, T_d) for f in flows)
            want = quota_slots[cell] * bytes_per_slot
            if got + 1e-6 < want:
                quota_slots[cell] = _to_slots(np.array(got))
                short = True
        if not short:
            break
    quota = quota_slots * float(bytes_per_slot)
    cut_plan = cut["plan"]
    disp_flows = [f for f in cut_plan.flows if f.origin.get("traffic_class") == "tail" and f.origin.get("phase") == "dispatch"]
    overshoot = max([f.finish_s - T_d for f in disp_flows] or [0.0])
    comb_delivered = np.zeros((n, n), dtype=np.float64)
    comb_bytes = np.zeros((n, n), dtype=np.float64)
    for cell, flows in _flows_by_cell(cut_plan, "tail", "combine").items():
        comb_delivered[cell] = sum(delivered_by(f, T_c) for f in flows)   # cell = dispatch (s, d)
        comb_bytes[cell] = sum(f.size_bytes for f in flows)
    return {
        "lateness_s": float(lateness_s),
        "T_dispatch_s": T_d,
        "T_combine_s": T_c,
        "tail_bytes": tail_bytes,
        "dispatch_quota_bytes": quota,
        "dispatch_overshoot_s": float(overshoot),
        "combine_bytes": comb_bytes,
        "combine_delivered_bytes": comb_delivered,
        "cut_step_makespan_s": float(cut["step_makespan_s"]),
        "cut_tail_combine_finish_s": float(cut["tail_combine_finish_s"]),
        "cut_reserved_dispatch_finish_s": float(cut["reserved_dispatch_finish_s"]),
        "cut_reserved_combine_finish_s": float(cut["reserved_combine_finish_s"]),
    }
