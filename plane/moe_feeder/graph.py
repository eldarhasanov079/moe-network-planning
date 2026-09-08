"""Build a Puppeteer JobGraph from a byte matrix — skip Chakra."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from puppeteer.ir import ComputeNode, JobGraph, Origin, ProcessGroup, TransferNode


def _split_bytes(size: int, parts: int) -> List[int]:
    if parts <= 1:
        return [size]
    base, rem = divmod(int(size), parts)
    return [base + 1] * rem + [base] * (parts - rem)


def graph_from_matrix(
    M_bytes: np.ndarray,
    *,
    envelope_slot: Optional[str] = None,
    name: str = "moe_a2a",
    topology=None,
    spray: bool = False,
) -> JobGraph:
    """One root compute per rank, then a TransferNode per off-diagonal cell > 0."""
    n = int(M_bytes.shape[0])
    if M_bytes.shape != (n, n):
        raise ValueError("matrix must be square, got {}".format(M_bytes.shape))
    graph = JobGraph(num_ranks=n)
    graph.process_groups["ep"] = ProcessGroup("ep", tuple(range(n)))

    roots = []
    exits = []
    for rank in range(n):
        root = ComputeNode(
            uid="r{}:root".format(rank),
            name="root",
            rank=rank,
            flops=0.0,
            duration_hint=0.0,
        )
        exit_n = ComputeNode(
            uid="r{}:exit".format(rank),
            name="exit",
            rank=rank,
            flops=0.0,
            duration_hint=0.0,
        )
        graph.add_node(root)
        graph.add_node(exit_n)
        roots.append(root.uid)
        exits.append(exit_n.uid)

    incoming = {rank: [] for rank in range(n)}
    outgoing = {rank: [] for rank in range(n)}
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            size = int(M_bytes[src, dst])
            if size <= 0:
                continue
            n_parts = 1
            if spray:
                if topology is None:
                    raise ValueError("spray graphs need a Clos topology")
                from puppeteer.planner.router import enumerate_clos_routes

                n_parts = max(1, len(enumerate_clos_routes(topology, src, dst)))
            for chunk, piece in enumerate(_split_bytes(size, n_parts)):
                if piece <= 0:
                    continue
                uid = "xfer:{}:{}:{}".format(src, dst, chunk) if n_parts > 1 else "xfer:{}:{}".format(src, dst)
                node = TransferNode(
                    uid=uid,
                    name="{}/r{}->r{}".format(name, src, dst),
                    src_rank=src,
                    dst_rank=dst,
                    size_bytes=piece,
                    origin=Origin(
                        source_node=name,
                        phase="dispatch",
                        matrix_entry=(src, dst),
                        envelope_slot=envelope_slot,
                        chunk=chunk if n_parts > 1 else None,
                    ),
                )
                graph.add_node(node)
                graph.add_edge(roots[src], uid)
                graph.add_edge(roots[dst], uid)
                outgoing[src].append(uid)
                incoming[dst].append(uid)

    for rank in range(n):
        parents = outgoing[rank] + incoming[rank]
        if not parents:
            graph.add_edge(roots[rank], exits[rank])
        else:
            for p in parents:
                graph.add_edge(p, exits[rank])
    return graph


def graph_from_classes(
    classes: Sequence[Dict[str, Any]],
    *,
    name: str = "moe_a2a",
    topology=None,
    envelope_slot: Optional[str] = None,
) -> JobGraph:
    """Several byte classes over one shared root/exit per rank."""
    if not classes:
        raise ValueError("graph_from_classes needs at least one class")
    n: Optional[int] = None
    seen = set()
    for cls in classes:
        cls_name = str(cls["name"])
        if cls_name in seen:
            raise ValueError("duplicate traffic class {!r}".format(cls_name))
        seen.add(cls_name)
        M = np.asarray(cls["M_bytes"])
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(
                "class {!r} matrix must be square, got {}".format(cls_name, M.shape)
            )
        if n is None:
            n = int(M.shape[0])
        elif int(M.shape[0]) != n:
            raise ValueError("all class matrices must share one rank count")
    assert n is not None

    graph = JobGraph(num_ranks=n)
    graph.process_groups["ep"] = ProcessGroup("ep", tuple(range(n)))

    roots = []
    exits = []
    for rank in range(n):
        root = ComputeNode(
            uid="r{}:root".format(rank),
            name="root",
            rank=rank,
            flops=0.0,
            duration_hint=0.0,
        )
        exit_n = ComputeNode(
            uid="r{}:exit".format(rank),
            name="exit",
            rank=rank,
            flops=0.0,
            duration_hint=0.0,
        )
        graph.add_node(root)
        graph.add_node(exit_n)
        roots.append(root.uid)
        exits.append(exit_n.uid)

    incoming = {rank: [] for rank in range(n)}
    outgoing = {rank: [] for rank in range(n)}
    for cls in classes:
        cls_name = str(cls["name"])
        priority = int(cls.get("priority", 0))
        spray = bool(cls.get("spray", False))
        M = np.asarray(cls["M_bytes"])
        for src in range(n):
            for dst in range(n):
                if src == dst:
                    continue
                size = int(M[src, dst])
                if size <= 0:
                    continue
                n_parts = 1
                if spray:
                    if topology is None:
                        raise ValueError("spray classes need a Clos topology")
                    from puppeteer.planner.router import enumerate_clos_routes

                    n_parts = max(1, len(enumerate_clos_routes(topology, src, dst)))
                for chunk, piece in enumerate(_split_bytes(size, n_parts)):
                    if piece <= 0:
                        continue
                    uid = "xfer:{}:{}:{}".format(cls_name, src, dst)
                    if n_parts > 1:
                        uid += ":{}".format(chunk)
                    node = TransferNode(
                        uid=uid,
                        name="{}/r{}->r{}".format(name, src, dst),
                        src_rank=src,
                        dst_rank=dst,
                        size_bytes=piece,
                        origin=Origin(
                            source_node=name,
                            phase="dispatch",
                            matrix_entry=(src, dst),
                            envelope_slot=envelope_slot,
                            chunk=chunk if n_parts > 1 else None,
                            traffic_class=cls_name,
                            priority=priority,
                        ),
                    )
                    graph.add_node(node)
                    graph.add_edge(roots[src], uid)
                    graph.add_edge(roots[dst], uid)
                    outgoing[src].append(uid)
                    incoming[dst].append(uid)

    for rank in range(n):
        parents = outgoing[rank] + incoming[rank]
        if not parents:
            graph.add_edge(roots[rank], exits[rank])
        else:
            for p in parents:
                graph.add_edge(p, exits[rank])
    return graph


def graph_two_class_step(
    reserved_bytes: np.ndarray,
    tail_bytes: np.ndarray,
    reserved_slots_in: np.ndarray,
    tail_slots_in: np.ndarray,
    *,
    compute_s_per_slot: float,
    structure: str = "chunked",
    topology=None,
    spray_tail: bool = False,
    name: str = "moe_step",
    semantics: str = "nccl",
    compute_floor_s: float = 0.0,
    tail_combine_priority: int = 2,
) -> JobGraph:
    """One MoE layer step: dispatch -> expert compute -> combine, two classes, one graph."""
    if structure not in ("monolithic", "chunked"):
        raise ValueError("structure must be 'monolithic' or 'chunked', got {!r}".format(structure))
    if semantics not in ("nccl", "rdma"):
        raise ValueError("semantics must be 'nccl' or 'rdma', got {!r}".format(semantics))

    def _dur(slots: float) -> float:
        return 0.0 if slots <= 0 else max(float(compute_floor_s), c * float(slots))
    R = np.asarray(reserved_bytes)
    T = np.asarray(tail_bytes)
    n = int(R.shape[0])
    if R.shape != (n, n) or T.shape != (n, n):
        raise ValueError("class matrices must be square and equal, got {} and {}".format(R.shape, T.shape))
    c = float(compute_s_per_slot)
    if c < 0:
        raise ValueError("compute_s_per_slot must be >= 0")

    graph = JobGraph(num_ranks=n)
    graph.process_groups["ep"] = ProcessGroup("ep", tuple(range(n)))
    roots, exits = [], []
    for rank in range(n):
        root = ComputeNode(uid="r{}:root".format(rank), name="root", rank=rank, flops=0.0, duration_hint=0.0)
        exit_n = ComputeNode(uid="r{}:exit".format(rank), name="exit", rank=rank, flops=0.0, duration_hint=0.0)
        graph.add_node(root)
        graph.add_node(exit_n)
        roots.append(root.uid)
        exits.append(exit_n.uid)

    def _n_parts(src: int, dst: int, spray: bool) -> int:
        if not spray:
            return 1
        if topology is None:
            raise ValueError("a sprayed tail needs a Clos topology")
        from puppeteer.planner.router import enumerate_clos_routes

        return max(1, len(enumerate_clos_routes(topology, src, dst)))

    def _add_flows(M: np.ndarray, cls: str, prio: int, phase: str, spray: bool, parents_of):
        """Add one flow per non-zero off-diagonal cell; ``phase`` is 'disp' or 'comb'."""
        made = {}
        for s in range(n):
            for d in range(n):
                if s == d:
                    continue
                size = int(M[s, d])
                if size <= 0:
                    continue
                src, dst = (s, d) if phase == "disp" else (d, s)
                parts = _n_parts(src, dst, spray)
                uids = []
                for chunk, piece in enumerate(_split_bytes(size, parts)):
                    if piece <= 0:
                        continue
                    uid = "xfer:{}:{}:{}:{}".format(cls, phase, src, dst)
                    if parts > 1:
                        uid += ":{}".format(chunk)
                    node = TransferNode(
                        uid=uid,
                        name="{}/{}/r{}->r{}".format(name, phase, src, dst),
                        src_rank=src,
                        dst_rank=dst,
                        size_bytes=piece,
                        origin=Origin(
                            source_node=name,
                            phase="dispatch" if phase == "disp" else "combine",
                            matrix_entry=(s, d),
                            envelope_slot=None,
                            chunk=chunk if parts > 1 else None,
                            traffic_class=cls,
                            priority=prio,
                        ),
                    )
                    graph.add_node(node)
                    for p in parents_of(src, dst):
                        graph.add_edge(p, uid)
                    uids.append(uid)
                made[(src, dst)] = uids
        return made

    disp_res = _add_flows(R, "reserved", 0, "disp", False, lambda s, d: (roots[s], roots[d]))
    disp_tail = _add_flows(T, "tail", 1, "disp", spray_tail, lambda s, d: (roots[s], roots[d]))

    def _into(flows, d: int):
        return [uid for (s, dd), uids in flows.items() if dd == d for uid in uids]

    def _outof(flows, d: int):
        return [uid for (ss, dd), uids in flows.items() if ss == d for uid in uids]

    def _gate(flows, d: int):
        """Dispatch flows a kernel on d must wait for under the chosen semantics."""
        return _into(flows, d) + (_outof(flows, d) if semantics == "nccl" else [])

    comp_res_uid, comp_tail_uid, comp_uid = {}, {}, {}
    for d in range(n):
        if structure == "monolithic":
            node = ComputeNode(
                uid="r{}:comp".format(d), name="expert", rank=d, flops=0.0,
                duration_hint=_dur(reserved_slots_in[d] + tail_slots_in[d]),
            )
            graph.add_node(node)
            graph.add_edge(roots[d], node.uid)
            for p in _gate(disp_res, d) + _gate(disp_tail, d):
                graph.add_edge(p, node.uid)
            comp_uid[d] = node.uid
        else:
            res = ComputeNode(
                uid="r{}:comp_res".format(d), name="expert_reserved", rank=d, flops=0.0,
                duration_hint=_dur(reserved_slots_in[d]),
            )
            tail = ComputeNode(
                uid="r{}:comp_tail".format(d), name="expert_tail", rank=d, flops=0.0,
                duration_hint=_dur(tail_slots_in[d]),
            )
            graph.add_node(res)
            graph.add_node(tail)
            graph.add_edge(roots[d], res.uid)
            for p in _gate(disp_res, d):
                graph.add_edge(p, res.uid)
            graph.add_edge(res.uid, tail.uid)
            for p in _gate(disp_tail, d):
                graph.add_edge(p, tail.uid)
            comp_res_uid[d], comp_tail_uid[d] = res.uid, tail.uid

    def _comb_parents(kernels):
        if semantics == "nccl":
            return lambda src, dst: (kernels[src], kernels[dst])
        return lambda src, dst: (kernels[src],)

    if structure == "monolithic":
        comb_res = _add_flows(R, "reserved", 0, "comb", False, _comb_parents(comp_uid))
        comb_tail = _add_flows(T, "tail", int(tail_combine_priority), "comb", spray_tail, _comb_parents(comp_uid))
    else:
        comb_res = _add_flows(R, "reserved", 0, "comb", False, _comb_parents(comp_res_uid))
        comb_tail = _add_flows(T, "tail", int(tail_combine_priority), "comb", spray_tail, _comb_parents(comp_tail_uid))

    for rank in range(n):
        parents = _into(comb_res, rank) + _into(comb_tail, rank)
        parents += [comp_uid[rank]] if structure == "monolithic" else [comp_res_uid[rank], comp_tail_uid[rank]]
        for p in parents:
            graph.add_edge(p, exits[rank])
    return graph
