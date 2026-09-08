"""Write a decided byte matrix as Chakra SEND/RECV traces for ASTRA-sim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from puppeteer.io.chakra import schema
from puppeteer.io.chakra.framing import encode_message

from .policy import slots_to_bytes


def _attr(name: str, **kwargs):
    from puppeteer.io.chakra.schema import _et  # type: ignore[attr-defined]

    return _et.AttributeProto(name=name, **kwargs)


def _node(next_id: list, name: str, node_type: int):
    node = schema.Node()
    node.id = next_id[0]
    next_id[0] += 1
    node.name = name
    node.type = node_type
    return node


def _link(node, parents) -> None:
    for parent in parents:
        if parent is not None:
            node.data_deps.append(parent.id)


def _compute(next_id, name: str, parents=()):
    node = _node(next_id, name, schema.COMP_NODE)
    node.attr.append(_attr("is_cpu_op", bool_val=False))
    node.attr.append(_attr("num_ops", int64_val=1))
    node.attr.append(_attr("tensor_size", uint64_val=0))
    _link(node, parents)
    return node


def _send(next_id, src: int, dst: int, size: int, tag: int, parents=()):
    node = _node(next_id, "DISP_SEND_{}_{}".format(src, dst), schema.COMM_SEND_NODE)
    node.attr.append(_attr("is_cpu_op", bool_val=False))
    node.attr.append(_attr("comm_size", int64_val=size))
    node.attr.append(_attr("comm_src", int32_val=src))
    node.attr.append(_attr("comm_dst", int32_val=dst))
    node.attr.append(_attr("comm_tag", int32_val=tag))
    _link(node, parents)
    return node


def _recv(next_id, src: int, dst: int, size: int, tag: int, parents=()):
    node = _node(next_id, "DISP_RECV_{}_{}".format(src, dst), schema.COMM_RECV_NODE)
    node.attr.append(_attr("is_cpu_op", bool_val=False))
    node.attr.append(_attr("comm_size", int64_val=size))
    node.attr.append(_attr("comm_src", int32_val=src))
    node.attr.append(_attr("comm_dst", int32_val=dst))
    node.attr.append(_attr("comm_tag", int32_val=tag))
    _link(node, parents)
    return node


def write_comm_et(
    M: np.ndarray,
    out_dir: str,
    *,
    name: str = "moe_dispatch",
    bytes_per_slot: Optional[int] = None,
    et_subdir: str = "et",
) -> int:
    """Write per-rank ``{name}.{rank}.et`` plus ``comm_groups.json``."""
    matrix = np.asarray(M)
    if bytes_per_slot is not None:
        matrix = slots_to_bytes(matrix, bytes_per_slot)
    n = int(matrix.shape[0])
    dest = Path(out_dir)
    et_dir = dest / et_subdir
    et_dir.mkdir(parents=True, exist_ok=True)

    for rank in range(n):
        next_id = [0]
        messages = [schema.GlobalMetadata(version="0.0.4")]
        root = _compute(next_id, "ROOT")
        messages.append(root)
        for src in range(n):
            if src == rank:
                continue
            size = int(matrix[src, rank])
            if size <= 0:
                continue
            messages.append(
                _recv(next_id, src, rank, size, tag=src * n + rank, parents=(root,))
            )
        for dst in range(n):
            if dst == rank:
                continue
            size = int(matrix[rank, dst])
            if size <= 0:
                continue
            messages.append(
                _send(next_id, rank, dst, size, tag=rank * n + dst, parents=(root,))
            )
        with (et_dir / "{}.{:d}.et".format(name, rank)).open("wb") as handle:
            for message in messages:
                encode_message(handle, message)

    (dest / "comm_groups.json").write_text(json.dumps({"0": list(range(n))}) + "\n")
    return n


def write_phased_et(
    phases,
    out_dir: str,
    *,
    name: str = "moe_forward",
    bytes_per_slot: Optional[int] = None,
    et_subdir: str = "et",
) -> int:
    """Sequential dispatch/combine (or any) phases on one rank's ET."""
    dest = Path(out_dir)
    et_dir = dest / et_subdir
    et_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for matrix in phases:
        arr = np.asarray(matrix)
        if bytes_per_slot is not None:
            arr = slots_to_bytes(arr, bytes_per_slot)
        converted.append(arr)
    if not converted:
        raise ValueError("write_phased_et needs at least one phase")
    n = int(converted[0].shape[0])
    for rank in range(n):
        next_id = [0]
        messages = [schema.GlobalMetadata(version="0.0.4")]
        prev = _compute(next_id, "ROOT")
        messages.append(prev)
        for phase, matrix in enumerate(converted):
            comm = []
            for src in range(n):
                if src == rank:
                    continue
                size = int(matrix[src, rank])
                if size <= 0:
                    continue
                node = _recv(
                    next_id,
                    src,
                    rank,
                    size,
                    tag=phase * n * n + src * n + rank,
                    parents=(prev,),
                )
                messages.append(node)
                comm.append(node)
            for dst in range(n):
                if dst == rank:
                    continue
                size = int(matrix[rank, dst])
                if size <= 0:
                    continue
                node = _send(
                    next_id,
                    rank,
                    dst,
                    size,
                    tag=phase * n * n + rank * n + dst,
                    parents=(prev,),
                )
                messages.append(node)
                comm.append(node)
            prev = _compute(next_id, "BARRIER_{}".format(phase), parents=comm or (prev,))
            messages.append(prev)
        with (et_dir / "{}.{:d}.et".format(name, rank)).open("wb") as handle:
            for message in messages:
                encode_message(handle, message)
    (dest / "comm_groups.json").write_text(json.dumps({"0": list(range(n))}) + "\n")
    return n
