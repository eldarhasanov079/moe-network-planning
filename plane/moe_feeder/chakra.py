"""Convert Chakra execution traces into PLANE source--destination matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from puppeteer.config import RunConfig
from puppeteer.io import read_workload, stitch
from puppeteer.ir import NodeKind, transfer_bytes
from puppeteer.pipeline import build_graph


class ChakraMatrixError(ValueError):
    """Raised when a Chakra workload cannot be reduced to rank-pair byte totals."""


def _run_config(topology: Union[str, Path, RunConfig]) -> RunConfig:
    if isinstance(topology, RunConfig):
        return topology
    return RunConfig.load(str(topology))


def matrix_from_chakra(
    workload_path: Union[str, Path],
    *,
    topology: Optional[Union[str, Path, RunConfig]] = None,
    strict: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return the aggregate off-rank byte matrix for one Chakra workload."""

    path = str(workload_path)
    if topology is None:
        graph = read_workload(path, "chakra-et", strict=strict)
        stitch(graph)
        if graph.collective_instances:
            raise ChakraMatrixError(
                "the Chakra workload contains collectives; pass a topology so they "
                "can be decomposed before building the byte matrix"
            )
        build_stats: Dict[str, Any] = {
            "graph": graph.stats(),
            "decomposition": {"collectives_expanded": 0, "flows_created": 0},
            "transfer_bytes": transfer_bytes(graph),
        }
    else:
        graph, build_stats = build_graph(
            path,
            _run_config(topology),
            workload_format="chakra-et",
            strict=strict,
        )

    matrix = np.zeros((graph.num_ranks, graph.num_ranks), dtype=np.int64)
    flows = 0
    for node in graph.of_kind(NodeKind.TRANSFER):
        src = int(node.src_rank)
        dst = int(node.dst_rank)
        if src == dst:
            continue
        if not (0 <= src < graph.num_ranks and 0 <= dst < graph.num_ranks):
            raise ChakraMatrixError(
                "transfer {!r} uses rank pair ({}, {}) outside 0..{}".format(
                    node.name, src, dst, graph.num_ranks - 1
                )
            )
        size = int(node.size_bytes)
        if size < 0:
            raise ChakraMatrixError(
                "transfer {!r} has a negative size ({})".format(node.name, size)
            )
        matrix[src, dst] += size
        flows += 1

    stats: Dict[str, Any] = {
        "workload": path,
        "ranks": graph.num_ranks,
        "off_rank_flows": flows,
        "off_rank_bytes": int(matrix.sum()),
        "warnings": list(graph.warnings),
        **build_stats,
    }
    return matrix, stats
