"""Shared netmodel toolkit -- map a traffic matrix onto topology link loads."""

from __future__ import annotations

import numpy as np


class FullMeshFabric:
    """Non-blocking fabric: directed link (s,d) load = off-diagonal M[s,d]."""

    name = "full_mesh"

    def __init__(self, num_devices: int):
        self.num_devices = num_devices

    def link_loads(self, M: np.ndarray) -> dict[tuple[int, int], float]:
        """Return {(s, d): load} for all s != d (the network links)."""
        D = self.num_devices
        loads: dict[tuple[int, int], float] = {}
        for s in range(D):
            for d in range(D):
                if s != d:
                    loads[(s, d)] = float(M[s, d])
        return loads

    def link_load_array(self, M: np.ndarray) -> np.ndarray:
        """Flat array of the off-diagonal (network) link loads -- handy for stats."""
        D = self.num_devices
        mask = ~np.eye(D, dtype=bool)
        return M[mask].astype(np.float64)


class TwoTierNodes:
    """Leaf-spine: intra-node direct links + one shared uplink/downlink per node."""

    name = "two_tier"

    def __init__(self, num_devices: int, node_size: int):
        if num_devices % node_size != 0:
            raise ValueError(f"num_devices ({num_devices}) must be divisible by "
                             f"node_size ({node_size}).")
        self.num_devices = num_devices
        self.node_size = node_size
        self.num_nodes = num_devices // node_size
        self.node_of = np.arange(num_devices) // node_size  # device -> node id

    def link_loads(self, M: np.ndarray) -> dict[tuple, float]:
        """Return a dict mixing two link kinds:"""
        D = self.num_devices
        node = self.node_of
        loads: dict[tuple, float] = {}
        for s in range(D):
            for d in range(D):
                if s != d and node[s] == node[d]:
                    loads[("intra", s, d)] = float(M[s, d])
        for n in range(self.num_nodes):
            src_in = node == n
            up = M[src_in, :][:, ~src_in].sum()       # leaving node n
            down = M[~src_in, :][:, src_in].sum()      # entering node n
            loads[("up", n)] = float(up)
            loads[("down", n)] = float(down)
        return loads

    def uplink_load_array(self, M: np.ndarray) -> np.ndarray:
        """Flat array of per-node uplink loads (the inter-node bottleneck metric)."""
        node = self.node_of
        ups = []
        for n in range(self.num_nodes):
            src_in = node == n
            ups.append(M[src_in, :][:, ~src_in].sum())
        return np.asarray(ups, dtype=np.float64)
