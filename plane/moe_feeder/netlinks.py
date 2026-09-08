"""Stage 2: link-level accounting on the 8-GPU Clos, no simulator."""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from puppeteer.config import RunConfig
from puppeteer.planner.router import ClosEcmpRouter, enumerate_clos_routes
from puppeteer.planner.state import NetworkState

from .config import EXPERIMENTS

Routes = Dict[Tuple[int, int], List[Tuple[List[str], float]]]


def _netmetrics():
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    from common import netmetrics

    return netmetrics


class LinkModel:
    def __init__(self, topology_path: str) -> None:
        self.config = RunConfig.load(topology_path)
        self.topology = self.config.topology
        self.links: List[str] = sorted(self.topology.links.keys())
        self.cap_bps: Dict[str, float] = {
            l: float(self.topology.links[l].capacity_bps) for l in self.links
        }
        self.n = int(self.topology.num_ranks) if hasattr(self.topology, "num_ranks") else 8

    # ------------------------------------------------------------ route maps

    def frozen_routes(self, paths: Dict[Tuple[int, int], object]) -> Routes:
        """``paths`` is a ``(src, dst) -> Route`` map (planner.paths_from_plan)."""
        out: Routes = {}
        for s in range(self.n):
            for d in range(self.n):
                if s == d:
                    continue
                r = paths.get((s, d))
                out[(s, d)] = [(list(r.links) if r is not None else [], 1.0)]
        return out

    def ecmp_routes(self, seed: int = 0) -> Routes:
        router = ClosEcmpRouter(self.topology, seed=seed)
        state = NetworkState(self.topology)
        out: Routes = {}
        for s in range(self.n):
            for d in range(self.n):
                if s == d:
                    continue
                out[(s, d)] = [(list(router.route(s, d, state).links), 1.0)]
        return out

    def spray_routes(self) -> Routes:
        out: Routes = {}
        for s in range(self.n):
            for d in range(self.n):
                if s == d:
                    continue
                rs = enumerate_clos_routes(self.topology, s, d)
                out[(s, d)] = [(list(r.links), 1.0 / len(rs)) for r in rs]
        return out

    # ------------------------------------------------------------ accounting

    def link_bytes(self, cells: np.ndarray, routes: Routes, bytes_per_slot: int) -> Dict[str, float]:
        load = {l: 0.0 for l in self.links}
        cells = np.asarray(cells, dtype=np.float64)
        for (s, d), opts in routes.items():
            v = float(cells[s, d])
            if v <= 0:
                continue
            for links, w in opts:
                for l in links:
                    load[l] += v * w * bytes_per_slot
        return load

    def lower_bound_s(self, load: Dict[str, float]) -> Tuple[float, Optional[str]]:
        """Busiest-link serialisation bound: max_l bytes_l * 8 / cap_l."""
        best, link = 0.0, None
        for l, b in load.items():
            t = b * 8.0 / self.cap_bps[l]
            if t > best:
                best, link = t, l
        return best, link

    def budget(
        self,
        M: np.ndarray,
        E: np.ndarray,
        routes_reserved: Routes,
        routes_tail: Routes,
        bytes_per_slot: int,
    ) -> Dict[str, object]:
        """Per-link base / refund / overflow / E and the derived coverage + bounds."""
        nm = _netmetrics()
        base, refund, overflow = nm.split_matrix(np.asarray(M, dtype=np.float64), np.asarray(E, dtype=np.float64))
        base_l = self.link_bytes(base, routes_reserved, bytes_per_slot)
        refund_l = self.link_bytes(refund, routes_reserved, bytes_per_slot)
        E_l = self.link_bytes(E, routes_reserved, bytes_per_slot)
        over_l = self.link_bytes(overflow, routes_tail, bytes_per_slot)
        tot_over = sum(over_l.values())
        covered = sum(min(over_l[l], refund_l[l]) for l in self.links)
        uncovered = [l for l in self.links if over_l[l] > refund_l[l] + 1e-9]
        lb_res, bind_res = self.lower_bound_s(base_l)
        total_l = {l: base_l[l] + over_l[l] for l in self.links}
        lb_total, bind_total = self.lower_bound_s(total_l)
        lb_E, _ = self.lower_bound_s(E_l)
        worst_ratio = 0.0
        for l in self.links:
            if over_l[l] > 0:
                worst_ratio = max(worst_ratio, over_l[l] / refund_l[l] if refund_l[l] > 0 else np.inf)
        return {
            "coverage": covered / tot_over if tot_over > 0 else 1.0,
            "n_links_uncovered": len(uncovered),
            "uncovered_links": uncovered,
            "worst_link_overflow_over_refund": float(worst_ratio),
            "overflow_bytes": tot_over,
            "refund_bytes": sum(refund_l.values()),
            "lb_reserved_s": lb_res,
            "lb_total_s": lb_total,
            "lb_E_s": lb_E,
            "predicted_extension_s": max(0.0, lb_total - lb_res),
            "fits_plan_budget": lb_total <= lb_E + 1e-12,
            "binding_link_reserved": bind_res,
            "binding_link_total": bind_total,
            "per_link": {
                l: {"base": base_l[l], "refund": refund_l[l], "overflow": over_l[l], "E": E_l[l]}
                for l in self.links
            },
        }
