"""Shared netmodel toolkit -- flow-level All-to-All network simulator."""

from __future__ import annotations

import numpy as np


def _offdiag_indices(D: int):
    s, d = np.meshgrid(np.arange(D), np.arange(D), indexing="ij")
    mask = s != d
    return s[mask], d[mask]


def device_io_totals(M: np.ndarray):
    """Return (egress_total, ingress_total) per device, excluding the diagonal."""
    D = M.shape[0]
    diag = np.diag(M)
    egress = M.sum(axis=1) - diag
    ingress = M.sum(axis=0) - diag
    return egress.astype(np.float64), ingress.astype(np.float64)


def ideal_maxload(M: np.ndarray, R: int) -> float:
    """Lower bound on max link load: busiest device's I/O split evenly over R rails."""
    egress, ingress = device_io_totals(M)
    busiest = max(egress.max(initial=0.0), ingress.max(initial=0.0))
    return busiest / R


def maxload_under(A: np.ndarray, M: np.ndarray, R: int) -> float:
    """Max link load when assignment ``A`` carries the actual traffic ``M``."""
    D = M.shape[0]
    ss, dd = _offdiag_indices(D)
    rr = A[ss, dd]
    w = M[ss, dd].astype(np.float64)
    egress = np.zeros((D, R)); ingress = np.zeros((D, R))
    np.add.at(egress, (ss, rr), w)
    np.add.at(ingress, (dd, rr), w)
    return float(max(egress.max(initial=0.0), ingress.max(initial=0.0)))


def ecmp_assign(D: int, R: int, seed: int) -> np.ndarray:
    """Uniform-random per-flow rail hash (oblivious ECMP). Diagonal set to -1."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, R, size=(D, D)).astype(np.int64)
    np.fill_diagonal(A, -1)
    return A


def greedy_assign(M_decision: np.ndarray, R: int) -> np.ndarray:
    """Balanced rail assignment via greedy list-scheduling on ``M_decision``."""
    D = M_decision.shape[0]
    A = -np.ones((D, D), dtype=np.int64)
    egress = np.zeros((D, R)); ingress = np.zeros((D, R))
    ss, dd = _offdiag_indices(D)
    w = M_decision[ss, dd].astype(np.float64)
    order = np.argsort(-w, kind="stable")
    for idx in order:
        s = int(ss[idx]); d = int(dd[idx]); wt = float(w[idx])
        cand = np.maximum(egress[s] + wt, ingress[d] + wt)
        r = int(np.argmin(cand))
        A[s, d] = r
        egress[s, r] += wt; ingress[d, r] += wt
    return A


def normalized_time(A: np.ndarray, M_actual: np.ndarray, R: int) -> float:
    """max_link_load(A, M_actual) / ideal(M_actual). >= 1; 1 = perfectly balanced."""
    ideal = ideal_maxload(M_actual, R)
    if ideal <= 0:
        return 1.0
    return maxload_under(A, M_actual, R) / ideal


def ecmp_normalized_time(M_actual: np.ndarray, R: int, seeds: int):
    """Expected (mean) and P95 normalized time for oblivious ECMP over ``seeds`` hashes."""
    D = M_actual.shape[0]
    vals = [normalized_time(ecmp_assign(D, R, s), M_actual, R) for s in range(seeds)]
    return float(np.mean(vals)), float(np.percentile(vals, 95))
