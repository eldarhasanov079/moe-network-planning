"""Shared netmodel toolkit -- similarity, link-load, and envelope metrics."""

from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def volume_approx(M: np.ndarray) -> np.ndarray:
    """Rank-1 (independent source x destination) approximation of M:"""
    P = M.astype(np.float64)
    tot = P.sum()
    if tot <= 0:
        return P
    r = P.sum(axis=1, keepdims=True)
    c = P.sum(axis=0, keepdims=True)
    return (r @ c) / tot


def rank1_residual(M: np.ndarray) -> float:
    """Fraction of matrix 'energy' NOT explained by the rank-1 outer product of its"""
    P = M.astype(np.float64)
    norm = np.linalg.norm(P)
    if norm == 0:
        return 0.0
    return float(np.linalg.norm(P - volume_approx(P)) / norm)


def norm_l1(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of absolute differences."""
    return float(np.abs(a.ravel().astype(np.float64) - b.ravel().astype(np.float64)).sum())


def topk_flow_overlap(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """Fraction of the k heaviest flows (matrix entries) shared between A and B."""
    a_idx = set(np.argsort(A.ravel())[::-1][:k].tolist())
    b_idx = set(np.argsort(B.ravel())[::-1][:k].tolist())
    if k == 0:
        return 1.0
    return len(a_idx & b_idx) / k


def link_stats(loads: np.ndarray) -> dict:
    """max / mean / p95 / max-over-mean for a flat array of link loads."""
    loads = np.asarray(loads, dtype=np.float64)
    if loads.size == 0:
        return {"max": 0.0, "mean": 0.0, "p95": 0.0, "max_over_mean": 0.0}
    mean = loads.mean()
    return {
        "max": float(loads.max()),
        "mean": float(mean),
        "p95": float(np.percentile(loads, 95)),
        "max_over_mean": float(loads.max() / mean) if mean > 0 else 0.0,
    }


def hot_link_overlap(loads_a: np.ndarray, loads_b: np.ndarray, frac: float = 0.25) -> float:
    """Overlap of the hottest ``frac`` of links between two equally-ordered load"""
    n = len(loads_a)
    k = max(1, int(round(frac * n)))
    a_idx = set(np.argsort(loads_a)[::-1][:k].tolist())
    b_idx = set(np.argsort(loads_b)[::-1][:k].tolist())
    return len(a_idx & b_idx) / k


def build_envelope(history: list[np.ndarray], percentile: float) -> np.ndarray:
    """Per-entry percentile over a stack of profiling-window matrices."""
    stack = np.stack(history, axis=0).astype(np.float64)
    return np.percentile(stack, percentile, axis=0)


def split_matrix(M: np.ndarray, E: np.ndarray):
    """Return (base, refund, overflow) per project-context section 4.2."""
    M = M.astype(np.float64)
    E = E.astype(np.float64)
    base = np.minimum(M, E)
    refund = np.maximum(E - M, 0.0)
    overflow = np.maximum(M - E, 0.0)
    return base, refund, overflow


def envelope_metrics(M: np.ndarray, E: np.ndarray) -> dict:
    """Coverage / overflow / waste for applying envelope E to actual matrix M."""
    base, refund, overflow = split_matrix(M, E)
    m_tot = M.sum()
    e_tot = E.sum()
    return {
        "overflow_ratio": float(overflow.sum() / m_tot) if m_tot > 0 else 0.0,
        "coverage_ratio": float(base.sum() / m_tot) if m_tot > 0 else 1.0,
        "frac_entries_overflow": float(np.mean(M > E)),
        "waste_ratio": float(refund.sum() / e_tot) if e_tot > 0 else 0.0,
    }
