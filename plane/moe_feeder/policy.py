"""Collapse a profile history into one reservation matrix."""

from __future__ import annotations

import sys
from typing import List

import numpy as np

from .config import EXPERIMENTS


def _netmetrics():
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    from common import netmetrics

    return netmetrics


def collapse_history(history: List[np.ndarray], reservation: str) -> np.ndarray:
    if not history:
        raise ValueError("empty profile history")
    nm = _netmetrics()
    stack = np.stack(history, axis=0).astype(np.float64)
    if reservation == "mean":
        return stack.mean(axis=0)
    if reservation == "p95":
        return nm.build_envelope(history, 95)
    if reservation == "p99":
        return nm.build_envelope(history, 99)
    if reservation == "worst":
        return stack.max(axis=0)
    if reservation == "volume":
        return nm.volume_approx(stack.mean(axis=0))
    if reservation == "uniform":
        mean = stack.mean(axis=0)
        n = mean.shape[0]
        total = mean.sum()
        off = total / max(n * (n - 1), 1)
        out = np.full_like(mean, off)
        np.fill_diagonal(out, 0.0)
        return out
    raise ValueError("unknown reservation {!r}".format(reservation))


def score_heldout(actual: np.ndarray, reservation: np.ndarray) -> dict:
    return _netmetrics().envelope_metrics(actual, reservation)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return _netmetrics().cosine(a, b)


def rank1_residual(M: np.ndarray) -> float:
    return _netmetrics().rank1_residual(M)


def slots_to_bytes(M: np.ndarray, bytes_per_slot: int) -> np.ndarray:
    return np.rint(np.asarray(M, dtype=np.float64) * bytes_per_slot).astype(np.int64)
