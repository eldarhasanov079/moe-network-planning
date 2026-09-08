"""Stage 2: per-step blocks and per-step reservations."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .policy import collapse_history


def step_blocks(n_tokens: int, B: int, num_ranks: int = 8) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Token index + source-rank arrays for each step of ``B`` tokens."""
    per = int(np.ceil(n_tokens / num_ranks))
    m = B // num_ranks
    if m <= 0:
        raise ValueError("B must be >= num_ranks")
    n_blocks = per // m
    out = []
    for b in range(n_blocks):
        idx = np.concatenate([
            np.arange(r * per + b * m, min(r * per + (b + 1) * m, n_tokens))
            for r in range(num_ranks)
        ])
        src = np.concatenate([
            np.full(min(r * per + (b + 1) * m, n_tokens) - (r * per + b * m), r, dtype=np.int64)
            for r in range(num_ranks)
        ])
        out.append((idx.astype(np.int64), src))
    return out


def offset_blocks(n_total: int, start: int, B: int, num_ranks: int = 8) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Different-token steps: consecutive runs of ``B`` tokens from ``start``."""
    m = B // num_ranks
    out = []
    pos = start
    while pos + m * num_ranks <= n_total:
        idx = np.arange(pos, pos + m * num_ranks, dtype=np.int64)
        src = np.repeat(np.arange(num_ranks, dtype=np.int64), m)
        out.append((idx, src))
        pos += m * num_ranks
    return out


def block_matrix(indices: np.ndarray, place: np.ndarray, src: np.ndarray, num_devices: int) -> np.ndarray:
    """Membership-lens dispatch matrix for one step (per-slot counting)."""
    k = indices.shape[1]
    dest = np.asarray(place, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    flat = np.repeat(np.asarray(src, dtype=np.int64), k) * num_devices + dest.ravel()
    return np.bincount(flat, minlength=num_devices * num_devices).reshape(num_devices, num_devices).astype(np.float64)


def scale_reservation(E: np.ndarray, B: int, n_tokens: int) -> np.ndarray:
    """Rate-sized reservation: the Stage 1 ``E`` scaled to a step of ``B`` tokens."""
    return np.asarray(E, dtype=np.float64) * (float(B) / float(n_tokens))


def step_reservation(
    window_indices: Sequence[np.ndarray],
    B: int,
    policy: str,
    place: np.ndarray,
    num_devices: int = 8,
) -> Tuple[np.ndarray, int]:
    """Per-step reservation: ``policy`` over every step of every window checkpoint."""
    mats = []
    for indices in window_indices:
        for idx, src in step_blocks(len(indices), B, num_devices):
            mats.append(block_matrix(indices[idx], place, src, num_devices))
    return collapse_history(mats, policy), len(mats)
