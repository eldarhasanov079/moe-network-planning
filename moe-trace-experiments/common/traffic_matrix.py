"""Shared netmodel toolkit -- build a source->destination All-to-All traffic matrix."""

from __future__ import annotations

import numpy as np


def build_dispatch_matrix(
    indices: np.ndarray,
    place: np.ndarray,
    src_rank: np.ndarray,
    num_src_ranks: int,
    num_devices: int,
    dedup_device: bool = False,
) -> np.ndarray:
    """Return the dispatch matrix ``M`` of shape (num_src_ranks, num_devices)."""
    n_tokens, k = indices.shape
    dest = place[indices.astype(np.int64)]            # (n_tokens, k)

    if dedup_device:
        hit = np.zeros((n_tokens, num_devices), dtype=np.int64)
        rows = np.repeat(np.arange(n_tokens), k)
        hit[rows, dest.ravel()] = 1
        M = np.zeros((num_src_ranks, num_devices), dtype=np.int64)
        np.add.at(M, src_rank, hit)
        return M

    src_flat = np.repeat(src_rank, k)                 # (n_tokens*k,)
    dst_flat = dest.ravel()                           # (n_tokens*k,)
    flat = src_flat * num_devices + dst_flat
    M = np.bincount(flat, minlength=num_src_ranks * num_devices)
    return M.reshape(num_src_ranks, num_devices).astype(np.int64)


def as_distribution(M: np.ndarray) -> np.ndarray:
    """Flatten ``M`` to a probability vector that sums to 1 (for cosine / L1)."""
    total = M.sum()
    if total == 0:
        return np.zeros(M.size, dtype=np.float64)
    return (M.astype(np.float64) / total).ravel()
