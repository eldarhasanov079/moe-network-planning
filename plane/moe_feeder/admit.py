"""Admit live traffic into a frozen reservation."""

from __future__ import annotations

from typing import Dict

import numpy as np


def admit_slots(actual: np.ndarray, reservation: np.ndarray) -> np.ndarray:
    """Slots that fit under the reservation (overflow discarded, no deflection)."""
    return np.minimum(
        np.asarray(actual, dtype=np.float64),
        np.asarray(reservation, dtype=np.float64),
    )


def token_drop_stats(
    indices: np.ndarray,
    place: np.ndarray,
    src: np.ndarray,
    reservation: np.ndarray,
) -> Dict[str, float]:
    """Admit top-k dests in order. Overflowing slots are dropped (no deflection)."""
    dests = place[indices]
    n, k = dests.shape
    sent = np.zeros_like(reservation, dtype=np.float64)
    any_drop = 0
    full_drop = 0
    slots_drop = 0
    slots = 0
    devices = int(reservation.shape[0])
    for token in range(n):
        source = int(src[token])
        kept = 0
        lost = 0
        for expert in range(k):
            dest = int(dests[token, expert])
            slots += 1
            if (
                0 <= source < devices
                and 0 <= dest < devices
                and sent[source, dest] + 1.0 <= reservation[source, dest] + 1e-9
            ):
                sent[source, dest] += 1.0
                kept += 1
            else:
                slots_drop += 1
                lost += 1
        if lost:
            any_drop += 1
        if kept == 0:
            full_drop += 1
    return {
        "n_tokens": float(n),
        "frac_tokens_dropped": any_drop / n if n else 0.0,
        "frac_tokens_fully_dropped": full_drop / n if n else 0.0,
        "frac_slots_dropped": slots_drop / slots if slots else 0.0,
    }


from dataclasses import dataclass, field  # noqa: E402
from typing import Optional  # noqa: E402

SLOT_ORDERS = ("arrival", "score", "rank")
OVERFLOW_MODES = ("drop", "tail")


@dataclass
class SlotAdmission:
    """Per-slot outcome of admitting one step of top-k traffic into a reservation."""

    reserved: np.ndarray
    tail: np.ndarray
    dropped: np.ndarray
    sent_reserved: np.ndarray
    sent_tail: np.ndarray
    sent_dropped: np.ndarray
    stats: Dict[str, float] = field(default_factory=dict)


def _cell_rank(cell_sorted: np.ndarray) -> np.ndarray:
    """Position of each element within its (already sorted) cell group."""
    n = len(cell_sorted)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.r_[0, np.flatnonzero(np.diff(cell_sorted)) + 1]
    starts = np.zeros(n, dtype=np.int64)
    starts[first] = first
    starts = np.maximum.accumulate(starts)
    return np.arange(n, dtype=np.int64) - starts


def admit_by_slot(
    indices: np.ndarray,
    place: np.ndarray,
    src: np.ndarray,
    reservation: np.ndarray,
    *,
    scores: Optional[np.ndarray] = None,
    order: str = "arrival",
    overflow: str = "tail",
    weight_gate: Optional[float] = None,
    rank_cut: Optional[int] = None,
) -> SlotAdmission:
    """Admit every top-k slot of every token into ``reservation`` (membership lens)."""
    if order not in SLOT_ORDERS:
        raise ValueError("unknown order {!r}".format(order))
    if overflow not in OVERFLOW_MODES:
        raise ValueError("unknown overflow {!r}".format(overflow))
    if order == "score" and scores is None:
        raise ValueError("order='score' needs router scores")
    if weight_gate is not None and scores is None:
        raise ValueError("weight_gate needs router scores")

    indices = np.asarray(indices, dtype=np.int64)
    n, k = indices.shape
    D = int(reservation.shape[0])
    E = np.asarray(reservation, dtype=np.float64)
    src = np.asarray(src, dtype=np.int64)
    dest = np.asarray(place, dtype=np.int64)[indices]
    s_flat = np.repeat(src, k)
    d_flat = dest.ravel()
    valid = (s_flat >= 0) & (s_flat < D) & (d_flat >= 0) & (d_flat < D)
    cell = np.where(valid, s_flat * D + d_flat, D * D)  # invalid -> sentinel cell
    arrival = np.arange(n * k, dtype=np.int64)
    col = np.tile(np.arange(k, dtype=np.int64), n)

    if scores is not None:
        sc = np.asarray(scores, dtype=np.float64)
        tot = sc.sum(axis=1, keepdims=True)
        tot[tot <= 0] = 1.0
        w = (sc / tot).ravel()
    else:
        w = None

    if order == "arrival":
        perm = np.lexsort((arrival, cell))
    elif order == "score":
        perm = np.lexsort((arrival, -w, cell))
    else:
        perm = np.lexsort((arrival, col, cell))

    cell_sorted = cell[perm]
    rank = _cell_rank(cell_sorted)
    cap_flat = np.append(np.floor(E.ravel() + 1e-9), 0.0)  # sentinel cell has cap 0
    cap = cap_flat[cell_sorted]
    admitted_sorted = rank < cap
    reserved = np.zeros(n * k, dtype=bool)
    reserved[perm] = admitted_sorted
    reserved &= valid

    over = ~reserved
    if overflow == "drop":
        dropped = over.copy()
    else:
        dropped = over & ~valid
        if weight_gate is not None:
            dropped |= over & (w <= weight_gate)
        if rank_cut is not None:
            dropped |= over & (col >= rank_cut)
    tail = over & ~dropped

    def _matrix(mask: np.ndarray) -> np.ndarray:
        sel = mask & valid
        return np.bincount(cell[sel], minlength=D * D)[: D * D].reshape(D, D).astype(np.float64)

    sent_reserved = _matrix(reserved)
    sent_tail = _matrix(tail)
    sent_dropped = _matrix(dropped)

    res2 = reserved.reshape(n, k)
    over2 = over.reshape(n, k)
    tail2 = tail.reshape(n, k)
    drop2 = dropped.reshape(n, k)
    slots = float(n * k) if n else 1.0
    ntok = float(n) if n else 1.0
    stats: Dict[str, float] = {
        "n_tokens": float(n),
        "n_slots": float(n * k),
        "order": order,
        "overflow": overflow,
        "frac_slots_over": float(over.sum() / slots),
        "frac_slots_tail": float(tail.sum() / slots),
        "frac_slots_dropped": float(dropped.sum() / slots),
        "frac_tokens_hit": float(over2.any(axis=1).mean()) if n else 0.0,
        "frac_tokens_fully_unreserved": float((~res2.any(axis=1)).mean()) if n else 0.0,
        "frac_tokens_dropped": float(drop2.any(axis=1).mean()) if n else 0.0,
        "frac_tokens_fully_dropped": float((~(res2 | tail2).any(axis=1)).mean()) if n else 0.0,
        "rank_hist_over": np.bincount(col[over], minlength=k).astype(float).tolist(),
        "rank_hist_tail": np.bincount(col[tail], minlength=k).astype(float).tolist(),
        "rank_hist_dropped": np.bincount(col[dropped], minlength=k).astype(float).tolist(),
    }
    if w is not None:
        stats.update({
            "weight_mass_over_per_token": float(w[over].sum() / ntok),
            "weight_mass_tail_per_token": float(w[tail].sum() / ntok),
            "weight_mass_dropped_per_token": float(w[dropped].sum() / ntok),
            "mean_weight_over": float(w[over].mean()) if over.any() else 0.0,
            "mean_weight_all": float(w.mean()),
        })
    return SlotAdmission(
        reserved=res2, tail=tail2, dropped=drop2,
        sent_reserved=sent_reserved, sent_tail=sent_tail, sent_dropped=sent_dropped,
        stats=stats,
    )
