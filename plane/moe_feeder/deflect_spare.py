"""Experiment D — spare-candidate deflection of overflow slots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .admit import admit_by_slot

POLICIES = ("drop", "spare", "any", "spare_any")


@dataclass
class DeflectResult:
    policy: str
    sent: np.ndarray
    stats: Dict[str, float] = field(default_factory=dict)


def _weights(scores: Optional[np.ndarray], k_use: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Renormalised weights of the k-1 chosen slots and of the spare, over the chosen k-1."""
    if scores is None:
        return None, None
    sc = np.asarray(scores, dtype=np.float64)
    tot = sc[:, :k_use].sum(axis=1, keepdims=True)
    tot[tot <= 0] = 1.0
    return sc[:, :k_use] / tot, sc[:, k_use] / tot[:, 0]


def deflect_step(
    indices: np.ndarray,
    scores: Optional[np.ndarray],
    place: np.ndarray,
    src: np.ndarray,
    E: np.ndarray,
    policy: str = "spare",
    order: str = "arrival",
    k_use: Optional[int] = None,
    n_spares: int = 1,
) -> DeflectResult:
    """One step: admit the k - n_spares primary slots, then deflect overflow per ``policy``."""
    if policy not in POLICIES:
        raise ValueError(policy)
    indices = np.asarray(indices, dtype=np.int64)
    n, k = indices.shape
    k_use = (k - n_spares) if k_use is None else k_use
    n_spares = k - k_use
    D = int(E.shape[0])
    place = np.asarray(place, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    prim = indices[:, :k_use]
    spares = indices[:, k_use:]
    w, w_sp = _weights(scores, k_use)
    sc_prim = None if scores is None else np.asarray(scores)[:, :k_use]

    adm = admit_by_slot(prim, place, src, E, scores=sc_prim, order=("score" if (order == "score" and scores is not None) else order if order != "score" else "rank"), overflow="drop")
    sent = adm.sent_reserved.copy()
    over = adm.dropped                      # (n, k_use) overflow slots
    cap = np.floor(np.asarray(E, dtype=np.float64) + 1e-9)
    room = cap - sent                       # per cell, after primaries

    n_over = int(over.sum())
    tokens_hit = float(over.any(axis=1).mean()) if n else 0.0
    resolved_spare = np.zeros_like(over)
    resolved_any = np.zeros_like(over)
    spare_dev = place[spares]                  # (n, n_spares)
    cost_pref = 0.0
    cost_pess = 0.0
    cost_drop = 0.0
    spare_cell_full = 0

    if policy in ("spare", "spare_any") and n_over:
        tok, col = np.nonzero(over)
        if w is not None:
            key = -w[tok, col]
        else:
            key = col.astype(np.float64)    # lower column = higher rank
        order_idx = np.lexsort((tok, key))
        next_spare = np.zeros(n, dtype=np.int64)
        for i in order_idx:
            t, j = tok[i], col[i]
            placed = False
            while next_spare[t] < n_spares:
                s_, d_ = src[t], spare_dev[t, next_spare[t]]
                next_spare[t] += 1
                if room[s_, d_] >= 1.0:
                    room[s_, d_] -= 1.0
                    sent[s_, d_] += 1.0
                    resolved_spare[t, j] = True
                    placed = True
                    break
                spare_cell_full += 1
            if not placed:
                continue
    if policy in ("any", "spare_any") and n_over:
        rem = over & ~resolved_spare
        tok, col = np.nonzero(rem)
        if w is not None:
            key = -w[tok, col]
        else:
            key = col.astype(np.float64)
        order_idx = np.lexsort((tok, key))
        for i in order_idx:
            t, j = tok[i], col[i]
            s_ = src[t]
            d_ = int(np.argmax(room[s_]))
            if room[s_, d_] >= 1.0 and d_ != place[prim[t, j]]:
                room[s_, d_] -= 1.0
                sent[s_, d_] += 1.0
                resolved_any[t, j] = True
    dropped = over & ~resolved_spare & ~resolved_any
    if w is not None:
        gap = np.maximum(w - w_sp[:, None], 0.0)
        cost_pref = float(gap[resolved_spare].sum() + gap[resolved_any].sum() + w[dropped].sum())
        cost_pess = float(w[resolved_spare].sum() + w[resolved_any].sum() + w[dropped].sum())
        cost_drop = float(w[over].sum())
    ntok = float(n) if n else 1.0
    stats = {
        "n_tokens": float(n), "k_use": k_use, "n_spares": n_spares, "policy": policy, "order": order,
        "frac_slots_over": n_over / float(n * k_use) if n else 0.0,
        "frac_tokens_hit": tokens_hit,
        "n_over": float(n_over),
        "resolved_spare_frac": float(resolved_spare.sum() / n_over) if n_over else 0.0,
        "resolved_any_frac": float(resolved_any.sum() / n_over) if n_over else 0.0,
        "dropped_frac_of_over": float(dropped.sum() / n_over) if n_over else 0.0,
        "frac_slots_dropped": float(dropped.sum() / (n * k_use)) if n else 0.0,
        "frac_tokens_dropped": float(dropped.any(axis=1).mean()) if n else 0.0,
        "spare_cell_full_frac": float(spare_cell_full / n_over) if n_over else 0.0,
        "E_utilisation": float(sent.sum() / max(cap.sum(), 1.0)),
        "cost_pref_per_token": cost_pref / ntok,
        "cost_pess_per_token": cost_pess / ntok,
        "cost_drop_per_token": cost_drop / ntok,
        "rank_hist_dropped": np.bincount(np.nonzero(dropped)[1], minlength=k_use).tolist(),
    }
    return DeflectResult(policy=policy, sent=sent, stats=stats)


def deflect_step_causal(
    indices: np.ndarray,
    scores: Optional[np.ndarray],
    place: np.ndarray,
    src: np.ndarray,
    E: np.ndarray,
    policy: str = "spare",
) -> DeflectResult:
    """Strictly causal per-token walk (validation of the batch variant)."""
    indices = np.asarray(indices, dtype=np.int64)
    n, k = indices.shape
    k_use = k - 1
    D = int(E.shape[0])
    cap = np.floor(np.asarray(E, dtype=np.float64) + 1e-9)
    sent = np.zeros((D, D))
    w, w_sp = _weights(scores, k_use)
    n_over = 0
    res_sp = 0
    res_any = 0
    dropped = 0
    cost_pref = cost_pess = cost_drop = 0.0
    tokens_hit = 0
    tokens_drop = 0
    for t in range(n):
        s_ = int(src[t])
        used_spare = False
        hit = False
        drp = False
        for j in range(k_use):
            d = int(place[indices[t, j]])
            if sent[s_, d] + 1.0 <= cap[s_, d]:
                sent[s_, d] += 1.0
                continue
            n_over += 1
            hit = True
            if w is not None:
                cost_drop += w[t, j]
            placed = False
            if policy in ("spare", "spare_any") and not used_spare:
                d2 = int(place[indices[t, k_use]])
                if sent[s_, d2] + 1.0 <= cap[s_, d2]:
                    sent[s_, d2] += 1.0
                    used_spare = True
                    placed = True
                    res_sp += 1
            if not placed and policy in ("any", "spare_any"):
                room = cap[s_] - sent[s_]
                room[d] = -1
                d2 = int(np.argmax(room))
                if room[d2] >= 1.0:
                    sent[s_, d2] += 1.0
                    placed = True
                    res_any += 1
            if w is not None:
                if placed:
                    cost_pref += max(w[t, j] - w_sp[t], 0.0)
                    cost_pess += w[t, j]
                else:
                    cost_pref += w[t, j]
                    cost_pess += w[t, j]
            if not placed:
                dropped += 1
                drp = True
        tokens_hit += hit
        tokens_drop += drp
    ntok = float(n) if n else 1.0
    stats = {
        "n_tokens": float(n), "k_use": k_use, "policy": policy, "order": "causal",
        "frac_slots_over": n_over / float(n * k_use) if n else 0.0,
        "frac_tokens_hit": tokens_hit / ntok, "n_over": float(n_over),
        "resolved_spare_frac": res_sp / n_over if n_over else 0.0,
        "resolved_any_frac": res_any / n_over if n_over else 0.0,
        "dropped_frac_of_over": dropped / n_over if n_over else 0.0,
        "frac_slots_dropped": dropped / float(n * k_use) if n else 0.0,
        "frac_tokens_dropped": tokens_drop / ntok,
        "spare_cell_full_frac": float("nan"),
        "E_utilisation": float(sent.sum() / max(cap.sum(), 1.0)),
        "cost_pref_per_token": cost_pref / ntok, "cost_pess_per_token": cost_pess / ntok,
        "cost_drop_per_token": cost_drop / ntok,
        "rank_hist_dropped": [],
    }
    return DeflectResult(policy=policy, sent=sent, stats=stats)
