"""Experiment R — runtime reconfiguration of a frozen plan."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .admit import admit_by_slot
from .config import EXPERIMENTS
from .policy import collapse_history
from .stage2_blocks import block_matrix, step_blocks

MODELS: Dict[str, Dict] = {
    "flame-moe-290m": {"source": "flame", "window": 4, "n_tokens": 250_000,
                       "layers": ["layer_{:02d}".format(i) for i in range(2, 10)],
                       "ckpts": [540, 1080, 1620, 2160, 2700, 3240, 3780, 4320, 4860, 5400, 5473]},
    "flame-moe-721m": {"source": "flame", "window": 4, "n_tokens": 250_000,
                       "layers": ["layer_{:02d}".format(i) for i in range(2, 13)],
                       "ckpts": [880, 1760, 2640, 3520, 4400, 5280, 6160, 7040, 7920, 8800, 8815]},
    "flame-moe-1.7b": {"source": "flame", "window": 2, "n_tokens": 250_000,
                       "layers": ["layer_{:02d}".format(i) for i in range(2, 19)],
                       "ckpts": [1100, 2200, 3300, 4400, 5500, 6600, 7700, 8800, 9900, 11000, 11029]},
    "olmoe": {"source": "olmoe", "window": 2, "n_tokens": 205_000,
              "layers": ["layer_{}".format(i) for i in range(16)],
              "ckpts": ["5000", "120000", "245000", "490000", "final"]},
}

DOWN_IDLE = 0.10
OUTLIER_MULT = 3.0
EMA_ALPHA = 0.2
EMA_HEADROOM = 0.05
COSINE_GATE = 0.99


def _common():
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    from common.flame_loader import load_layer
    from common.olmoe_loader import load_checkpoint
    from common.placement import expert_to_device, token_source_ranks

    return load_layer, load_checkpoint, expert_to_device, token_source_ranks


def load_tokens(model: str, layer: str, ckpt) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(indices (n, k), scores or None), cache-first, no downloads."""
    load_layer, load_checkpoint, _, _ = _common()
    spec = MODELS[model]
    if spec["source"] == "flame":
        scores, indices = load_layer(int(ckpt), layer, spec["n_tokens"], model=model, verbose=False)
        return indices.astype(np.int64), scores.astype(np.float32)
    packed = load_checkpoint(str(ckpt), spec["n_tokens"], verbose=False)
    li = int(layer.split("_")[1])
    return packed[:, li, :].astype(np.int64), None


def maps(model: str, placement: str = "contiguous", seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    _, _, expert_to_device, token_source_ranks = _common()
    return (expert_to_device(64, 8, placement, seed=seed),
            token_source_ranks(MODELS[model]["n_tokens"], 8, "contiguous"))


def robust_collapse(mats: List[np.ndarray], policy: str = "p95") -> Tuple[np.ndarray, int]:
    """P95 over the samples that are not outliers (L1 distance to the elementwise"""
    if len(mats) < 3:
        return collapse_history(mats, policy), 0
    stack = np.stack(mats)
    med = np.median(stack, axis=0)
    dist = np.abs(stack - med).sum(axis=(1, 2))
    thr = OUTLIER_MULT * max(float(np.median(dist)), 1e-9)
    keep = [m for m, d in zip(mats, dist) if d <= thr]
    if len(keep) < 2:
        keep = mats
    return collapse_history(keep, policy), len(mats) - len(keep)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel(); b = b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


@dataclass
class StepObs:
    ckpt: str
    step: int
    overflow: float
    idle: float
    tokens_hit: float
    E_total: float
    fired: str = ""          # "", "up", "down", "periodic", "ema"


@dataclass
class PolicyRun:
    policy: str
    obs: List[StepObs] = field(default_factory=list)
    n_reconfig: int = 0
    E_history: List[Tuple[str, int, np.ndarray]] = field(default_factory=list)
    calibration: Optional[Dict] = None


def _score(M: np.ndarray, E: np.ndarray, indices_b, place, src_b) -> Tuple[float, float, float]:
    over = float(np.maximum(M - E, 0).sum() / max(M.sum(), 1))
    idle = float(np.maximum(E - M, 0).sum() / max(E.sum(), 1))
    adm = admit_by_slot(indices_b, place, src_b, E, order="arrival", overflow="drop")
    return over, idle, float(adm.stats["frac_tokens_hit"])


def run_layer(
    model: str,
    layer: str,
    *,
    B: int,
    factors: Sequence[float] = (1.0, 1.5, 2.0),
    W: int = 3,
    percentile_policy: str = "p95",
    window: Optional[int] = None,
    placement: str = "contiguous",
    seed: int = 0,
) -> Dict[str, PolicyRun]:
    """All policies on one layer's timeline. Returns per-policy runs."""
    spec = MODELS[model]
    N, K = spec["n_tokens"], (window or spec["window"])
    place, src_full = maps(model, placement, seed)
    ckpts = list(spec["ckpts"])
    toks = {c: load_tokens(model, layer, c)[0] for c in ckpts}
    full_mats = {c: block_matrix(toks[c], place, src_full, 8) for c in ckpts}
    blocks = step_blocks(N, B, 8)
    scale = float(B) / float(N)
    rng = np.random.default_rng(seed)
    step_order = {c: rng.permutation(len(blocks)) for c in ckpts}
    step_mats = {c: [block_matrix(toks[c][idx], place, srcb, 8) for idx, srcb in blocks] for c in ckpts}

    def collapse(mats: List[np.ndarray]) -> np.ndarray:
        return collapse_history(mats, percentile_policy)

    def walk(name: str, E0: np.ndarray, update=None) -> PolicyRun:
        """Visit every step; ``update(state, ci, c, b, M)`` may return a new E (250k scale) and a tag."""
        pr = PolicyRun(name)
        E = E0
        pr.E_history.append((str(ckpts[K - 1]), 0, E0))
        state: Dict = {"recent_o": [], "recent_i": [], "recent_M": [], "refractory": 0, "seen": [], "flagged": set()}
        for ci, c in enumerate(ckpts):
            state["seen"] = []
            for b in step_order[c]:
                idx, srcb = blocks[b]
                M = step_mats[c][b]
                fired = ""
                if update is not None and ci >= K:
                    newE, fired = update(state, ci, c, int(b), M, E)
                    if newE is not None:
                        E = newE
                        pr.n_reconfig += 1
                        pr.E_history.append((str(c), int(b), E))
                o, i, h = _score(M, E * scale, toks[c][idx], place, srcb)
                pr.obs.append(StepObs(str(c), int(b), o, i, h, float(E.sum()), fired))
                if ci >= K:
                    state["recent_o"].append(o); state["recent_i"].append(i); state["recent_M"].append(M)
                    if state["refractory"] > 0:
                        state["refractory"] -= 1
                state["seen"].append(M)
        return pr

    E_freeze = collapse([full_mats[c] for c in ckpts[:K]])
    runs: Dict[str, PolicyRun] = {}
    runs["freeze"] = walk("freeze", E_freeze)
    E_rob, _ = robust_collapse([full_mats[c] for c in ckpts[:K]], percentile_policy)
    runs["freeze_robust"] = walk("freeze_robust", E_rob)
    for m in (1.05, 1.10):
        runs["margin_{:.2f}".format(m)] = walk("margin_{:.2f}".format(m), E_freeze * m)
    mean_win = np.mean([full_mats[c] for c in ckpts[:K]], axis=0)
    off = ~np.eye(8, dtype=bool)
    E_uni = np.full((8, 8), 1.25 * mean_win[off].mean()); np.fill_diagonal(E_uni, mean_win.diagonal())
    runs["uniform_cf1.25"] = walk("uniform_cf1.25", E_uni)

    def make_periodic(expanding: bool):
        def upd(state, ci, c, b, M, E):
            if b == step_order[c][0]:
                hist = [full_mats[cc] for cc in (ckpts[:ci] if expanding else ckpts[ci - K:ci])]
                return collapse(hist), "periodic"
            return None, ""
        return upd
    runs["periodic_slide"] = walk("periodic_slide", E_freeze, make_periodic(False))
    runs["periodic_expand"] = walk("periodic_expand", E_freeze, make_periodic(True))

    def ema_upd(state, ci, c, b, M, E):
        if "ema" not in state:
            state["ema"] = E_freeze.copy()
            return None, ""
        state["ema"] = (1 - EMA_ALPHA) * state["ema"] + EMA_ALPHA * (M / scale)
        return state["ema"] * (1 + EMA_HEADROOM), "ema"
    runs["ema"] = walk("ema", E_freeze, ema_upd)

    pr = PolicyRun("oracle")
    for c in ckpts:
        E_step = collapse(step_mats[c])
        for b in step_order[c]:
            idx, srcb = blocks[b]
            o, i, h = _score(step_mats[c][b], E_step, toks[c][idx], place, srcb)
            pr.obs.append(StepObs(str(c), int(b), o, i, h, float(E_step.sum() / scale)))
    runs["oracle"] = pr

    loo_over: List[float] = []
    for c in ckpts[:K]:
        others = [full_mats[cc] for cc in ckpts[:K] if cc != c]
        E_loo = collapse(others) if others else full_mats[c]
        for M in step_mats[c]:
            loo_over.append(float(np.maximum(M - E_loo * scale, 0).sum() / max(M.sum(), 1)))
    loo_max = float(max(loo_over)) if loo_over else 0.0
    loo_med = float(np.median(loo_over)) if loo_over else 0.0

    def make_guard(kind: str, param: float):
        def upd(state, ci, c, b, M, E):
            fired = ""
            if state["refractory"] == 0 and len(state["recent_o"]) >= W:
                mo = float(np.mean(state["recent_o"][-W:]))
                mi = float(np.mean(state["recent_i"][-W:]))
                if kind == "loo" and mo > param * loo_max:
                    fired = "up"
                elif kind == "cap" and mo > param:
                    fired = "up"
                elif kind == "cosine" and _cosine(np.sum(state["recent_M"][-W:], axis=0), E) < param:
                    fired = "up"
                elif mi > DOWN_IDLE:
                    fired = "down"
            if not fired:
                return None, ""
            if fired == "up":
                state["flagged"].add(c)
            hist_ck = [cc for cc in ckpts[max(0, ci - (K - 1)):ci] if (fired == "up" or cc not in state["flagged"])]
            hist = [full_mats[cc] for cc in hist_ck]
            seen = list(state["seen"]) + [M]
            hist.append(collapse(seen) / scale)
            if fired == "down":
                newE, _ = robust_collapse(hist, percentile_policy) if len(hist) >= 3 else (collapse(hist), 0)
            else:
                newE = collapse(hist)
            state["refractory"] = W
            state["recent_o"], state["recent_i"], state["recent_M"] = [], [], []
            return newE, fired
        return upd

    for f in factors:
        name = "guard_loo_{:g}".format(f)
        runs[name] = walk(name, E_freeze, make_guard("loo", f))
        runs[name].calibration = {"loo_max_over": loo_max, "loo_median_over": loo_med, "thr_over": f * loo_max, "thr_idle": DOWN_IDLE}
    for cap in (0.02, 0.05):
        name = "cap_{:g}".format(cap * 100)
        runs[name] = walk(name, E_freeze, make_guard("cap", cap))
        runs[name].calibration = {"thr_over": cap, "thr_idle": DOWN_IDLE}
    runs["cosine_0.99"] = walk("cosine_0.99", E_freeze, make_guard("cosine", COSINE_GATE))
    runs["cosine_0.99"].calibration = {"gate": COSINE_GATE, "thr_idle": DOWN_IDLE}
    return runs


def summarize_runs(runs: Dict[str, PolicyRun], window_ckpts: Sequence) -> Dict[str, Dict]:
    """Post-window means per policy, reconfig counts, per-checkpoint series."""
    out: Dict[str, Dict] = {}
    wset = {str(c) for c in window_ckpts}
    for name, pr in runs.items():
        post = [o for o in pr.obs if o.ckpt not in wset]
        order = []
        for o in pr.obs:
            if o.ckpt not in order:
                order.append(o.ckpt)
        by_ckpt: Dict[str, Dict[str, float]] = {}
        for c in order:
            sel = [o for o in pr.obs if o.ckpt == c]
            by_ckpt[c] = {"overflow": float(np.mean([o.overflow for o in sel])), "idle": float(np.mean([o.idle for o in sel])),
                          "tokens_hit": float(np.mean([o.tokens_hit for o in sel])), "E_total": float(np.mean([o.E_total for o in sel])),
                          "fires": sum(1 for o in sel if o.fired), "fires_up": sum(1 for o in sel if o.fired == "up"),
                          "fires_down": sum(1 for o in sel if o.fired == "down")}
        post_ckpts = [c for c in order if c not in wset]
        out[name] = {
            "overflow": float(np.mean([o.overflow for o in post])) if post else 0.0,
            "idle": float(np.mean([o.idle for o in post])) if post else 0.0,
            "tokens_hit": float(np.mean([o.tokens_hit for o in post])) if post else 0.0,
            "overflow_max_step": float(max([o.overflow for o in post])) if post else 0.0,
            "n_reconfig": pr.n_reconfig,
            "fires_up": sum(1 for o in pr.obs if o.fired == "up"),
            "fires_down": sum(1 for o in pr.obs if o.fired == "down"),
            "ckpts_with_fire": sum(1 for c in post_ckpts if by_ckpt[c]["fires"] > 0),
            "n_post_ckpts": len(post_ckpts),
            "by_ckpt": by_ckpt,
            "calibration": pr.calibration,
        }
    return out
