"""Suites for the two Stage 3 experiments."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .claim_v1 import BYTES, _write_csv, _write_json
from .config import DEFAULT_TOPOLOGY
from .deflect_spare import POLICIES, deflect_step, deflect_step_causal
from .planner import plan_matrix
from .policy import collapse_history
from .reconfig import MODELS, load_tokens, maps, run_layer, summarize_runs
from .stage2_blocks import block_matrix, scale_reservation, step_blocks

QUICK = os.environ.get("MOE_STAGE3_QUICK", "") not in ("", "0")
DEFLECT_MODELS = ("flame-moe-290m", "flame-moe-721m", "flame-moe-1.7b", "olmoe")
RECONFIG_MODELS = DEFLECT_MODELS
STEP_SIZES = {"flame": (250_000, 125_000, 32_768), "olmoe": (205_000, 102_400, 32_768)}
HEADLINE = {"flame": 125_000, "olmoe": 102_400}
FACTORS = (1.0, 1.5, 2.0)
GUARD_W = 3
WINDOWS = {"flame-moe-290m": (4,), "flame-moe-721m": (4,), "flame-moe-1.7b": (2, 4), "olmoe": (2,)}
INSTALL_S = {"quota_only": 1e-5, "qp_rate_10ms": 0.01, "switch_path_100ms": 0.1}
PLACEMENTS = (("contiguous", 0), ("roundrobin", 0), ("random", 0), ("random", 1))
N_SPARES = (1, 2)
FWD_PER_LAYER_S = {"flame": 205.6e-3 / 8, "olmoe": 576.1e-3 / 16}
ITER_MULT = 3.0
GPUS_PER_HOST = 2


def _layers(model: str) -> List[str]:
    spec = MODELS[model]
    return spec["layers"][:1] if QUICK else spec["layers"]


# ---------------------------------------------------------------------- Experiment D


def _deflect_layer(args) -> List[Dict]:
    model, layer = args
    spec = MODELS[model]
    N, K = spec["n_tokens"], spec["window"]
    ckpts = list(spec["ckpts"])
    toks = {c: load_tokens(model, layer, c) for c in ckpts}
    k = toks[ckpts[0]][0].shape[1]
    held = ckpts[K:]
    if QUICK:
        held = held[-1:]
    rows: List[Dict] = []
    orders = ("arrival", "score") if spec["source"] == "flame" else ("arrival", "rank")
    head = HEADLINE[spec["source"]]

    def emit(base: Dict, r) -> None:
        row = dict(base)
        row.update({kk: (json.dumps(v) if isinstance(v, list) else v) for kk, v in r.stats.items()})
        rows.append(row)

    for placement, seed in PLACEMENTS:
        place, src_full = maps(model, placement, seed)
        pl = "{}{}".format(placement, seed if placement == "random" else "")
        primary = placement == "contiguous"
        for n_sp in (N_SPARES if primary else (1,)):
            k_use = k - n_sp
            E = collapse_history([block_matrix(toks[c][0][:, :k_use], place, src_full, 8) for c in ckpts[:K]], "p95")
            E_full = collapse_history([block_matrix(toks[c][0], place, src_full, 8) for c in ckpts[:K]], "p95") if n_sp == 1 else None
            sizes = STEP_SIZES[spec["source"]] if (primary and n_sp == 1) else (head,)
            for ckpt in held:
                ind, sc = toks[ckpt]
                for B in sizes:
                    blocks = step_blocks(N, B, 8)
                    if QUICK:
                        blocks = blocks[:1]
                    Eb = scale_reservation(E, B, N)
                    for b, (idx, srcb) in enumerate(blocks):
                        ind_b = ind[idx]
                        sc_b = sc[idx] if sc is not None else None
                        base = dict(model=model, layer=layer, checkpoint=str(ckpt), B=B, block=b, placement=pl, n_spares=n_sp)
                        for order in (orders if (primary and n_sp == 1) else ("arrival",)):
                            for pol in POLICIES:
                                emit(dict(base, order=order, policy=pol),
                                     deflect_step(ind_b, sc_b, place, srcb, Eb, policy=pol, order=order, n_spares=n_sp))
                        if B == head and n_sp == 1:
                            for pol in ("spare", "spare_any"):
                                emit(dict(base, order="causal", policy=pol), deflect_step_causal(ind_b, sc_b, place, srcb, Eb, policy=pol))
                            if E_full is not None:
                                from .admit import admit_by_slot

                                adm = admit_by_slot(ind_b, place, srcb, scale_reservation(E_full, B, N), order="arrival", overflow="drop")
                                rows.append(dict(base, order="arrival", policy="control_topk_drop", k_use=k, n_spares=0,
                                                 frac_slots_over=adm.stats["frac_slots_over"], frac_tokens_hit=adm.stats["frac_tokens_hit"],
                                                 frac_slots_dropped=adm.stats["frac_slots_dropped"], frac_tokens_dropped=adm.stats["frac_tokens_hit"]))
    return rows


def _deflect_puppeteer(model: str, layers: Sequence[str]) -> List[Dict]:
    """Price the deflected reserved matrix in Puppeteer."""
    from .planner import freeze_paths, plan_two_class_step
    from .stage2 import KERNEL_FLOOR_S, PRIMARY_ALLOCATOR, WIRE_S_PER_SLOT, compute_anchor

    spec = MODELS[model]
    N, K = spec["n_tokens"], spec["window"]
    B = HEADLINE[spec["source"]]
    c = compute_anchor(spec["source"])["ratio"] * WIRE_S_PER_SLOT
    place, src_full = maps(model)
    ckpts = list(spec["ckpts"])
    rows = []
    for layer in layers:
        toks = {c_: load_tokens(model, layer, c_) for c_ in ckpts[:K] + [ckpts[-1]]}
        k = toks[ckpts[0]][0].shape[1]
        E = collapse_history([block_matrix(toks[c_][0][:, :k - 1], place, src_full, 8) for c_ in ckpts[:K]], "p95")
        Eb = scale_reservation(E, B, N)
        paths = freeze_paths(E, str(DEFAULT_TOPOLOGY))
        ind, sc = toks[ckpts[-1]]
        idx, srcb = step_blocks(N, B, 8)[0]
        for pol in ("drop", "spare", "spare_any"):
            r = deflect_step(ind[idx], sc[idx] if sc is not None else None, place, srcb, Eb, policy=pol, order="arrival")
            assert np.all(r.sent <= np.floor(Eb + 1e-9) + 1e-9)
            base = dict(model=model, layer=layer, policy=pol, sent_slots=float(r.sent.sum()))
            step = plan_two_class_step(r.sent, np.zeros_like(r.sent), str(DEFAULT_TOPOLOGY), frozen_paths=paths,
                                       compute_s_per_slot=c, structure="chunked", bytes_per_slot=BYTES,
                                       deadline_deltas=(), reserved_only=True, allocator=PRIMARY_ALLOCATOR,
                                       semantics="nccl", compute_floor_s=KERNEL_FLOOR_S)
            rows.append(dict(base, arm="full_step", allocator=PRIMARY_ALLOCATOR,
                             iteration_time_s=float(step["step_makespan_s"])))
            p_ = plan_matrix(r.sent, str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES, route_slots=E)
            q_ = plan_matrix(r.sent.T.copy(), str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES, route_slots=E.T.copy())
            rows.append(dict(base, arm="sizes_only", allocator="planner-default",
                             iteration_time_s=float(p_["iteration_time_s"] + q_["iteration_time_s"])))
    return rows


def run_deflect(out: Path, models: Sequence[str] = DEFLECT_MODELS) -> Dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(m, l) for m in models for l in _layers(m)]
    workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    print("  deflect layers", len(jobs), "workers", workers, flush=True)
    rows: List[Dict] = []
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_deflect_layer, jobs):
                rows += res
    else:
        for j in jobs:
            rows += _deflect_layer(j)
    _write_csv(out / "deflect.csv", rows)
    pup = []
    for m in models:
        if MODELS[m]["source"] == "flame" and m == "flame-moe-290m":
            pup += _deflect_puppeteer(m, _layers(m))
    _write_csv(out / "deflect_puppeteer.csv", pup)
    summary = summarize_deflect(rows, pup)
    _write_json(out / "deflect_summary.json", summary)
    return summary


def _mean(vals) -> float:
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _dist(vals) -> Dict[str, float]:
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    if not v:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v)), "mean": float(np.mean(v))}


def summarize_deflect(rows: List[Dict], pup: List[Dict]) -> Dict:
    S: Dict = {}
    keys = ("frac_slots_over", "frac_tokens_hit", "resolved_spare_frac", "resolved_any_frac", "dropped_frac_of_over",
            "frac_slots_dropped", "frac_tokens_dropped", "spare_cell_full_frac", "E_utilisation",
            "cost_pref_per_token", "cost_pess_per_token", "cost_drop_per_token")
    for model in sorted({r["model"] for r in rows}):
        M: Dict = {}
        for pl in sorted({r["placement"] for r in rows if r["model"] == model}):
            for n_sp in sorted({int(r["n_spares"]) for r in rows if r["model"] == model and r["placement"] == pl}):
                for B in sorted({int(r["B"]) for r in rows if r["model"] == model and r["placement"] == pl and int(r["n_spares"]) == n_sp}):
                    block: Dict = {}
                    for order in sorted({r["order"] for r in rows if r["model"] == model}):
                        for pol in POLICIES + ("control_topk_drop",):
                            sel = [r for r in rows if r["model"] == model and r["placement"] == pl and int(r["n_spares"]) == n_sp
                                   and int(r["B"]) == B and r["order"] == order and r["policy"] == pol]
                            if not sel:
                                continue
                            ent = {k: _mean(r.get(k) for r in sel) for k in keys if any(r.get(k) not in (None, "") for r in sel)}
                            ent["n"] = len(sel)
                            per_cell = []
                            for layer in sorted({r["layer"] for r in sel}):
                                for ck in sorted({r["checkpoint"] for r in sel if r["layer"] == layer}):
                                    ls = [r for r in sel if r["layer"] == layer and r["checkpoint"] == ck]
                                    if ls[0].get("resolved_spare_frac") not in (None, ""):
                                        per_cell.append(_mean(float(r["resolved_spare_frac"]) + float(r["resolved_any_frac"]) for r in ls))
                            if per_cell:
                                ent["resolved_dist"] = _dist(per_cell)
                            per_layer = {}
                            for layer in sorted({r["layer"] for r in sel}):
                                ls = [r for r in sel if r["layer"] == layer]
                                if ls[0].get("resolved_spare_frac") not in (None, ""):
                                    per_layer[layer] = _mean(float(r["resolved_spare_frac"]) + float(r["resolved_any_frac"]) for r in ls)
                            ent["resolved_per_layer"] = per_layer
                            block["{}_{}".format(order, pol)] = ent
                    M["{}|spares{}|B{}".format(pl, n_sp, B)] = block
        S[model] = M
    S["puppeteer"] = {}
    for model in sorted({r["model"] for r in pup}):
        per_arm = {}
        for arm in sorted({r.get("arm", "sizes_only") for r in pup if r["model"] == model}):
            sel = [r for r in pup if r["model"] == model and r.get("arm", "sizes_only") == arm]
            pols = sorted({r["policy"] for r in sel})
            d = {p: sum(float(r["iteration_time_s"]) for r in sel if r["policy"] == p) for p in pols}
            sl = {p: sum(float(r["sent_slots"]) for r in sel if r["policy"] == p) for p in pols}
            per_arm[arm] = {"allocator": sel[0].get("allocator", "planner-default"), "iteration_s": d, "sent_slots": sl}
            for p in pols:
                if p == "drop" or not d.get("drop"):
                    continue
                per_arm[arm]["rel_time_" + p] = d[p] / d["drop"] - 1
                per_arm[arm]["rel_slots_" + p] = sl[p] / sl["drop"] - 1 if sl["drop"] else 0.0
        S["puppeteer"][model] = per_arm
    return S


# ---------------------------------------------------------------------- Experiment R


def _reconfig_layer(args) -> Tuple[str, str, int, int, Dict]:
    model, layer, B, window = args
    runs = run_layer(model, layer, B=B, factors=FACTORS, W=GUARD_W, window=window)
    spec = MODELS[model]
    summ = summarize_runs(runs, spec["ckpts"][:window])
    hist = {name: [(c, b, float(E.sum())) for (c, b, E) in pr.E_history] for name, pr in runs.items()}
    mats = {}
    if model == "flame-moe-1.7b":
        for name in ("freeze", "freeze_robust", "guard_loo_1", "cosine_0.99", "periodic_slide"):
            mats[name] = [(c, b, E.tolist()) for (c, b, E) in runs[name].E_history]
        place, src_full = maps(model)
        mats["_M"] = {str(c): block_matrix(load_tokens(model, layer, c)[0], place, src_full, 8).tolist() for c in (2200, 3300, 4400)}
    return model, layer, B, window, {"summary": summ, "E_history": hist, "E_mats": mats}


def _measure_plan_time(model: str) -> Dict[str, float]:
    """Wall time of a Puppeteer re-plan (sizes/rates on frozen paths), whole model."""
    spec = MODELS[model]
    place, src_full = maps(model)
    layer = spec["layers"][0]
    ind, _ = load_tokens(model, layer, spec["ckpts"][0])
    E = block_matrix(ind, place, src_full, 8)
    from .planner import freeze_paths

    paths = freeze_paths(E, str(DEFAULT_TOPOLOGY))
    t = time.time()
    for _ in range(5):
        plan_matrix(E, str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES, frozen_paths=paths)
    per_dir = (time.time() - t) / 5
    return {"plan_s_per_layer_direction": per_dir, "plan_s_whole_model": per_dir * 2 * len(spec["layers"]),
            "envelope_s_whole_model": 1e-3}


def _price_collapse(E_mats: Dict[str, List], model: str) -> Dict:
    """Collapse case study for one 1.7B layer: timetable of each installed E, the"""
    from .planner import WIRE_S_PER_SLOT, freeze_paths, plan_two_class_step

    out: Dict = {"installed": {}}
    for name, hist in E_mats.items():
        if name == "_M":
            continue
        out["installed"][name] = []
        for (c, b, Elist) in hist:
            E = np.asarray(Elist, dtype=np.float64)
            p = plan_matrix(E, str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES)
            q = plan_matrix(E.T.copy(), str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES)
            out["installed"][name].append({"ckpt": c, "step": b, "E_total": float(E.sum()),
                                           "timetable_s": float(p["iteration_time_s"] + q["iteration_time_s"])})
    Ms = {c: np.asarray(m, dtype=np.float64) for c, m in E_mats.get("_M", {}).items()}
    if Ms:
        E_clean = np.asarray(E_mats["freeze"][0][2], dtype=np.float64)
        rank = np.arange(8)
        off = rank[:, None] // GPUS_PER_HOST != rank[None, :] // GPUS_PER_HOST
        case = {}
        for c, M in Ms.items():
            over = float(np.maximum(M - E_clean, 0).sum() / M.sum())
            hot_in = float((M * off).sum(axis=0).max())
            hot_out = float((M * off).sum(axis=1).max())
            bound = (hot_in + hot_out) * BYTES * 8.0 / 100e9
            hot_share = float((M * off).sum(axis=0).max() / (M * off).sum())   # of inter-host bytes
            case[c] = {"overflow_under_clean_E": over, "hot_device_share_of_bytes": hot_share, "downlink_bound_s": bound}
            r = 0.132 * WIRE_S_PER_SLOT
            paths = freeze_paths(E_clean, str(DEFAULT_TOPOLOGY))
            R = np.minimum(M, np.floor(E_clean + 1e-9)); T = M - R
            a = plan_two_class_step(R, T, str(DEFAULT_TOPOLOGY), frozen_paths=paths, compute_s_per_slot=r, structure="chunked",
                                    allocator="max-min", compute_floor_s=30e-6, deadline_deltas=())
            bE = plan_two_class_step(M, np.zeros_like(M), str(DEFAULT_TOPOLOGY), frozen_paths=freeze_paths(M, str(DEFAULT_TOPOLOGY)),
                                     compute_s_per_slot=r, structure="chunked", allocator="max-min", compute_floor_s=30e-6,
                                     deadline_deltas=(), reserved_only=True)
            case[c].update({"step_clean_E_plus_tail_s": float(a["step_makespan_s"]), "step_E_accommodates_s": float(bE["step_makespan_s"]),
                            "tail_share": float(T.sum() / M.sum())})
        out["case"] = case
    return out


def run_reconfig(out: Path, models: Sequence[str] = RECONFIG_MODELS, step_sizes: Sequence[int] = (32_768,)) -> Dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(m, l, B, w) for m in models for w in WINDOWS[m] for l in _layers(m) for B in step_sizes]
    workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    print("  reconfig layer runs", len(jobs), "workers", workers, flush=True)
    results = []
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_reconfig_layer, jobs):
                results.append(res)
                print("  reconfig done", res[0], res[1], "window", res[3], flush=True)
    else:
        results = [_reconfig_layer(j) for j in jobs]
    rows: List[Dict] = []
    per_model: Dict = {}
    collapse_mats: Dict[str, Dict] = {}
    for model, layer, B, window, res in results:
        key = "{}|K{}".format(model, window)
        for pol, s in res["summary"].items():
            rows.append(dict(model=model, window=window, layer=layer, B=B, policy=pol, overflow=s["overflow"], idle=s["idle"],
                             tokens_hit=s["tokens_hit"], overflow_max_step=s["overflow_max_step"],
                             n_reconfig=s["n_reconfig"], fires_up=s["fires_up"], fires_down=s["fires_down"],
                             by_ckpt=json.dumps(s["by_ckpt"]), calibration=json.dumps(s["calibration"])))
        per_model.setdefault(key, {}).setdefault(str(B), {})[layer] = res["summary"]
        if res["E_mats"]:
            collapse_mats.setdefault(key, {})[layer] = res["E_mats"]
    _write_csv(out / "reconfig.csv", rows)
    summary: Dict = {"models": {}, "cost": {}, "collapse": {}}
    for key, per_B in per_model.items():
        model = key.split("|")[0]
        spec = MODELS[model]
        summary["models"][key] = {}
        for B, per_layer in per_B.items():
            agg: Dict = {}
            policies = sorted({p for s in per_layer.values() for p in s})
            for pol in policies:
                ent = {k: _mean(s[pol][k] for s in per_layer.values()) for k in ("overflow", "idle", "tokens_hit", "overflow_max_step")}
                ent["overflow_dist"] = _dist(s[pol]["overflow"] for s in per_layer.values())
                ent["idle_dist"] = _dist(s[pol]["idle"] for s in per_layer.values())
                ent["n_reconfig_total"] = int(sum(s[pol]["n_reconfig"] for s in per_layer.values()))
                ent["n_reconfig_per_layer"] = _mean(s[pol]["n_reconfig"] for s in per_layer.values())
                ent["ckpts_with_fire_dist"] = _dist(s[pol]["ckpts_with_fire"] for s in per_layer.values())
                ent["n_post_ckpts"] = int(next(iter(per_layer.values()))[pol]["n_post_ckpts"])
                ent["fires_up"] = int(sum(s[pol]["fires_up"] for s in per_layer.values()))
                ent["fires_down"] = int(sum(s[pol]["fires_down"] for s in per_layer.values()))
                ck = {}
                for c in spec["ckpts"]:
                    c = str(c)
                    vals = [s[pol]["by_ckpt"].get(c) for s in per_layer.values() if s[pol]["by_ckpt"].get(c)]
                    if vals:
                        ck[c] = {k: _mean(v[k] for v in vals) for k in ("overflow", "idle", "tokens_hit", "E_total")}
                        ck[c]["fires"] = int(sum(v["fires"] for v in vals))
                ent["by_ckpt"] = ck
                agg[pol] = ent
            summary["models"][key][B] = agg
        pt = _measure_plan_time(model)
        n_layers = len(spec["layers"])
        fwd = FWD_PER_LAYER_S[spec["source"]] * n_layers
        iter_s = ITER_MULT * fwd
        n_iters = int(spec["ckpts"][-1]) if spec["source"] == "flame" else 1_200_000
        cost = dict(pt, forward_chain_s=fwd, iteration_s_estimate=iter_s, n_iterations=n_iters, install_s=dict(INSTALL_S))
        for B, agg in summary["models"][key].items():
            for pol, ent in agg.items():
                n = max([s[pol]["n_reconfig"] for s in per_B[B].values()] or [0])
                ent["reconfig_events_model"] = int(n)
                ent["overhead"] = {}
                for name, inst in INSTALL_S.items():
                    pipe = n * inst
                    bound = n * (pt["plan_s_whole_model"] + inst)     # stop-the-world upper bound
                    ent["overhead"][name] = {"pipelined_s": pipe, "stop_the_world_bound_s": bound,
                                             "pipelined_frac_of_training": pipe / max(n_iters * iter_s, 1e-9),
                                             "bound_frac_of_training": bound / max(n_iters * iter_s, 1e-9)}
        summary["cost"][key] = cost
    if collapse_mats:
        priced = {}
        for key, per_layer in collapse_mats.items():
            priced[key] = {}
            for layer, mats in list(per_layer.items())[: (1 if QUICK else 6)]:
                priced[key][layer] = _price_collapse(mats, "flame-moe-1.7b")
        summary["collapse"] = priced
    _write_json(out / "reconfig_summary.json", summary)
    return summary
