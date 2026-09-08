"""Stage 2 — runtime handling of a frozen plan's leftovers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .admit import admit_by_slot
from .astrasim import run_phased_jobs, write_switch_yaml
from .claim_v1 import (
    BYTES,
    SWITCH_BW,
    SWITCH_LAT,
    MatrixBank,
    _ckpt_key,
    _write_csv,
    _write_json,
    source_bundle,
)
from .config import DEFAULT_TOPOLOGY, EXPERIMENTS
from .netlinks import LinkModel
from .planner import paths_from_plan, plan_matrix
from .policy import collapse_history, score_heldout
from .stage2_blocks import block_matrix, offset_blocks, scale_reservation, step_blocks, step_reservation

POLICIES = ("mean", "p95", "worst")
HEADLINE_B = {"flame": 125_000, "olmoe": 102_400}
BLOCK_SIZES = {
    "flame": (8192, 32768, 65536, 125_000, 250_000),
    "olmoe": (8192, 32768, 65536, 102_400, 205_000),
}
OLMOE_PREFIX = 205_000
OLMOE_DIFF_B = 131_072
TAIL_SEEDS = 4
WEIGHT_GATES = (0.05, 0.10, 0.15)
ALL_STAGES = ("granularity", "shedding", "links", "astrasim", "clos", "tightness", "step", "ns3")
QUICK = os.environ.get("MOE_STAGE2_QUICK", "") not in ("", "0")


# --------------------------------------------------------------------------- data


class Stage2Data:
    """Traces, frozen reservations and per-layer bookkeeping for both models."""

    def __init__(self, sources: Sequence[str]) -> None:
        self.bank = MatrixBank()
        self.bundles = [source_bundle(s) for s in sources]
        for bundle in self.bundles:
            self.bank.load(bundle, keep_tokens=True, keep_scores=True)
            if QUICK:
                bundle["layers"] = bundle["layers"][:1]
        self.step_E: Dict[Tuple, Tuple[np.ndarray, int]] = {}

    def layers(self, bundle: Dict) -> List[str]:
        return list(bundle["layers"])

    def window(self, bundle: Dict) -> List:
        return list(bundle["ckpts"][: bundle["default_window"]])

    def held(self, bundle: Dict) -> List:
        held = list(bundle["ckpts"][bundle["default_window"]:])
        return held[-1:] if QUICK else held

    def tokens(self, bundle: Dict, layer: str, ckpt) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        key = (bundle["name"], layer, _ckpt_key(ckpt))
        indices, place, src = self.bank.tokens[key]
        return indices, self.bank.scores.get(key), place, src

    def reservations(self, bundle: Dict, layer: str) -> Dict[str, np.ndarray]:
        hist = self.bank.history(bundle, layer)[: bundle["default_window"]]
        return {p: collapse_history(hist, p) for p in POLICIES}

    def step_reservation(self, bundle: Dict, layer: str, B: int, policy: str) -> Tuple[np.ndarray, int]:
        key = (bundle["name"], layer, B, policy)
        if key not in self.step_E:
            window = [self.tokens(bundle, layer, c)[0] for c in self.window(bundle)]
            place = self.tokens(bundle, layer, self.window(bundle)[0])[2]
            self.step_E[key] = step_reservation(window, B, policy, place, 8)
        return self.step_E[key]

    def block_sizes(self, bundle: Dict) -> Tuple[int, ...]:
        sizes = BLOCK_SIZES[bundle["name"]]
        return (sizes[1], sizes[-1]) if QUICK else sizes

    def headline_B(self, bundle: Dict) -> int:
        return HEADLINE_B[bundle["name"]]

    def olmoe_full(self, ckpt) -> Optional[np.ndarray]:
        if str(EXPERIMENTS) not in sys.path:
            sys.path.insert(0, str(EXPERIMENTS))
        from common.olmoe_loader import _cache_path

        path = _cache_path(str(ckpt), None)
        if not path.exists():
            return None
        return np.load(path)


def _mean(vals) -> float:
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------- granularity


def run_granularity(data: Stage2Data, out: Path) -> List[Dict]:
    rows: List[Dict] = []
    disp_rows: List[Dict] = []
    for bundle in data.bundles:
        N = bundle["n_tokens"]
        for layer in data.layers(bundle):
            E = data.reservations(bundle, layer)
            for B in data.block_sizes(bundle):
                blocks = step_blocks(N, B, 8)
                envs = {}
                for p in POLICIES:
                    envs[(p, "rate")] = (scale_reservation(E[p], B, N), bundle["default_window"])
                    envs[(p, "step")] = data.step_reservation(bundle, layer, B, p)
                cell_samples: List[np.ndarray] = []
                for ckpt in data.held(bundle):
                    indices, scores, place, _ = data.tokens(bundle, layer, ckpt)
                    for b, (idx, srcb) in enumerate(blocks):
                        M_b = block_matrix(indices[idx], place, srcb, 8)
                        cell_samples.append(M_b)
                        for (p, sizing), (Emat, n_samples) in envs.items():
                            env = score_heldout(M_b, Emat)
                            adm = admit_by_slot(indices[idx], place, srcb, Emat, order="arrival", overflow="drop")
                            rows.append({
                                "source": bundle["name"], "layer": layer, "checkpoint": _ckpt_key(ckpt),
                                "token_source": "aligned", "B": B, "per_rank": B // 8, "block": b,
                                "policy": p, "sizing": sizing, "n_samples_E": n_samples,
                                "E_total_slots": float(Emat.sum()),
                                "overflow_ratio": env["overflow_ratio"], "waste_ratio": env["waste_ratio"],
                                "coverage_ratio": env["coverage_ratio"],
                                "frac_tokens_hit": adm.stats["frac_tokens_hit"],
                                "frac_tokens_fully_unreserved": adm.stats["frac_tokens_fully_unreserved"],
                            })
                stack = np.stack(cell_samples)
                mean = stack.mean(axis=0)
                std = stack.std(axis=0)
                off = ~np.eye(8, dtype=bool) & (mean > 0)
                disp_rows.append({
                    "source": bundle["name"], "layer": layer, "B": B,
                    "n_steps": int(stack.shape[0]),
                    "cell_cv": float(np.mean(std[off] / mean[off])),
                    "cv_over_poisson": float(np.mean(std[off] / np.sqrt(mean[off]))),
                })
            if bundle["name"] == "olmoe":
                for ckpt in data.held(bundle):
                    full = data.olmoe_full(ckpt)
                    if full is None:
                        continue
                    L = int(layer.split("_")[1])
                    ind_full = full[:, L, :].astype(np.int64)
                    place = data.tokens(bundle, layer, ckpt)[2]
                    for B in sorted({OLMOE_DIFF_B, data.headline_B(bundle)}):
                        envs = {}
                        for p in POLICIES:
                            envs[(p, "rate")] = (scale_reservation(E[p], B, N), bundle["default_window"])
                            envs[(p, "step")] = data.step_reservation(bundle, layer, B, p)
                        for b, (idx, srcb) in enumerate(offset_blocks(len(ind_full), OLMOE_PREFIX, B, 8)):
                            M_b = block_matrix(ind_full[idx], place, srcb, 8)
                            for (p, sizing), (Emat, n_samples) in envs.items():
                                env = score_heldout(M_b, Emat)
                                adm = admit_by_slot(ind_full[idx], place, srcb, Emat, order="arrival", overflow="drop")
                                rows.append({
                                    "source": bundle["name"], "layer": layer, "checkpoint": _ckpt_key(ckpt),
                                    "token_source": "different", "B": B, "per_rank": B // 8, "block": b,
                                    "policy": p, "sizing": sizing, "n_samples_E": n_samples,
                                    "E_total_slots": float(Emat.sum()),
                                    "overflow_ratio": env["overflow_ratio"], "waste_ratio": env["waste_ratio"],
                                    "coverage_ratio": env["coverage_ratio"],
                                    "frac_tokens_hit": adm.stats["frac_tokens_hit"],
                                    "frac_tokens_fully_unreserved": adm.stats["frac_tokens_fully_unreserved"],
                                })
            print("  granularity", bundle["name"], layer, flush=True)
    _write_csv(out / "granularity.csv", rows)
    _write_csv(out / "granularity_dispersion.csv", disp_rows)
    return rows


def summarize_granularity(rows: List[Dict]) -> Dict:
    summary: Dict = {}
    for source in sorted({r["source"] for r in rows}):
        summary[source] = {}
        for tsrc in sorted({r["token_source"] for r in rows if r["source"] == source}):
            summary[source][tsrc] = {}
            for B in sorted({r["B"] for r in rows if r["source"] == source and r["token_source"] == tsrc}):
                entry = {}
                for p in POLICIES:
                    for sizing in ("rate", "step"):
                        sel = [r for r in rows if r["source"] == source and r["token_source"] == tsrc
                               and r["B"] == B and r["policy"] == p and r["sizing"] == sizing]
                        if not sel:
                            continue
                        entry["{}_{}".format(p, sizing)] = {
                            "overflow": _mean(r["overflow_ratio"] for r in sel),
                            "waste": _mean(r["waste_ratio"] for r in sel),
                            "tokens_hit": _mean(r["frac_tokens_hit"] for r in sel),
                            "overflow_p90": float(np.percentile([r["overflow_ratio"] for r in sel], 90)),
                            "n": len(sel),
                            "E_ratio_vs_rate": (_mean(r["E_total_slots"] for r in sel)
                                                / max(_mean(r["E_total_slots"] for r in rows
                                                            if r["source"] == source and r["token_source"] == tsrc
                                                            and r["B"] == B and r["policy"] == p and r["sizing"] == "rate"), 1e-9)),
                        }
                summary[source][tsrc][str(B)] = entry
    return summary


# ------------------------------------------------------------------------ shedding


def _shedding_arms(k: int, has_scores: bool) -> List[Dict]:
    arms = [
        {"arm": "arrival-drop", "order": "arrival", "overflow": "drop"},
        {"arm": "rank-drop", "order": "rank", "overflow": "drop"},
        {"arm": "tail-all", "order": "arrival", "overflow": "tail"},
        {"arm": "rank-tail", "order": "rank", "overflow": "tail"},
        {"arm": "tail-rankcut-{}".format(k - 1), "order": "arrival", "overflow": "tail", "rank_cut": k - 1},
        {"arm": "tail-rankcut-{}".format(k - 2), "order": "arrival", "overflow": "tail", "rank_cut": k - 2},
    ]
    if has_scores:
        arms += [
            {"arm": "score-drop", "order": "score", "overflow": "drop"},
            {"arm": "score-tail", "order": "score", "overflow": "tail"},
        ]
        arms += [{"arm": "tail-gate-{:.2f}".format(g), "order": "arrival", "overflow": "tail", "weight_gate": g}
                 for g in WEIGHT_GATES]
    return arms


def run_shedding(data: Stage2Data, out: Path) -> List[Dict]:
    rows: List[Dict] = []
    for bundle in data.bundles:
        N = bundle["n_tokens"]
        for layer in data.layers(bundle):
            E = data.reservations(bundle, layer)
            for B in sorted({N, data.headline_B(bundle)}):
                blocks = step_blocks(N, B, 8)
                for ckpt in data.held(bundle):
                    indices, scores, place, _ = data.tokens(bundle, layer, ckpt)
                    k = indices.shape[1]
                    for b, (idx, srcb) in enumerate(blocks):
                        ind_b = indices[idx]
                        sc_b = scores[idx] if scores is not None else None
                        for p in ("p95", "mean"):
                            Emat = scale_reservation(E[p], B, N)
                            masks = {}
                            for spec in _shedding_arms(k, sc_b is not None):
                                adm = admit_by_slot(
                                    ind_b, place, srcb, Emat, scores=sc_b,
                                    order=spec["order"], overflow=spec["overflow"],
                                    weight_gate=spec.get("weight_gate"), rank_cut=spec.get("rank_cut"),
                                )
                                masks[spec["arm"]] = ~adm.reserved
                                row = {
                                    "source": bundle["name"], "layer": layer, "checkpoint": _ckpt_key(ckpt),
                                    "B": B, "block": b, "policy": p, "arm": spec["arm"],
                                    "order": spec["order"], "overflow": spec["overflow"],
                                    "gate": spec.get("weight_gate", spec.get("rank_cut", "")),
                                }
                                for key, val in adm.stats.items():
                                    if isinstance(val, list):
                                        row[key] = json.dumps(val)
                                    else:
                                        row[key] = val
                                rows.append(row)
                            if "score-drop" in masks:
                                a, bmask = masks["score-drop"], masks["rank-drop"]
                                inter = float((a & bmask).sum())
                                union = float((a | bmask).sum())
                                for r in rows[-len(masks):]:
                                    r["jaccard_score_vs_rank"] = inter / union if union else 1.0
            print("  shedding", bundle["name"], layer, flush=True)
    _write_csv(out / "shedding.csv", rows)
    return rows


def summarize_shedding(rows: List[Dict]) -> Dict:
    summary: Dict = {}
    for source in sorted({r["source"] for r in rows}):
        summary[source] = {}
        for B in sorted({r["B"] for r in rows if r["source"] == source}):
            summary[source][str(B)] = {}
            for p in ("p95", "mean"):
                block = {}
                for arm in sorted({r["arm"] for r in rows if r["source"] == source}):
                    sel = [r for r in rows if r["source"] == source and r["B"] == B and r["policy"] == p and r["arm"] == arm]
                    if not sel:
                        continue
                    ent = {
                        "frac_slots_over": _mean(r["frac_slots_over"] for r in sel),
                        "frac_slots_tail": _mean(r["frac_slots_tail"] for r in sel),
                        "frac_slots_dropped": _mean(r["frac_slots_dropped"] for r in sel),
                        "frac_tokens_hit": _mean(r["frac_tokens_hit"] for r in sel),
                        "frac_tokens_fully_dropped": _mean(r["frac_tokens_fully_dropped"] for r in sel),
                    }
                    for key in ("weight_mass_over_per_token", "weight_mass_tail_per_token",
                                "weight_mass_dropped_per_token", "mean_weight_over", "mean_weight_all",
                                "jaccard_score_vs_rank"):
                        if key in sel[0] and sel[0][key] not in ("", None):
                            ent[key] = _mean(r[key] for r in sel if r.get(key) not in ("", None))
                    hist = np.sum([json.loads(r["rank_hist_over"]) for r in sel], axis=0)
                    ent["rank_hist_over"] = (hist / max(hist.sum(), 1)).tolist()
                    dh = np.sum([json.loads(r["rank_hist_dropped"]) for r in sel], axis=0)
                    ent["rank_hist_dropped"] = (dh / max(dh.sum(), 1)).tolist()
                    block[arm] = ent
                summary[source][str(B)][p] = block
    return summary


# ---------------------------------------------------------------------------- links


def frozen_paths_for(E_route: np.ndarray, topology: str) -> Dict:
    payload = plan_matrix(E_route, topology, bytes_per_slot=BYTES)
    from puppeteer.config import RunConfig

    return paths_from_plan(payload["plan"], RunConfig.load(topology).topology)


def run_links(data: Stage2Data, out: Path) -> List[Dict]:
    model = LinkModel(str(DEFAULT_TOPOLOGY))
    routes_ecmp = [model.ecmp_routes(seed) for seed in range(TAIL_SEEDS)]
    routes_spray = model.spray_routes()
    rows: List[Dict] = []
    for bundle in data.bundles:
        N = bundle["n_tokens"]
        for layer in data.layers(bundle):
            E = data.reservations(bundle, layer)
            frozen = {
                "dispatch": model.frozen_routes(frozen_paths_for(E["mean"], str(DEFAULT_TOPOLOGY))),
                "combine": model.frozen_routes(frozen_paths_for(E["mean"].T.copy(), str(DEFAULT_TOPOLOGY))),
            }
            for B in sorted({N, data.headline_B(bundle)}):
                blocks = step_blocks(N, B, 8)
                for ckpt in data.held(bundle):
                    indices, _, place, _ = data.tokens(bundle, layer, ckpt)
                    for b, (idx, srcb) in enumerate(blocks):
                        M_b = block_matrix(indices[idx], place, srcb, 8)
                        for p in ("p95", "mean"):
                            Emat = scale_reservation(E[p], B, N)
                            for direction, M_d, E_d in (("dispatch", M_b, Emat), ("combine", M_b.T.copy(), Emat.T.copy())):
                                tails = [("frozen", frozen[direction]), ("spray", routes_spray)] + \
                                        [("ecmp{}".format(s), routes_ecmp[s]) for s in range(TAIL_SEEDS)]
                                for tail_name, routes_tail in tails:
                                    bud = model.budget(M_d, E_d, frozen[direction], routes_tail, BYTES)
                                    rows.append({
                                        "source": bundle["name"], "layer": layer, "checkpoint": _ckpt_key(ckpt),
                                        "B": B, "block": b, "policy": p, "direction": direction, "tail_routing": tail_name,
                                        "coverage": bud["coverage"], "n_links_uncovered": bud["n_links_uncovered"],
                                        "worst_link_overflow_over_refund": bud["worst_link_overflow_over_refund"],
                                        "overflow_bytes": bud["overflow_bytes"], "refund_bytes": bud["refund_bytes"],
                                        "lb_reserved_s": bud["lb_reserved_s"], "lb_total_s": bud["lb_total_s"],
                                        "lb_E_s": bud["lb_E_s"], "predicted_extension_s": bud["predicted_extension_s"],
                                        "fits_plan_budget": bud["fits_plan_budget"],
                                        "binding_link_reserved": bud["binding_link_reserved"],
                                        "binding_link_total": bud["binding_link_total"],
                                    })
            print("  links", bundle["name"], layer, flush=True)
    _write_csv(out / "links.csv", rows)
    return rows


def summarize_links(rows: List[Dict]) -> Dict:
    summary: Dict = {}
    for source in sorted({r["source"] for r in rows}):
        summary[source] = {}
        for B in sorted({r["B"] for r in rows if r["source"] == source}):
            summary[source][str(B)] = {}
            for p in ("p95", "mean"):
                ent = {}
                for tail in sorted({r["tail_routing"] for r in rows}):
                    sel = [r for r in rows if r["source"] == source and r["B"] == B and r["policy"] == p and r["tail_routing"] == tail]
                    if not sel:
                        continue
                    ent[tail] = {
                        "coverage": _mean(r["coverage"] for r in sel),
                        "n_links_uncovered": _mean(r["n_links_uncovered"] for r in sel),
                        "frac_fits_plan_budget": _mean(1.0 if r["fits_plan_budget"] else 0.0 for r in sel),
                        "predicted_extension_rel": _mean(r["predicted_extension_s"] / r["lb_reserved_s"] for r in sel if r["lb_reserved_s"] > 0),
                        "refund_over_overflow_bytes": float(np.median([r["refund_bytes"] / r["overflow_bytes"] for r in sel if r["overflow_bytes"] > 0] or [0.0])),
                    }
                summary[source][str(B)][p] = ent
    return summary


# ------------------------------------------------------------------------- astrasim


def run_astrasim(data: Stage2Data, out: Path) -> List[Dict]:
    """Switch 8x100 GB/s bracket per layer and direction at the last held-out ckpt."""
    work = out / "astrasim_work"
    cache = out / "cache" / "astrasim"
    work.mkdir(parents=True, exist_ok=True)
    switch = write_switch_yaml(work / "network_100G.yml", npus=8, bandwidth=SWITCH_BW, latency=SWITCH_LAT)
    jobs, meta = [], []

    def enqueue(phases, name, row):
        jobs.append({
            "phases": [np.asarray(p, dtype=np.float64) for p in phases],
            "work": str(work / name), "name": name, "network": str(switch),
            "bytes_per_slot": BYTES, "cache_dir": str(cache),
            "bandwidth": SWITCH_BW, "latency": SWITCH_LAT,
        })
        meta.append(row)

    for bundle in data.bundles:
        N = bundle["n_tokens"]
        ckpt = data.held(bundle)[-1]
        for layer in data.layers(bundle):
            E = data.reservations(bundle, layer)
            indices, _, place, _ = data.tokens(bundle, layer, ckpt)
            for B in sorted({N, data.headline_B(bundle)}):
                for b, (idx, srcb) in enumerate(step_blocks(N, B, 8)):
                    M_b = block_matrix(indices[idx], place, srcb, 8)
                    for p in ("p95", "mean"):
                        Emat = scale_reservation(E[p], B, N)
                        M_res = np.minimum(M_b, np.floor(Emat + 1e-9))
                        M_tail = M_b - M_res
                        for direction, R, T, Mfull in (("dispatch", M_res, M_tail, M_b),
                                                       ("combine", M_res.T.copy(), M_tail.T.copy(), M_b.T.copy())):
                            base = dict(source=bundle["name"], layer=layer, checkpoint=_ckpt_key(ckpt),
                                        B=B, block=b, policy=p, direction=direction,
                                        reserved_slots=float(R.sum()), tail_slots=float(T.sum()))
                            tag = "{}_{}_{}_{}_b{}_{}_{}".format(bundle["name"], layer, _ckpt_key(ckpt), B, b, p, direction)
                            enqueue([R], "s2_res_" + tag, dict(base, arm="reserved_only"))
                            enqueue([Mfull], "s2_one_" + tag, dict(base, arm="one_class"))
                            enqueue([R, T], "s2_ser_" + tag, dict(base, arm="serialized"))
    workers = min(6, os.cpu_count() or 2)
    print("  astrasim jobs", len(jobs), "workers", workers, flush=True)
    finished = run_phased_jobs(jobs, workers=workers)
    rows = []
    for row, result in zip(meta, finished):
        r = dict(row)
        r["cycles"] = int(result["cycles"])
        rows.append(r)
    _write_csv(out / "astrasim.csv", rows)
    return rows


def summarize_astrasim(rows: List[Dict]) -> Dict:
    summary: Dict = {}
    for source in sorted({r["source"] for r in rows}):
        summary[source] = {}
        for B in sorted({r["B"] for r in rows if r["source"] == source}):
            summary[source][str(B)] = {}
            for p in ("p95", "mean"):
                ent = {}
                blocks = sorted({r["block"] for r in rows if r["source"] == source and r["B"] == B})
                for arm in ("reserved_only", "one_class", "serialized"):
                    totals = []
                    for b in blocks:
                        sel = [r for r in rows if r["source"] == source and r["B"] == B and r["policy"] == p
                               and r["arm"] == arm and r["block"] == b]
                        totals.append(sum(r["cycles"] for r in sel))
                    ent[arm] = _mean(totals)
                ent["tail_share"] = _mean(
                    r["tail_slots"] / max(r["reserved_slots"] + r["tail_slots"], 1)
                    for r in rows if r["source"] == source and r["B"] == B and r["policy"] == p and r["arm"] == "reserved_only")
                ent["serialized_over_one_class"] = ent["serialized"] / ent["one_class"] if ent["one_class"] else 0.0
                ent["one_class_over_reserved"] = ent["one_class"] / ent["reserved_only"] if ent["reserved_only"] else 0.0
                summary[source][str(B)][p] = ent
    return summary


# ----------------------------------------------------------------------------- clos


_PATH_CACHE: Dict[bytes, Dict] = {}


def _frozen(E_route: np.ndarray) -> Dict:
    key = np.asarray(E_route, dtype=np.float64).tobytes()
    if key not in _PATH_CACHE:
        _PATH_CACHE[key] = frozen_paths_for(np.asarray(E_route, dtype=np.float64), str(DEFAULT_TOPOLOGY))
    return _PATH_CACHE[key]


def _plan_job(job: Dict) -> Dict:
    """Worker: one Puppeteer plan. Returns metrics only (picklable)."""
    kind = job["kind"]
    if kind == "single":
        frozen = _frozen(job["route"]) if job.get("route") is not None else None
        payload = plan_matrix(
            job["matrix"], str(DEFAULT_TOPOLOGY), bytes_per_slot=BYTES,
            router_kind=job.get("router", "least-loaded"), router_seed=job.get("seed", 0),
            frozen_paths=frozen,
        )
        m = payload["plan"].metrics
        return {
            "iteration_time_s": float(payload["iteration_time_s"]),
            "congestion_overhead_s": float(m.get("congestion_overhead_s") or 0.0),
            "ideal_iteration_time_s": float(payload["ideal_iteration_time_s"]),
        }
    if kind == "two_class":
        from .planner import plan_two_class

        res = plan_two_class(
            job["reserved"], job["tail"], str(DEFAULT_TOPOLOGY),
            frozen_paths=_frozen(job["route"]), tail_router_kind=job.get("router", "ecmp"),
            tail_seed=job.get("seed", 0), bytes_per_slot=BYTES,
        )
        cls = res["classes"]
        return {
            "iteration_time_s": float(res["total_makespan_s"]),
            "reserved_makespan_s": float(cls.get("reserved", {}).get("makespan_s", 0.0)),
            "tail_makespan_s": float(cls.get("tail", {}).get("makespan_s", 0.0)),
            "reserved_alone_makespan_s": float(res["reserved_alone_makespan_s"]),
            "tail_extension_s": float(res["tail_extension_s"]),
            "tail_bytes": float(cls.get("tail", {}).get("bytes", 0.0)),
        }
    raise ValueError(kind)


def _run_plan_jobs(jobs: List[Dict], workers: int) -> List[Dict]:
    if workers <= 1 or len(jobs) < 4:
        return [_plan_job(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor

    out: List[Optional[Dict]] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(_plan_job, jobs, chunksize=4)):
            out[i] = res
            if i % 200 == 0:
                print("  clos plans {}/{}".format(i, len(jobs)), flush=True)
    return out  # type: ignore[return-value]


def run_clos(data: Stage2Data, out: Path) -> List[Dict]:
    """Puppeteer fluid Clos: two classes in one loop, plus every control."""
    jobs: List[Dict] = []
    meta: List[Dict] = []

    def add(job: Dict, row: Dict) -> None:
        jobs.append(job)
        meta.append(row)

    for bundle in data.bundles:
        N = bundle["n_tokens"]
        ckpt = data.held(bundle)[-1]
        headline = data.headline_B(bundle)
        for layer in data.layers(bundle):
            E = data.reservations(bundle, layer)
            indices, _, place, _ = data.tokens(bundle, layer, ckpt)
            for B in sorted({N, headline}):
                envelopes: Dict[Tuple[str, str], np.ndarray] = {}
                if B == headline:
                    for p in POLICIES:
                        envelopes[(p, "rate")] = scale_reservation(E[p], B, N)
                        envelopes[(p, "step")] = data.step_reservation(bundle, layer, B, p)[0]
                else:
                    envelopes[("p95", "rate")] = scale_reservation(E["p95"], B, N)
                for b, (idx, srcb) in enumerate(step_blocks(N, B, 8)):
                    M_b = block_matrix(indices[idx], place, srcb, 8)
                    for direction in ("dispatch", "combine"):
                        Mfull = M_b if direction == "dispatch" else M_b.T.copy()
                        route = E["mean"] if direction == "dispatch" else E["mean"].T.copy()
                        base = dict(source=bundle["name"], layer=layer, checkpoint=_ckpt_key(ckpt),
                                    B=B, block=b, direction=direction, policy="", sizing="", seed=0)
                        add({"kind": "single", "matrix": Mfull, "route": route}, dict(base, arm="full_one_class_frozen"))
                        add({"kind": "single", "matrix": Mfull, "router": "spray"}, dict(base, arm="full_spray"))
                        for s in range(TAIL_SEEDS):
                            add({"kind": "single", "matrix": Mfull, "router": "ecmp", "seed": s}, dict(base, arm="full_ecmp", seed=s))
                        for (p, sizing), Emat in envelopes.items():
                            E_d = Emat if direction == "dispatch" else Emat.T.copy()
                            R = np.minimum(Mfull, np.floor(E_d + 1e-9))
                            T = Mfull - R
                            env_base = dict(base, policy=p, sizing=sizing,
                                            reserved_slots=float(R.sum()), tail_slots=float(T.sum()),
                                            E_slots=float(E_d.sum()))
                            add({"kind": "single", "matrix": R, "route": route}, dict(env_base, arm="reserved_only"))
                            add({"kind": "single", "matrix": E_d, "route": route}, dict(env_base, arm="plan_E"))
                            add({"kind": "two_class", "reserved": R, "tail": T, "route": route, "router": "ecmp", "seed": 0},
                                dict(env_base, arm="two_class_ecmp", seed=0))
                            if p == "p95" and sizing == "rate":
                                for s in range(1, TAIL_SEEDS):
                                    add({"kind": "two_class", "reserved": R, "tail": T, "route": route, "router": "ecmp", "seed": s},
                                        dict(env_base, arm="two_class_ecmp", seed=s))
                                add({"kind": "two_class", "reserved": R, "tail": T, "route": route, "router": "spray"},
                                    dict(env_base, arm="two_class_spray"))
    workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    print("  clos plans", len(jobs), "workers", workers, flush=True)
    results = _run_plan_jobs(jobs, workers)
    rows = []
    for row, res in zip(meta, results):
        r = dict(row)
        r.update(res or {})
        rows.append(r)
    _write_csv(out / "clos.csv", rows)
    return rows


def _whole_model(rows: List[Dict], source: str, B: int, arm: str, policy: str = "", sizing: str = "",
                 key: str = "iteration_time_s") -> float:
    """Sum over layers and directions, mean over blocks and seeds."""
    sel = [r for r in rows if r["source"] == source and r["B"] == B and r["arm"] == arm
           and (policy == "" or r["policy"] == policy) and (sizing == "" or r["sizing"] == sizing)]
    if not sel:
        return 0.0
    totals = []
    for b in sorted({r["block"] for r in sel}):
        for s in sorted({r["seed"] for r in sel}):
            part = [r for r in sel if r["block"] == b and r["seed"] == s]
            if part:
                totals.append(sum(float(r.get(key, 0.0) or 0.0) for r in part))
    return _mean(totals)


def summarize_clos(rows: List[Dict]) -> Dict:
    summary: Dict = {}
    for source in sorted({r["source"] for r in rows}):
        summary[source] = {}
        for B in sorted({r["B"] for r in rows if r["source"] == source}):
            ent: Dict = {
                "full_one_class_frozen_s": _whole_model(rows, source, B, "full_one_class_frozen"),
                "full_ecmp_s": _whole_model(rows, source, B, "full_ecmp"),
                "full_spray_s": _whole_model(rows, source, B, "full_spray"),
                "envelopes": {},
            }
            for (p, sizing) in sorted({(r["policy"], r["sizing"]) for r in rows
                                       if r["source"] == source and r["B"] == B and r["policy"]}):
                e = {
                    "reserved_only_s": _whole_model(rows, source, B, "reserved_only", p, sizing),
                    "plan_E_s": _whole_model(rows, source, B, "plan_E", p, sizing),
                    "two_class_ecmp_total_s": _whole_model(rows, source, B, "two_class_ecmp", p, sizing),
                    "two_class_ecmp_reserved_s": _whole_model(rows, source, B, "two_class_ecmp", p, sizing, "reserved_makespan_s"),
                    "two_class_ecmp_tail_s": _whole_model(rows, source, B, "two_class_ecmp", p, sizing, "tail_makespan_s"),
                    "two_class_ecmp_extension_s": _whole_model(rows, source, B, "two_class_ecmp", p, sizing, "tail_extension_s"),
                    "two_class_spray_total_s": _whole_model(rows, source, B, "two_class_spray", p, sizing),
                    "two_class_spray_extension_s": _whole_model(rows, source, B, "two_class_spray", p, sizing, "tail_extension_s"),
                    "reserved_alone_check_max_rel": max(
                        [abs(r["reserved_makespan_s"] - r["reserved_alone_makespan_s"]) / max(r["reserved_alone_makespan_s"], 1e-12)
                         for r in rows if r["source"] == source and r["B"] == B and r["policy"] == p and r["sizing"] == sizing
                         and r["arm"].startswith("two_class")] or [0.0]),
                    "tail_share": _mean(r["tail_slots"] / max(r["reserved_slots"] + r["tail_slots"], 1)
                                        for r in rows if r["source"] == source and r["B"] == B and r["policy"] == p
                                        and r["sizing"] == sizing and r["arm"] == "reserved_only"),
                }
                if e["reserved_only_s"]:
                    e["extension_rel"] = e["two_class_ecmp_extension_s"] / e["reserved_only_s"]
                    e["total_over_reserved"] = e["two_class_ecmp_total_s"] / e["reserved_only_s"]
                    e["total_over_plan_E"] = e["two_class_ecmp_total_s"] / e["plan_E_s"] if e["plan_E_s"] else 0.0
                    e["total_over_full_one_class"] = e["two_class_ecmp_total_s"] / ent["full_one_class_frozen_s"] if ent["full_one_class_frozen_s"] else 0.0
                ent["envelopes"]["{}_{}".format(p, sizing)] = e
            summary[source][str(B)] = ent
    return summary


# ------------------------------------------------------------------------ tightness


def summarize_tightness(gran: Dict, clos: Dict) -> Dict:
    """Join envelope policy x sizing at the headline B: share, idle, timetable, tail."""
    out: Dict = {}
    for source, per_B in clos.items():
        headline = str(HEADLINE_B[source])
        if headline not in per_B:
            continue
        out[source] = {}
        g = gran.get(source, {}).get("aligned", {}).get(headline, {})
        for env, e in per_B[headline]["envelopes"].items():
            ge = g.get(env, {})
            out[source][env] = {
                "guaranteed_share": 1.0 - ge.get("overflow", 0.0),
                "idle_reserve": ge.get("waste", 0.0),
                "tokens_hit": ge.get("tokens_hit", 0.0),
                "timetable_plan_E_s": e.get("plan_E_s", 0.0),
                "reserved_only_s": e.get("reserved_only_s", 0.0),
                "two_class_total_s": e.get("two_class_ecmp_total_s", 0.0),
                "extension_rel": e.get("extension_rel", 0.0),
                "total_over_plan_E": e.get("total_over_plan_E", 0.0),
            }
    return out


# ------------------------------------------------------------------------------ ns3


def run_ns3_export(data: Stage2Data, out: Path, links_rows: Optional[List[Dict]] = None) -> List[Dict]:
    """Prepare (do not run) packet-level inputs for the busiest layer of each model."""
    from .ns3_export import export_step

    rows = []
    for bundle in data.bundles:
        N = bundle["n_tokens"]
        ckpt = data.held(bundle)[-1]
        B = data.headline_B(bundle)
        layer = data.layers(bundle)[0]
        if links_rows:
            best = -1.0
            for cand in data.layers(bundle):
                sel = [float(r["predicted_extension_s"]) for r in links_rows
                       if r["source"] == bundle["name"] and r["layer"] == cand and int(r["B"]) == B
                       and r["policy"] == "p95" and r["tail_routing"] == "frozen"]
                if sel and np.mean(sel) > best:
                    best, layer = float(np.mean(sel)), cand
        E = data.reservations(bundle, layer)
        indices, _, place, _ = data.tokens(bundle, layer, ckpt)
        idx, srcb = step_blocks(N, B, 8)[0]
        M_b = block_matrix(indices[idx], place, srcb, 8)
        Emat = scale_reservation(E["p95"], B, N)
        R = np.minimum(M_b, np.floor(Emat + 1e-9))
        T = M_b - R
        for routing in ("ecmp", "spray"):
            anchor = compute_anchor(bundle["name"])
            dest = export_step(
                str(out / "ns3" / "{}_{}_{}".format(bundle["name"], layer, routing)),
                manifest={"source": bundle["name"], "layer": layer, "checkpoint": _ckpt_key(ckpt),
                          "B": B, "block": 0, "policy": "p95", "sizing": "rate", "tail_routing": routing,
                          "kernel": {"seconds_per_slot": anchor["seconds_per_slot_mid"],
                                     "compute_to_wire_ratio": anchor["ratio"],
                                     "launch_floor_s": KERNEL_FLOOR_S,
                                     "structures": ["monolithic (today)", "chunked (proposal)"],
                                     "semantics": "nccl: a rank's collective completes when all its sends and receives are done"}},
                M_reserved_slots=R, M_tail_slots=T, route_slots=E["mean"], tail_routing=routing,
                bytes_per_slot=BYTES,
            )
            rows.append({"source": bundle["name"], "layer": layer, "tail_routing": routing, "dir": str(dest),
                         "reserved_bytes": float(R.sum() * BYTES), "tail_bytes": float(T.sum() * BYTES)})
    _write_csv(out / "ns3_exports.csv", rows)
    return rows


# ---------------------------------------------------------------------- resummarize


def _load_rows(out: Path, name: str) -> List[Dict]:
    import csv

    path = out / "{}.csv".format(name)
    if not path.exists():
        return []
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    text_cols = {"source", "layer", "checkpoint", "token_source", "policy", "sizing", "arm", "order",
                 "overflow", "direction", "tail_routing", "gate", "binding_link_reserved",
                 "binding_link_total", "dir", "structure", "allocator", "deadline", "tail_router_kind"}
    for r in rows:
        for k, v in list(r.items()):
            if k in text_cols or v is None or (isinstance(v, str) and v.startswith("[")):
                continue
            if v in ("True", "False"):
                r[k] = v == "True"
                continue
            try:
                r[k] = int(v) if v.lstrip("-").isdigit() else float(v)
            except (ValueError, AttributeError):
                pass
    return rows


def resummarize(out: Path) -> Dict:
    """Rebuild summary.json from the CSVs (stages may have run in separate processes)."""
    out = Path(out)
    summary: Dict = {}
    for name, fn in (("granularity", summarize_granularity), ("shedding", summarize_shedding),
                     ("links", summarize_links), ("astrasim", summarize_astrasim), ("clos", summarize_clos)):
        rows = _load_rows(out, name)
        if rows:
            summary[name] = fn(rows)
    step_rows = _load_rows(out, "step")
    if step_rows:
        summary["step"] = summarize_step(step_rows, _load_rows(out, "deadline"))
    if "granularity" in summary and "clos" in summary:
        summary["tightness"] = summarize_tightness(summary["granularity"], summary["clos"])
    _write_json(out / "summary.json", summary)
    return summary


# ----------------------------------------------------------------------------- main


def run_stage2(out: Path, sources: Sequence[str] = ("flame", "olmoe"), stages: Sequence[str] = ALL_STAGES) -> Dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stages = tuple(stages)
    print("loading traces (scores kept)", flush=True)
    data = Stage2Data(sources)
    summary_path = out / "summary.json"
    summary: Dict = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    def load_rows(name: str) -> List[Dict]:
        import csv

        path = out / "{}.csv".format(name)
        if not path.exists():
            return []
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        for r in rows:
            for k, v in list(r.items()):
                if k in ("source", "layer", "checkpoint", "token_source", "policy", "sizing", "arm", "order",
                         "overflow", "direction", "tail_routing", "gate", "binding_link_reserved",
                         "binding_link_total") or (isinstance(v, str) and v.startswith("[")):
                    continue
                if v in ("True", "False"):
                    r[k] = v == "True"
                    continue
                try:
                    r[k] = int(v) if v.isdigit() else float(v)
                except (ValueError, AttributeError):
                    pass
        return rows

    if "granularity" in stages:
        rows = run_granularity(data, out)
        summary["granularity"] = summarize_granularity(rows)
        _write_json(summary_path, summary)
    if "shedding" in stages:
        rows = run_shedding(data, out)
        summary["shedding"] = summarize_shedding(rows)
        _write_json(summary_path, summary)
    if "links" in stages:
        rows = run_links(data, out)
        summary["links"] = summarize_links(rows)
        _write_json(summary_path, summary)
    if "astrasim" in stages:
        rows = run_astrasim(data, out)
        summary["astrasim"] = summarize_astrasim(rows)
        _write_json(summary_path, summary)
    if "clos" in stages:
        rows = run_clos(data, out)
        summary["clos"] = summarize_clos(rows)
        _write_json(summary_path, summary)
    if "tightness" in stages:
        if "granularity" not in summary:
            summary["granularity"] = summarize_granularity(load_rows("granularity"))
        if "clos" not in summary:
            summary["clos"] = summarize_clos(load_rows("clos"))
        summary["tightness"] = summarize_tightness(summary["granularity"], summary["clos"])
        _write_json(summary_path, summary)
    if "step" in stages:
        rows, dl_rows = run_step(data, out)
        summary["step"] = summarize_step(rows, dl_rows)
        _write_json(summary_path, summary)
    if "ns3" in stages:
        summary["ns3"] = run_ns3_export(data, out, load_rows("links"))
        _write_json(summary_path, summary)
    summary["note"] = (
        "Membership lens (k slots per token). Reserved = min(M, floor(E)) for every arm. "
        "rate = Stage 1 E scaled by B/N; step = policy over window steps of B tokens. "
        "ASTRA-sim Switch cannot express classes: one_class and serialized bracket the "
        "priority schedule. Puppeteer fluid strict priority is ideal preemption (no queues, "
        "DCQCN, or reorder); tail numbers are lower bounds."
    )
    _write_json(summary_path, summary)
    return summary


# ===================================================================== Stage 2b: step

from .planner import WIRE_S_PER_SLOT, deadline_quota_replan, plan_two_class_step  # noqa: E402

COMPUTE_RATIOS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
LATENESS_FRACS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)
PRIMARY_ALLOCATOR = "max-min"
SENSITIVITY_ALLOCATOR = "tte-priority"
ALLOCATORS = (PRIMARY_ALLOCATOR, SENSITIVITY_ALLOCATOR)
KERNEL_FLOOR_S = 30e-6
KERNEL_FLOOR_SENS = (0.0, 50e-6)
EXPERT_DIMS = {"flame": (1024, 704), "olmoe": (2048, 1024)}
TRUE_BYTES_PER_SLOT = {"flame": 2048, "olmoe": 4096}
GPU_BANDS = (("H100 bf16 989T x MFU 0.3", 989e12 * 0.3), ("500T x MFU 0.4 (topology yaml)", 500e12 * 0.4),
             ("A100 bf16 312T x MFU 0.4", 312e12 * 0.4))


def compute_anchor(source: str) -> Dict[str, object]:
    """Expert compute per slot vs the wire time of one slot at 100 Gbps, as a band."""
    d, f = EXPERT_DIMS[source]
    flops = 6.0 * d * f
    mid = flops / GPU_BANDS[1][1]
    band = {name: (flops / fl) / WIRE_S_PER_SLOT for name, fl in GPU_BANDS}
    true_wire = TRUE_BYTES_PER_SLOT[source] * 8.0 / 100e9
    return {
        "d_model": d, "d_ff": f, "flops_per_slot": flops,
        "seconds_per_slot_mid": mid, "ratio": mid / WIRE_S_PER_SLOT,
        "ratio_band": band, "ratio_true_bytes": mid / true_wire,
        "ratio_400G": 4 * mid / WIRE_S_PER_SLOT,
        "note": "forward expert FLOPs only; backward is 2x; shared expert (FLAME, 1408 hidden) and "
                "permute gathers are local extra hide budget, not modelled",
    }


def _step_job(job: Dict) -> Dict:
    """Worker: one step plan (+ optional quota re-plans). Returns metrics only."""
    kw = dict(frozen_paths=_frozen(job["route"]), compute_s_per_slot=job["c"], structure=job["structure"],
              tail_router_kind=job.get("router", "ecmp"), tail_seed=job.get("seed", 0), bytes_per_slot=BYTES,
              allocator=job.get("allocator"), semantics=job.get("semantics", "nccl"),
              compute_floor_s=job.get("floor", KERNEL_FLOOR_S))
    res = plan_two_class_step(job["reserved"], job["tail"], str(DEFAULT_TOPOLOGY), deadline_deltas=(),
                              reserved_only=job.get("reserved_only", False), **kw)
    out = {k: v for k, v in res.items() if k not in ("plan", "deadline", "expert_reserved_finish_s", "expert_tail_finish_s")}
    ef = res["expert_reserved_finish_s"]
    out["expert_reserved_finish_max_s"] = max(ef.values()) if ef else 0.0
    if job.get("deadline") and not job.get("reserved_only") and res["tail_bytes"] > 0:
        T_step = res["reserved_combine_finish_s"]
        cuts = []
        for frac in LATENESS_FRACS:
            cut = deadline_quota_replan(job["reserved"], job["tail"], res, frac * T_step, str(DEFAULT_TOPOLOGY), **kw)
            cuts.append({
                "lateness_frac": frac, "lateness_s": cut["lateness_s"],
                "dispatch_overshoot_s": cut["dispatch_overshoot_s"],
                "cut_step_makespan_s": cut["cut_step_makespan_s"],
                "tail_bytes": cut["tail_bytes"].tolist(),
                "dispatch_quota_bytes": cut["dispatch_quota_bytes"].tolist(),
                "combine_bytes": cut["combine_bytes"].tolist(),
                "combine_delivered_bytes": cut["combine_delivered_bytes"].tolist(),
            })
        out["_cuts"] = cuts
    return out


def _tail_profiles(indices, scores, place, src, Emat, D: int = 8) -> Dict[str, Dict]:
    """Per admission order: per dispatch cell, tail-slot weights ascending / positions descending."""
    out = {}
    n, k = indices.shape
    dest = place[indices]
    cell = (src[:, None] * D + dest).ravel()
    col = np.tile(np.arange(k), n)
    w = (scores / scores.sum(1, keepdims=True)).ravel() if scores is not None else None
    for order in (("arrival", "rank") + (("score",) if scores is not None else ())):
        adm = admit_by_slot(indices, place, src, Emat, scores=scores, order=order, overflow="tail")
        tail = adm.tail.ravel()
        prof: Dict[Tuple[int, int], Dict] = {}
        for c in np.unique(cell[tail]):
            sel = tail & (cell == c)
            s_, d_ = divmod(int(c), D)
            entry = {"n": int(sel.sum()), "pos_desc": np.sort(col[sel])[::-1]}
            if w is not None:
                entry["w_asc"] = np.sort(w[sel])
            prof[(s_, d_)] = entry
        out[order] = {"prof": prof, "n_tokens": n, "k": k}
    return out


def _attribute_cut(cut: Dict, profiles: Dict) -> List[Dict]:
    """Dropped slots and their weight for one lateness, per admission order and attribution."""
    tail_b = np.asarray(cut["tail_bytes"])
    quota = np.asarray(cut["dispatch_quota_bytes"])
    comb_b = np.asarray(cut["combine_bytes"])
    comb_del = np.asarray(cut["combine_delivered_bytes"])
    rows = []
    for order, P in profiles.items():
        prof, n_tokens, k = P["prof"], P["n_tokens"], P["k"]
        n_tail = sum(p["n"] for p in prof.values())
        acc = {"best": [0.0, 0.0, np.zeros(k)], "random": [0.0, 0.0, np.zeros(k)]}
        for (s_, d_), p in prof.items():
            b = tail_b[s_, d_]
            if b <= 0:
                continue
            fd = 1.0 - min(quota[s_, d_], b) / b                          # dropped at dispatch (sender quota)
            cb = comb_b[s_, d_]
            fc = (1.0 - min(comb_del[s_, d_], cb) / cb) if cb > 0 else 0.0
            f_total = fd + (1.0 - fd) * fc
            nc = p["n"]
            kd = int(round(f_total * nc))
            acc["best"][0] += kd
            acc["random"][0] += f_total * nc
            if "w_asc" in p:
                acc["best"][1] += float(p["w_asc"][:kd].sum())
                acc["random"][1] += float(f_total * p["w_asc"].sum())
            if kd > 0:
                acc["best"][2] += np.bincount(p["pos_desc"][:kd], minlength=k)
                acc["random"][2] += f_total * np.bincount(p["pos_desc"], minlength=k)
        for attr, (n_drop, cost, hist) in acc.items():
            rows.append({
                "order": order, "attribution": attr,
                "dropped_frac_of_tail_slots": n_drop / n_tail if n_tail else 0.0,
                "dropped_frac_of_all_slots": n_drop / (n_tokens * k) if n_tokens else 0.0,
                "cost_per_token": (cost / n_tokens) if (n_tokens and "w_asc" in next(iter(prof.values()), {})) else None,
                "dropped_pos_hist": json.dumps((hist / max(hist.sum(), 1e-9)).tolist()),
            })
    return rows


def run_step(data: Stage2Data, out: Path) -> Tuple[List[Dict], List[Dict]]:
    jobs: List[Dict] = []
    meta: List[Dict] = []
    profiles: Dict[Tuple, Dict] = {}

    def add(job: Dict, row: Dict) -> None:
        jobs.append(job)
        meta.append(row)

    for bundle in data.bundles:
        N = bundle["n_tokens"]
        ckpt = data.held(bundle)[-1]
        B = data.headline_B(bundle)
        anchor = round(compute_anchor(bundle["name"])["ratio"], 4)
        ratios = sorted(set(COMPUTE_RATIOS) | {anchor})
        for layer in data.layers(bundle):
            E = data.reservations(bundle, layer)
            indices, scores, place, _ = data.tokens(bundle, layer, ckpt)
            steps = [("aligned", b, idx, srcb, indices, scores) for b, (idx, srcb) in enumerate(step_blocks(N, B, 8))]
            if bundle["name"] == "olmoe":
                full = data.olmoe_full(ckpt)
                if full is not None:
                    L = int(layer.split("_")[1])
                    ind_full = full[:, L, :].astype(np.int64)
                    blocks = offset_blocks(len(ind_full), OLMOE_PREFIX, B, 8)
                    if blocks:
                        idx, srcb = blocks[0]
                        steps.append(("different", 0, idx, srcb, ind_full, None))
            for token_source, b, idx, srcb, ind_all, sc_all in steps:
                M_b = block_matrix(ind_all[idx], place, srcb, 8)
                Emat = scale_reservation(E["p95"], B, N)
                R = np.minimum(M_b, np.floor(Emat + 1e-9))
                T = M_b - R
                ind_b = ind_all[idx]
                sc_b = sc_all[idx] if sc_all is not None else None
                profiles[(bundle["name"], layer, token_source, b)] = _tail_profiles(ind_b, sc_b, place, srcb, Emat)
                route = E["mean"]
                base = dict(source=bundle["name"], layer=layer, checkpoint=_ckpt_key(ckpt), B=B, block=b,
                            token_source=token_source, policy="p95", sizing="rate", seed=0,
                            allocator=PRIMARY_ALLOCATOR, semantics="nccl", floor=KERNEL_FLOOR_S, structure="chunked",
                            r=0.0, reserved_slots=float(R.sum()), tail_slots=float(T.sum()))
                is_head = token_source == "aligned"
                r_list = ratios if is_head else [anchor]
                for r in r_list:
                    c = r * WIRE_S_PER_SLOT
                    at_anchor = abs(r - anchor) < 1e-9
                    add({"reserved": R, "tail": T, "route": route, "c": c, "structure": "chunked", "reserved_only": True,
                         "allocator": PRIMARY_ALLOCATOR},
                        dict(base, arm="reserved_only", r=r))
                    for structure in ("monolithic", "chunked"):
                        add({"reserved": R, "tail": T, "route": route, "c": c, "structure": structure, "router": "ecmp",
                             "seed": 0, "allocator": PRIMARY_ALLOCATOR,
                             "deadline": (structure == "chunked" and (at_anchor or r == 0.0))},
                            dict(base, arm="two_class", structure=structure, r=r))
                    if at_anchor:
                        P = {"allocator": PRIMARY_ALLOCATOR}
                        for s_ in range(1, TAIL_SEEDS):
                            for structure in ("monolithic", "chunked"):
                                add({"reserved": R, "tail": T, "route": route, "c": c, "structure": structure, "router": "ecmp",
                                     "seed": s_, "deadline": structure == "chunked", **P},
                                    dict(base, arm="two_class", structure=structure, r=r, seed=s_))
                        add({"reserved": R, "tail": T, "route": route, "c": c, "structure": "chunked", "router": "spray", **P},
                            dict(base, arm="two_class_spray", structure="chunked", r=r))
                        zero = np.zeros_like(M_b)
                        add({"reserved": M_b, "tail": zero, "route": route, "c": c, "structure": "chunked", "reserved_only": True, **P},
                            dict(base, arm="one_class_plan", r=r, reserved_slots=float(M_b.sum()), tail_slots=0.0))
                        add({"reserved": zero, "tail": M_b, "route": route, "c": c, "structure": "chunked", "router": "ecmp", **P},
                            dict(base, arm="full_ecmp", r=r, reserved_slots=0.0, tail_slots=float(M_b.sum())))
                        add({"reserved": zero, "tail": M_b, "route": route, "c": c, "structure": "chunked", "router": "spray", **P},
                            dict(base, arm="full_spray", r=r, reserved_slots=0.0, tail_slots=float(M_b.sum())))
                        if is_head:
                            for structure in ("monolithic", "chunked"):
                                add({"reserved": R, "tail": T, "route": route, "c": c, "structure": structure, "router": "ecmp", "semantics": "rdma", **P},
                                    dict(base, arm="two_class", structure=structure, r=r, semantics="rdma"))
                            add({"reserved": R, "tail": T, "route": route, "c": c, "structure": "chunked", "reserved_only": True, "semantics": "rdma", **P},
                                dict(base, arm="reserved_only", r=r, semantics="rdma"))
                            for fl in KERNEL_FLOOR_SENS:
                                for structure in ("monolithic", "chunked"):
                                    add({"reserved": R, "tail": T, "route": route, "c": c, "structure": structure, "router": "ecmp", "floor": fl, **P},
                                        dict(base, arm="two_class", structure=structure, r=r, floor=fl))
                                add({"reserved": R, "tail": T, "route": route, "c": c, "structure": "chunked", "reserved_only": True, "floor": fl, **P},
                                    dict(base, arm="reserved_only", r=r, floor=fl))
                            Q = {"allocator": SENSITIVITY_ALLOCATOR}
                            for structure in ("monolithic", "chunked"):
                                add({"reserved": R, "tail": T, "route": route, "c": c, "structure": structure, "router": "ecmp",
                                     "deadline": structure == "chunked", **Q},
                                    dict(base, arm="two_class", structure=structure, r=r, allocator=SENSITIVITY_ALLOCATOR))
                            add({"reserved": R, "tail": T, "route": route, "c": c, "structure": "chunked", "reserved_only": True, **Q},
                                dict(base, arm="reserved_only", r=r, allocator=SENSITIVITY_ALLOCATOR))
                            for arm, Rm, Tm in (("one_class_plan", M_b, zero), ("full_ecmp", zero, M_b)):
                                add({"reserved": Rm, "tail": Tm, "route": route, "c": c, "structure": "chunked",
                                     "reserved_only": arm == "one_class_plan", "router": "ecmp", **Q},
                                    dict(base, arm=arm, r=r, allocator=SENSITIVITY_ALLOCATOR,
                                         reserved_slots=float(Rm.sum()), tail_slots=float(Tm.sum())))
    workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    print("  step plans", len(jobs), "workers", workers, flush=True)
    results = _run_plan_jobs_fn(_step_job, jobs, workers)
    rows, dl_rows = [], []
    for row, res in zip(meta, results):
        if res is None:
            continue
        r = dict(row)
        cuts = res.pop("_cuts", None)
        r.update(res)
        rows.append(r)
        if cuts:
            prof = profiles[(row["source"], row["layer"], row["token_source"], row["block"])]
            for cut in cuts:
                common = {k: v for k, v in r.items()}
                common.update({"lateness_frac": cut["lateness_frac"], "lateness_s": cut["lateness_s"],
                               "dispatch_overshoot_s": cut["dispatch_overshoot_s"], "cut_step_makespan_s": cut["cut_step_makespan_s"],
                               "cut_step_lateness_rel": cut["cut_step_makespan_s"] / r["reserved_combine_finish_s"] - 1.0,
                               "dispatch_delivered_frac": float(np.sum(cut["dispatch_quota_bytes"]) / max(np.sum(cut["tail_bytes"]), 1e-9)),
                               "combine_delivered_frac": float(np.sum(cut["combine_delivered_bytes"]) / max(np.sum(cut["combine_bytes"]), 1e-9))})
                for a in _attribute_cut(cut, prof):
                    d = dict(common)
                    d.update(a)
                    dl_rows.append(d)
    _write_csv(out / "step.csv", rows)
    _write_csv(out / "deadline.csv", dl_rows)
    return rows, dl_rows


def _run_plan_jobs_fn(fn, jobs: List[Dict], workers: int) -> List[Optional[Dict]]:
    if workers <= 1 or len(jobs) < 4:
        return [fn(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor

    out: List[Optional[Dict]] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(fn, jobs, chunksize=2)):
            out[i] = res
            if i % 250 == 0:
                print("  plans {}/{}".format(i, len(jobs)), flush=True)
    return out


def _wm(rows: List[Dict], source: str, key: str = "step_makespan_s", **sel) -> float:
    """Whole model: sum over layers, mean over blocks and seeds."""
    part = [r for r in rows if r["source"] == source and all(r.get(k) == v for k, v in sel.items())]
    if not part:
        return 0.0
    totals = []
    for b in sorted({r["block"] for r in part}):
        for s_ in sorted({r["seed"] for r in part}):
            p = [r for r in part if r["block"] == b and r["seed"] == s_]
            if p:
                totals.append(sum(float(r[key]) for r in p))
    return _mean(totals)


def summarize_step(rows: List[Dict], dl_rows: List[Dict]) -> Dict:
    summary: Dict = {"assumptions": {src: compute_anchor(src) for src in ("flame", "olmoe")},
                     "kernel_floor_s": KERNEL_FLOOR_S, "lateness_fracs": list(LATENESS_FRACS)}
    for source in sorted({r["source"] for r in rows}):
        anchor = round(compute_anchor(source)["ratio"], 4)
        S: Dict = {"anchor_ratio": anchor}
        sel0 = dict(token_source="aligned", allocator=PRIMARY_ALLOCATOR, semantics="nccl", floor=KERNEL_FLOOR_S)
        S["primary_allocator"] = PRIMARY_ALLOCATOR
        sweep = {}
        for r in sorted({float(x["r"]) for x in rows if x["source"] == source and x["token_source"] == "aligned" and x["allocator"] == PRIMARY_ALLOCATOR}):
            res_only = _wm(rows, source, arm="reserved_only", r=r, **sel0)
            mono = _wm(rows, source, arm="two_class", structure="monolithic", r=r, seed=0, **sel0)
            chunk = _wm(rows, source, arm="two_class", structure="chunked", r=r, seed=0, **sel0)
            disp_ext = _wm(rows, source, key="tail_dispatch_finish_s", arm="two_class", structure="chunked", r=r, seed=0, **sel0) - \
                _wm(rows, source, key="reserved_dispatch_finish_s", arm="two_class", structure="chunked", r=r, seed=0, **sel0)
            comb_ext = _wm(rows, source, key="tail_combine_finish_s", arm="two_class", structure="chunked", r=r, seed=0, **sel0) - \
                _wm(rows, source, key="reserved_combine_finish_s", arm="two_class", structure="chunked", r=r, seed=0, **sel0)
            e = {"reserved_only_s": res_only, "monolithic_s": mono, "chunked_s": chunk,
                 "ext_monolithic": (mono / res_only - 1) if res_only and mono else 0.0,
                 "ext_chunked": (chunk / res_only - 1) if res_only and chunk else 0.0,
                 "dispatch_side_lateness_s": disp_ext, "combine_side_lateness_s": comb_ext}
            e["tail_hidden_frac"] = (1.0 - (chunk - res_only) / (mono - res_only)) if (mono - res_only) > 1e-9 and chunk else None
            sweep[str(r)] = e
        S["sweep"] = sweep
        A: Dict = {}
        for name, sel in ((PRIMARY_ALLOCATOR, sel0), (SENSITIVITY_ALLOCATOR, dict(sel0, allocator=SENSITIVITY_ALLOCATOR)),
                          ("rdma", dict(sel0, semantics="rdma"))):
            A[name] = {
                "reserved_only_s": _wm(rows, source, arm="reserved_only", r=anchor, **sel),
                "monolithic_s": _wm(rows, source, arm="two_class", structure="monolithic", r=anchor, **sel),
                "chunked_s": _wm(rows, source, arm="two_class", structure="chunked", r=anchor, **sel),
                "one_class_plan_s": _wm(rows, source, arm="one_class_plan", r=anchor, **sel),
                "full_ecmp_s": _wm(rows, source, arm="full_ecmp", r=anchor, **sel),
            }
        for fl in KERNEL_FLOOR_SENS:
            sel = dict(sel0, floor=fl)
            A["floor_{:.0f}us".format(fl * 1e6)] = {
                "reserved_only_s": _wm(rows, source, arm="reserved_only", r=anchor, **sel),
                "monolithic_s": _wm(rows, source, arm="two_class", structure="monolithic", r=anchor, **sel),
                "chunked_s": _wm(rows, source, arm="two_class", structure="chunked", r=anchor, **sel),
            }
        A["controls"] = {
            "one_class_plan_s": _wm(rows, source, arm="one_class_plan", r=anchor, **sel0),
            "full_ecmp_s": _wm(rows, source, arm="full_ecmp", r=anchor, **sel0),
            "full_spray_s": _wm(rows, source, arm="full_spray", r=anchor, **sel0),
            "two_class_spray_s": _wm(rows, source, arm="two_class_spray", r=anchor, **sel0),
        }
        seeds = [x for x in rows if x["source"] == source and x["arm"] == "two_class" and x["structure"] == "chunked"
                 and float(x["r"]) == anchor and x["token_source"] == "aligned" and x["allocator"] == PRIMARY_ALLOCATOR
                 and x["semantics"] == "nccl" and x["floor"] == KERNEL_FLOOR_S]
        if seeds:
            per_seed = [_wm(rows, source, arm="two_class", structure="chunked", r=anchor, seed=s_, **sel0) for s_ in sorted({x["seed"] for x in seeds})]
            A["chunked_seed_spread_s"] = float(np.std(per_seed))
            base_rows = [y for y in rows if y["source"] == source and y["arm"] == "reserved_only" and float(y["r"]) == anchor
                         and y["token_source"] == "aligned" and y["allocator"] == PRIMARY_ALLOCATOR and y["semantics"] == "nccl" and y["floor"] == KERNEL_FLOOR_S]
            A["reserved_dispatch_rel_check"] = max(
                abs(x["reserved_dispatch_finish_s"] - y["reserved_dispatch_finish_s"]) / max(y["reserved_dispatch_finish_s"], 1e-12)
                for x in seeds for y in base_rows if y["layer"] == x["layer"] and y["block"] == x["block"])
            A["reserved_combine_rel_check"] = max(
                abs(x["reserved_combine_finish_s"] - y["reserved_combine_finish_s"]) / max(y["reserved_combine_finish_s"], 1e-12)
                for x in seeds for y in base_rows if y["layer"] == x["layer"] and y["block"] == x["block"])
            A["tail_share"] = _mean(x["tail_slots"] / max(x["tail_slots"] + x["reserved_slots"], 1) for x in seeds)
        diff = [x for x in rows if x["source"] == source and x["token_source"] == "different"]
        if diff:
            seld = dict(sel0, token_source="different")
            A["different_tokens"] = {
                "reserved_only_s": _wm(rows, source, arm="reserved_only", r=anchor, **seld),
                "monolithic_s": _wm(rows, source, arm="two_class", structure="monolithic", r=anchor, **seld),
                "chunked_s": _wm(rows, source, arm="two_class", structure="chunked", r=anchor, **seld),
                "tail_share": _mean(x["tail_slots"] / max(x["tail_slots"] + x["reserved_slots"], 1) for x in diff if x["arm"] == "two_class"),
            }
        S["anchor"] = A
        D: Dict = {}
        for tsrc in sorted({x["token_source"] for x in dl_rows if x["source"] == source}):
            for alloc in sorted({x["allocator"] for x in dl_rows if x["source"] == source and x["token_source"] == tsrc}):
                for r in sorted({float(x["r"]) for x in dl_rows if x["source"] == source and x["token_source"] == tsrc and x["allocator"] == alloc}):
                    key = "{}|{}|r={}".format(tsrc, alloc, r)
                    curve = {}
                    sel = [x for x in dl_rows if x["source"] == source and x["token_source"] == tsrc and x["allocator"] == alloc and float(x["r"]) == r]
                    for frac in sorted({float(x["lateness_frac"]) for x in sel}):
                        part = [x for x in sel if float(x["lateness_frac"]) == frac]
                        ent = {
                            "dispatch_delivered_frac": _mean(float(x["dispatch_delivered_frac"]) for x in part),
                            "combine_delivered_frac": _mean(float(x["combine_delivered_frac"]) for x in part),
                            "max_dispatch_overshoot_rel": max(0.0, max(float(x["dispatch_overshoot_s"]) / max(float(x["reserved_dispatch_finish_s"]), 1e-12) for x in part)),
                            "cut_step_lateness_rel_mean": _mean(float(x["cut_step_lateness_rel"]) for x in part),
                            "cut_step_lateness_rel_max": max(float(x["cut_step_lateness_rel"]) for x in part),
                            "orders": {},
                        }
                        for order in sorted({x["order"] for x in part}):
                            for attr in ("best", "random"):
                                pp = [x for x in part if x["order"] == order and x["attribution"] == attr]
                                if not pp:
                                    continue
                                costs = [float(x["cost_per_token"]) for x in pp if x.get("cost_per_token") not in (None, "", "None")]
                                hist = np.mean([json.loads(x["dropped_pos_hist"]) for x in pp], axis=0)
                                ent["orders"]["{}_{}".format(order, attr)] = {
                                    "dropped_frac_of_tail_slots": _mean(float(x["dropped_frac_of_tail_slots"]) for x in pp),
                                    "dropped_frac_of_all_slots": _mean(float(x["dropped_frac_of_all_slots"]) for x in pp),
                                    "cost_per_token": _mean(costs) if costs else None,
                                    "cost_per_token_seed_std": float(np.std([np.mean([float(x["cost_per_token"]) for x in pp if x["seed"] == s_])
                                                                              for s_ in sorted({x["seed"] for x in pp})])) if costs else None,
                                    "dropped_pos_hist": hist.tolist(),
                                }
                        curve[str(frac)] = ent
                    D[key] = curve
        S["deadline"] = D
        summary[source] = S
    return summary
