"""Three experiments that make the frozen-plan claim defensible."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .admit import admit_slots, token_drop_stats
from .astrasim import run_phased, run_phased_jobs, write_switch_yaml
from .config import (
    DEFAULT_TOPOLOGY,
    FeederConfig,
    flame_checkpoints,
    flame_layers,
    olmoe_checkpoints,
    olmoe_layers,
)
from .loaders import load_tokens
from .matrix import build_matrix, maps
from .planner import plan_matrix
from .policy import collapse_history, score_heldout

POLICIES = ("uniform", "mean", "p95", "worst")
BYTES = 2048
ECMP_SEEDS = 64
SWITCH_BW = 100.0
FLOOR_BW = 1_000_000.0
SWITCH_LAT = 500.0
COLORS = {
    "uniform": "#4a4a4a",
    "mean": "#2f6fed",
    "p95": "#c45c26",
    "worst": "#1a7f4b",
    "live": "#111111",
    "replan": "#7b2d8e",
    "floor": "#888888",
    "planned": "#2f6fed",
    "ecmp": "#c45c26",
}


def _ckpt_key(ckpt) -> str:
    return str(ckpt)


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222",
        "axes.grid": True,
        "grid.color": "#ddd",
        "grid.linewidth": 0.6,
        "font.size": 11,
        "axes.titlesize": 13,
        "legend.frameon": False,
    })


def source_bundle(name: str) -> Dict:
    if name == "flame":
        return {
            "name": "flame",
            "source": "flame",
            "model": "flame-moe-290m",
            "n_tokens": 250_000,
            "layers": flame_layers("flame-moe-290m"),
            "ckpts": flame_checkpoints("flame-moe-290m"),
            "default_window": 4,
        }
    if name == "olmoe":
        return {
            "name": "olmoe",
            "source": "olmoe",
            "model": "olmoe",
            "n_tokens": 205_000,
            "layers": olmoe_layers(),
            "ckpts": olmoe_checkpoints(),
            "default_window": 2,
        }
    raise ValueError("unknown source {!r}".format(name))


def make_cfg(bundle: Dict, layer: str) -> FeederConfig:
    return FeederConfig(
        source=bundle["source"],
        model=bundle["model"],
        layer=layer,
        n_tokens=bundle["n_tokens"],
        traffic_lens="membership",
        deflection="none",
        bytes_per_slot=BYTES,
        topology=str(DEFAULT_TOPOLOGY),
    )


def whole_phases(layer_mats: Sequence[np.ndarray]) -> List[np.ndarray]:
    phases = []
    for matrix in layer_mats:
        phases.append(np.asarray(matrix, dtype=np.float64))
        phases.append(np.asarray(matrix, dtype=np.float64).T.copy())
    return phases


class MatrixBank:
    """All (source, layer, checkpoint) dispatch matrices, loaded once."""

    def __init__(self) -> None:
        self.matrices: Dict[Tuple[str, str, str], np.ndarray] = {}
        self.tokens: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.scores: Dict[Tuple[str, str, str], Optional[np.ndarray]] = {}

    def load(self, bundle: Dict, keep_tokens: bool = False, keep_scores: bool = False) -> None:
        for layer in bundle["layers"]:
            cfg = make_cfg(bundle, layer)
            for ckpt in bundle["ckpts"]:
                key = (bundle["name"], layer, _ckpt_key(ckpt))
                if key in self.matrices and (not keep_scores or key in self.scores):
                    continue
                scores, indices = load_tokens(cfg, ckpt)
                place, src = maps(cfg, len(indices))
                self.matrices[key] = build_matrix(cfg, indices, place, src)
                if keep_tokens:
                    self.tokens[key] = (indices, place, src)
                if keep_scores:
                    self.scores[key] = scores
            print("  loaded", bundle["name"], layer)

    def M(self, bundle: Dict, layer: str, ckpt) -> np.ndarray:
        return self.matrices[(bundle["name"], layer, _ckpt_key(ckpt))]

    def history(self, bundle: Dict, layer: str) -> List[np.ndarray]:
        return [self.M(bundle, layer, ckpt) for ckpt in bundle["ckpts"]]

    def reservations(self, bundle: Dict, window: int) -> Dict[str, Dict[str, np.ndarray]]:
        out = {policy: {} for policy in POLICIES}
        for layer in bundle["layers"]:
            hist = self.history(bundle, layer)[:window]
            for policy in POLICIES:
                out[policy][layer] = collapse_history(hist, policy)
        return out


def _network(work: Path, bandwidth: float) -> Path:
    return write_switch_yaml(
        work / "network_{:g}G.yml".format(bandwidth),
        npus=8,
        bandwidth=bandwidth,
        latency=SWITCH_LAT,
    )


def validate_phased(work: Path, cache: Path) -> Dict[str, int]:
    """Sanity: two sequential phases should cost about the sum of the singles."""
    matrix = np.full((8, 8), 8000.0)
    np.fill_diagonal(matrix, 0.0)
    switch = _network(work, SWITCH_BW)
    one = run_phased(
        [matrix], work, "val_one", switch,
        bytes_per_slot=BYTES, cache_dir=cache, bandwidth=SWITCH_BW,
    )
    two = run_phased(
        [matrix, matrix.T.copy()], work, "val_two", switch,
        bytes_per_slot=BYTES, cache_dir=cache, bandwidth=SWITCH_BW,
    )
    print("phased validation: one={} two={} ratio={:.3f}".format(one, two, two / one if one else 0))
    return {"one_phase": one, "two_phase": two}


def run_claim_v1(out: Path, sources: Sequence[str] = ("flame", "olmoe")) -> Dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache" / "astrasim"
    work = out / "astrasim_work"
    work.mkdir(parents=True, exist_ok=True)
    switch = _network(work, SWITCH_BW)
    floor_net = _network(work, FLOOR_BW)
    workers = min(6, os.cpu_count() or 2)

    bank = MatrixBank()
    bundles = [source_bundle(name) for name in sources]
    print("loading traces")
    for bundle in bundles:
        bank.load(bundle, keep_tokens=True)

    validation = validate_phased(work / "validate", cache)
    _write_json(out / "phased_validation.json", validation)

    jobs = []
    job_meta = []

    def enqueue(phases, name, bandwidth, network, meta):
        for phase_i, matrix in enumerate(phases):
            phase_name = "{}_p{}".format(name, phase_i)
            jobs.append({
                "phases": [matrix],
                "work": str(work / phase_name),
                "name": phase_name,
                "network": str(network),
                "bytes_per_slot": BYTES,
                "cache_dir": str(cache),
                "bandwidth": bandwidth,
                "latency": SWITCH_LAT,
            })
            row = dict(meta)
            row["group"] = name
            row["phase"] = phase_i
            job_meta.append(row)

    for bundle in bundles:
        window = bundle["default_window"]
        held = list(bundle["ckpts"][window:])
        reserved = bank.reservations(bundle, window)
        for ckpt in held:
            live = [bank.M(bundle, layer, ckpt) for layer in bundle["layers"]]
            enqueue(
                whole_phases(live),
                "{}_live_{}".format(bundle["name"], _ckpt_key(ckpt)),
                SWITCH_BW,
                switch,
                {"exp": "admitted", "kind": "live", "source": bundle["name"],
                 "checkpoint": _ckpt_key(ckpt), "reservation": "live"},
            )
            if ckpt == held[-1]:
                enqueue(
                    whole_phases(live),
                    "{}_floor_{}".format(bundle["name"], _ckpt_key(ckpt)),
                    FLOOR_BW,
                    floor_net,
                    {"exp": "admitted", "kind": "floor", "source": bundle["name"],
                     "checkpoint": _ckpt_key(ckpt), "reservation": "floor"},
                )
            for policy in POLICIES:
                admitted = [admit_slots(bank.M(bundle, layer, ckpt), reserved[policy][layer])
                            for layer in bundle["layers"]]
                enqueue(
                    whole_phases(admitted),
                    "{}_adm_{}_{}".format(bundle["name"], policy, _ckpt_key(ckpt)),
                    SWITCH_BW,
                    switch,
                    {"exp": "admitted", "kind": "admitted", "source": bundle["name"],
                     "checkpoint": _ckpt_key(ckpt), "reservation": policy},
                )

        n_ckpts = len(bundle["ckpts"])
        prefix_E = {layer: {} for layer in bundle["layers"]}
        for layer in bundle["layers"]:
            hist = bank.history(bundle, layer)
            for freeze in range(1, n_ckpts):
                prefix_E[layer][freeze] = {
                    policy: collapse_history(hist[:freeze], policy) for policy in POLICIES
                }

        for freeze in range(1, n_ckpts):
            for t, ckpt in enumerate(bundle["ckpts"]):
                if t < freeze:
                    continue
                for policy in POLICIES:
                    admitted = [
                        admit_slots(bank.M(bundle, layer, ckpt), prefix_E[layer][freeze][policy])
                        for layer in bundle["layers"]
                    ]
                    enqueue(
                        whole_phases(admitted),
                        "{}_fr{}_{}_{}".format(bundle["name"], freeze, policy, _ckpt_key(ckpt)),
                        SWITCH_BW,
                        switch,
                        {"exp": "sweep", "kind": "freeze", "source": bundle["name"],
                         "freeze": freeze, "checkpoint": _ckpt_key(ckpt),
                         "t": t, "reservation": policy},
                    )

        for t, ckpt in enumerate(bundle["ckpts"]):
            if t < 1:
                continue
            for policy in POLICIES:
                admitted = [
                    admit_slots(bank.M(bundle, layer, ckpt), prefix_E[layer][t][policy])
                    for layer in bundle["layers"]
                ]
                enqueue(
                    whole_phases(admitted),
                    "{}_rp_{}_{}".format(bundle["name"], policy, _ckpt_key(ckpt)),
                    SWITCH_BW,
                    switch,
                    {"exp": "sweep", "kind": "replan", "source": bundle["name"],
                     "freeze": t, "checkpoint": _ckpt_key(ckpt),
                     "t": t, "reservation": policy},
                )
            live = [bank.M(bundle, layer, ckpt) for layer in bundle["layers"]]
            enqueue(
                whole_phases(live),
                "{}_swlive_{}".format(bundle["name"], _ckpt_key(ckpt)),
                SWITCH_BW,
                switch,
                {"exp": "sweep", "kind": "live", "source": bundle["name"],
                 "freeze": 0, "checkpoint": _ckpt_key(ckpt),
                 "t": t, "reservation": "live"},
            )

    print("ASTRA-sim jobs", len(jobs), "workers", workers)
    finished = run_phased_jobs(jobs, workers=workers)
    grouped = {}
    for meta, result in zip(job_meta, finished):
        group = meta["group"]
        entry = grouped.setdefault(group, {"cycles": 0, "meta": None})
        entry["cycles"] += int(result["cycles"])
        if entry["meta"] is None:
            entry["meta"] = {k: v for k, v in meta.items() if k not in ("group", "phase")}
    sim_rows = []
    for group, entry in grouped.items():
        row = dict(entry["meta"])
        row["group"] = group
        row["cycles"] = entry["cycles"]
        sim_rows.append(row)
    _write_csv(out / "astrasim_jobs.csv", sim_rows)

    admitted_rows, drop_rows = _score_admitted(bank, bundles)
    _write_csv(out / "admitted-live" / "overflow.csv", admitted_rows)
    _write_csv(out / "admitted-live" / "token_drops.csv", drop_rows)

    sweep_rows = _score_sweep(bank, bundles)
    _write_csv(out / "freeze-sweep" / "overflow.csv", sweep_rows)

    print("planning Clos least-loaded vs ECMP ({} seeds)".format(ECMP_SEEDS))
    clos_rows = _run_clos_ecmp(bank, bundles)
    _write_csv(out / "clos-ecmp" / "plans.csv", clos_rows)

    summaries = {
        "phased_validation": validation,
        "admitted_live": _summarize_admitted(sim_rows, admitted_rows, drop_rows, bundles),
        "freeze_sweep": _summarize_sweep(sim_rows, sweep_rows, bundles),
        "clos_ecmp": _summarize_clos(clos_rows, bundles),
        "note": (
            "ASTRA-sim analytical Switch 8x 100GB/s, sizes only; whole-model "
            "time sums per-layer dispatch+combine. Admitted traffic is "
            "element-wise min(M, E). Clos-vs-ECMP is Puppeteer on the 8-GPU "
            "Clos, not ASTRA-sim."
        ),
    }
    _write_json(out / "summary.json", summaries)
    _draw_all(out, sim_rows, admitted_rows, drop_rows, sweep_rows, clos_rows, bundles)
    report = render_report(summaries)
    (out / "REPORT.md").write_text(report)
    reports = Path(__file__).resolve().parents[1] / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "claim-v1.md").write_text(report)
    print(json.dumps({k: (v if k == "note" else {kk: vv for kk, vv in v.items() if kk != "detail"})
                      for k, v in summaries.items() if k != "phased_validation"}, indent=2, default=str)[:2000])
    return summaries


def _score_admitted(bank: MatrixBank, bundles: List[Dict]):
    overflow_rows = []
    drop_rows = []
    for bundle in bundles:
        window = bundle["default_window"]
        reserved = bank.reservations(bundle, window)
        for ckpt in bundle["ckpts"][window:]:
            for layer in bundle["layers"]:
                actual = bank.M(bundle, layer, ckpt)
                token_key = (bundle["name"], layer, _ckpt_key(ckpt))
                indices, place, src = bank.tokens[token_key]
                for policy in POLICIES:
                    env = score_heldout(actual, reserved[policy][layer])
                    overflow_rows.append({
                        "source": bundle["name"],
                        "layer": layer,
                        "checkpoint": _ckpt_key(ckpt),
                        "reservation": policy,
                        **env,
                    })
                    drops = token_drop_stats(indices, place, src, reserved[policy][layer])
                    drop_rows.append({
                        "source": bundle["name"],
                        "layer": layer,
                        "checkpoint": _ckpt_key(ckpt),
                        "reservation": policy,
                        **drops,
                    })
        print("scored admitted", bundle["name"])
    return overflow_rows, drop_rows


def _score_sweep(bank: MatrixBank, bundles: List[Dict]):
    rows = []
    for bundle in bundles:
        n_ckpts = len(bundle["ckpts"])
        for layer in bundle["layers"]:
            hist = bank.history(bundle, layer)
            prefix = {}
            for freeze in range(1, n_ckpts):
                prefix[freeze] = {p: collapse_history(hist[:freeze], p) for p in POLICIES}
            for t, ckpt in enumerate(bundle["ckpts"]):
                actual = hist[t]
                if t >= 1:
                    for policy in POLICIES:
                        env = score_heldout(actual, prefix[t][policy])
                        rows.append({
                            "source": bundle["name"], "layer": layer,
                            "kind": "replan", "freeze": t,
                            "checkpoint": _ckpt_key(ckpt), "t": t,
                            "reservation": policy, **env,
                        })
                for freeze in range(1, n_ckpts):
                    if t < freeze:
                        continue
                    for policy in POLICIES:
                        env = score_heldout(actual, prefix[freeze][policy])
                        rows.append({
                            "source": bundle["name"], "layer": layer,
                            "kind": "freeze", "freeze": freeze,
                            "checkpoint": _ckpt_key(ckpt), "t": t,
                            "reservation": policy, **env,
                        })
            print("scored sweep", bundle["name"], layer)
    return rows


_PLAN_CACHE: Dict[Tuple, Dict[str, float]] = {}


def _plan_metrics(matrix: np.ndarray, router_kind: str, seed: int) -> Dict[str, float]:
    key = (np.asarray(matrix, dtype=np.float64).tobytes(), router_kind, int(seed))
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached
    planned = plan_matrix(
        matrix,
        str(DEFAULT_TOPOLOGY),
        bytes_per_slot=BYTES,
        router_kind=router_kind,
        router_seed=seed,
    )
    metrics = planned["plan"].metrics
    payload = {
        "iteration_time_s": float(planned["iteration_time_s"]),
        "exposed_comm_s": float(planned["exposed_comm_s"]),
        "ideal_iteration_time_s": float(planned["ideal_iteration_time_s"]),
        "congestion_overhead_s": float(metrics.get("congestion_overhead_s") or 0.0),
        "contended_links": int(metrics.get("contended_links") or 0),
    }
    _PLAN_CACHE[key] = payload
    return payload


def _run_clos_ecmp(bank: MatrixBank, bundles: List[Dict]) -> List[Dict]:
    rows = []
    for bundle in bundles:
        window = bundle["default_window"]
        reserved = bank.reservations(bundle, window)
        held = list(bundle["ckpts"][window:])
        last = held[-1]
        for policy in POLICIES:
            for layer in bundle["layers"]:
                variants = [
                    ("reservation", reserved[policy][layer]),
                    ("admitted_last", admit_slots(bank.M(bundle, layer, last), reserved[policy][layer])),
                ]
                for variant, matrix in variants:
                    planned = _plan_metrics(matrix, "least-loaded", 0)
                    comb = _plan_metrics(matrix.T, "least-loaded", 0)
                    rows.append({
                        "source": bundle["name"], "layer": layer, "reservation": policy,
                        "variant": variant, "router": "least-loaded", "seed": 0,
                        "iteration_time_s": planned["iteration_time_s"] + comb["iteration_time_s"],
                        "exposed_comm_s": planned["exposed_comm_s"] + comb["exposed_comm_s"],
                        "ideal_iteration_time_s": planned["ideal_iteration_time_s"] + comb["ideal_iteration_time_s"],
                        "congestion_overhead_s": planned["congestion_overhead_s"] + comb["congestion_overhead_s"],
                    })
                    for seed in range(ECMP_SEEDS):
                        hashed = _plan_metrics(matrix, "ecmp", seed)
                        hashed_t = _plan_metrics(matrix.T, "ecmp", seed)
                        rows.append({
                            "source": bundle["name"], "layer": layer, "reservation": policy,
                            "variant": variant, "router": "ecmp", "seed": seed,
                            "iteration_time_s": hashed["iteration_time_s"] + hashed_t["iteration_time_s"],
                            "exposed_comm_s": hashed["exposed_comm_s"] + hashed_t["exposed_comm_s"],
                            "ideal_iteration_time_s": hashed["ideal_iteration_time_s"] + hashed_t["ideal_iteration_time_s"],
                            "congestion_overhead_s": hashed["congestion_overhead_s"] + hashed_t["congestion_overhead_s"],
                        })
            print("clos-ecmp", bundle["name"], policy)
    return rows


def _mean(values) -> float:
    return float(np.mean(values)) if values else 0.0


def _summarize_admitted(sim_rows, overflow_rows, drop_rows, bundles):
    detail = {}
    for bundle in bundles:
        name = bundle["name"]
        held = [_ckpt_key(c) for c in bundle["ckpts"][bundle["default_window"]:]]
        live = [r["cycles"] for r in sim_rows
                if r["exp"] == "admitted" and r["kind"] == "live" and r["source"] == name]
        floor = [r["cycles"] for r in sim_rows
                 if r["exp"] == "admitted" and r["kind"] == "floor" and r["source"] == name]
        live_avg = _mean(live)
        by_policy = {}
        for policy in POLICIES:
            adm = [r["cycles"] for r in sim_rows
                   if r["exp"] == "admitted" and r["kind"] == "admitted"
                   and r["source"] == name and r["reservation"] == policy]
            ov = [float(r["overflow_ratio"]) for r in overflow_rows
                  if r["source"] == name and r["reservation"] == policy]
            waste = [float(r.get("waste_ratio") or r.get("unused_ratio") or 0.0)
                     for r in overflow_rows if r["source"] == name and r["reservation"] == policy]
            drops = [float(r["frac_tokens_dropped"]) for r in drop_rows
                     if r["source"] == name and r["reservation"] == policy]
            slots = [float(r["frac_slots_dropped"]) for r in drop_rows
                     if r["source"] == name and r["reservation"] == policy]
            adm_avg = _mean(adm)
            by_policy[policy] = {
                "admitted_avg_cycles": adm_avg,
                "overhead_vs_live_pct": 100.0 * (adm_avg - live_avg) / live_avg if live_avg else 0.0,
                "overflow_ratio_pct": 100.0 * _mean(ov),
                "waste_ratio_pct": 100.0 * _mean(waste),
                "token_drop_pct": 100.0 * _mean(drops),
                "slot_drop_pct": 100.0 * _mean(slots),
            }
        detail[name] = {
            "profile_window": bundle["default_window"],
            "held_out": held,
            "live_avg_cycles": live_avg,
            "floor_cycles": floor[0] if floor else None,
            "policies": by_policy,
        }
    return detail


def _summarize_sweep(sim_rows, overflow_rows, bundles):
    detail = {}
    for bundle in bundles:
        name = bundle["name"]
        n_ckpts = len(bundle["ckpts"])
        by_policy = {}
        for policy in POLICIES:
            freeze_curve = []
            for freeze in range(1, n_ckpts):
                cyc = [r["cycles"] for r in sim_rows
                       if r["exp"] == "sweep" and r["kind"] == "freeze"
                       and r["source"] == name and r["reservation"] == policy
                       and int(r["freeze"]) == freeze]
                ov = [float(r["overflow_ratio"]) for r in overflow_rows
                      if r["source"] == name and r["kind"] == "freeze"
                      and r["reservation"] == policy and int(r["freeze"]) == freeze]
                freeze_curve.append({
                    "freeze": freeze,
                    "admitted_avg_cycles": _mean(cyc),
                    "overflow_ratio_pct": 100.0 * _mean(ov),
                })
            rp = [r["cycles"] for r in sim_rows
                  if r["exp"] == "sweep" and r["kind"] == "replan"
                  and r["source"] == name and r["reservation"] == policy]
            live = [r["cycles"] for r in sim_rows
                    if r["exp"] == "sweep" and r["kind"] == "live" and r["source"] == name]
            default = bundle["default_window"]
            frozen = next((c for c in freeze_curve if c["freeze"] == default), freeze_curve[-1])
            by_policy[policy] = {
                "replan_avg_cycles": _mean(rp),
                "live_avg_cycles": _mean(live),
                "default_freeze": frozen,
                "gap_default_vs_replan_pct": (
                    100.0 * (frozen["admitted_avg_cycles"] - _mean(rp)) / _mean(rp)
                    if _mean(rp) else 0.0
                ),
                "freeze_curve": freeze_curve,
            }
        detail[name] = {"n_checkpoints": n_ckpts, "policies": by_policy}
    return detail


def _summarize_clos(clos_rows, bundles):
    detail = {}
    for bundle in bundles:
        name = bundle["name"]
        by_policy = {}
        for policy in POLICIES:
            variants = {}
            for variant in ("reservation", "admitted_last"):
                ll = [r["iteration_time_s"] for r in clos_rows
                      if r["source"] == name and r["reservation"] == policy
                      and r["variant"] == variant and r["router"] == "least-loaded"]
                ecmp = [r["iteration_time_s"] for r in clos_rows
                        if r["source"] == name and r["reservation"] == policy
                        and r["variant"] == variant and r["router"] == "ecmp"]
                layers = sorted({r["layer"] for r in clos_rows if r["source"] == name})
                ll_total = 0.0
                ecmp_totals = []
                for seed in range(ECMP_SEEDS):
                    seed_sum = 0.0
                    for layer in layers:
                        match = [r["iteration_time_s"] for r in clos_rows
                                 if r["source"] == name and r["reservation"] == policy
                                 and r["variant"] == variant and r["router"] == "ecmp"
                                 and r["layer"] == layer and int(r["seed"]) == seed]
                        if match:
                            seed_sum += match[0]
                    ecmp_totals.append(seed_sum)
                for layer in layers:
                    match = [r["iteration_time_s"] for r in clos_rows
                             if r["source"] == name and r["reservation"] == policy
                             and r["variant"] == variant and r["router"] == "least-loaded"
                             and r["layer"] == layer]
                    if match:
                        ll_total += match[0]
                variants[variant] = {
                    "planned_s": ll_total,
                    "ecmp_mean_s": _mean(ecmp_totals),
                    "ecmp_std_s": float(np.std(ecmp_totals)) if ecmp_totals else 0.0,
                    "speedup_vs_ecmp": (
                        _mean(ecmp_totals) / ll_total if ll_total else 0.0
                    ),
                }
            by_policy[policy] = variants
        detail[name] = {"ecmp_seeds": ECMP_SEEDS, "policies": by_policy}
    return detail


def _draw_all(out, sim_rows, overflow_rows, drop_rows, sweep_rows, clos_rows, bundles):
    _style()
    _draw_admitted(out / "admitted-live" / "figures", sim_rows, drop_rows, bundles)
    _draw_sweep(out / "freeze-sweep" / "figures", sim_rows, sweep_rows, bundles)
    _draw_clos(out / "clos-ecmp" / "figures", clos_rows, bundles)


def _draw_admitted(fig_dir: Path, sim_rows, drop_rows, bundles):
    fig_dir.mkdir(parents=True, exist_ok=True)
    for bundle in bundles:
        name = bundle["name"]
        held = [_ckpt_key(c) for c in bundle["ckpts"][bundle["default_window"]:]]
        fig, ax = plt.subplots(figsize=(9.2, 4.6))
        xs = list(range(len(held)))
        live = []
        for ckpt in held:
            row = next(r for r in sim_rows if r["exp"] == "admitted" and r["kind"] == "live"
                       and r["source"] == name and r["checkpoint"] == ckpt)
            live.append(row["cycles"] / 1e6)
        ax.plot(xs, live, marker="o", color=COLORS["live"], label="unadmitted live")
        floor = next((r for r in sim_rows if r["exp"] == "admitted" and r["kind"] == "floor"
                      and r["source"] == name), None)
        if floor:
            ax.axhline(floor["cycles"] / 1e6, color=COLORS["floor"], linestyle=":",
                       label="congestion-free floor")
        for policy in POLICIES:
            ys = []
            for ckpt in held:
                row = next(r for r in sim_rows if r["exp"] == "admitted" and r["kind"] == "admitted"
                           and r["source"] == name and r["reservation"] == policy
                           and r["checkpoint"] == ckpt)
                ys.append(row["cycles"] / 1e6)
            ax.plot(xs, ys, marker="o", color=COLORS[policy], label="admitted {}".format(policy))
        ax.set_xticks(xs, held, rotation=30)
        ax.set_ylabel("Whole-model comm (million cycles)")
        ax.set_title("{} — admitted live vs unadmitted (Switch 100 GB/s)".format(name))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "{}_admitted_over_training.png".format(name), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        live_avg = float(np.mean(live))
        names = list(POLICIES)
        vals = []
        for policy in POLICIES:
            adm = [r["cycles"] / 1e6 for r in sim_rows
                   if r["exp"] == "admitted" and r["kind"] == "admitted"
                   and r["source"] == name and r["reservation"] == policy]
            vals.append(100.0 * (float(np.mean(adm)) - live_avg) / live_avg)
        bars = ax.bar(names, vals, color=[COLORS[p] for p in names])
        ax.axhline(0.0, color="#888", linewidth=1)
        ax.set_ylabel("Admitted comm / live comm − 1 (%)")
        ax.set_title("{} — admitted latency vs unadmitted live".format(name))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val, "{:+.2f}%".format(val),
                    ha="center", va="bottom" if val >= 0 else "top", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "{}_admitted_overhead.png".format(name), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        x = np.arange(len(POLICIES))
        width = 0.38
        tok = []
        slot = []
        for policy in POLICIES:
            xs_d = [float(r["frac_tokens_dropped"]) for r in drop_rows
                    if r["source"] == name and r["reservation"] == policy]
            xs_s = [float(r["frac_slots_dropped"]) for r in drop_rows
                    if r["source"] == name and r["reservation"] == policy]
            tok.append(100.0 * _mean(xs_d))
            slot.append(100.0 * _mean(xs_s))
        b1 = ax.bar(x - width / 2, tok, width, label="tokens losing ≥1 slot", color="#c45c26")
        b2 = ax.bar(x + width / 2, slot, width, label="slots dropped", color="#4a4a4a")
        ax.set_xticks(x, list(POLICIES))
        ax.set_ylabel("Drop rate if overflow is discarded (%)")
        ax.set_title("{} — admission cost of the frozen envelope".format(name))
        ax.legend()
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        "{:.2f}".format(bar.get_height()), ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "{}_drops.png".format(name), dpi=160)
        plt.close(fig)


def _draw_sweep(fig_dir: Path, sim_rows, sweep_rows, bundles):
    fig_dir.mkdir(parents=True, exist_ok=True)
    for bundle in bundles:
        name = bundle["name"]
        n_ckpts = len(bundle["ckpts"])
        xs = list(range(1, n_ckpts))
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        for policy in POLICIES:
            ys = []
            for freeze in xs:
                cyc = [r["cycles"] / 1e6 for r in sim_rows
                       if r["exp"] == "sweep" and r["kind"] == "freeze"
                       and r["source"] == name and r["reservation"] == policy
                       and int(r["freeze"]) == freeze]
                ys.append(_mean(cyc))
            ax.plot(xs, ys, marker="o", color=COLORS[policy], label="freeze {}".format(policy))
        for policy, style in (("mean", "--"), ("p95", ":")):
            rp = [r["cycles"] / 1e6 for r in sim_rows
                  if r["exp"] == "sweep" and r["kind"] == "replan"
                  and r["source"] == name and r["reservation"] == policy]
            ax.axhline(_mean(rp), color=COLORS[policy], linestyle=style,
                       label="replan {} ({:.2f}M)".format(policy, _mean(rp)))
        live = [r["cycles"] / 1e6 for r in sim_rows
                if r["exp"] == "sweep" and r["kind"] == "live" and r["source"] == name]
        ax.axhline(_mean(live), color=COLORS["live"], linestyle="-.",
                   label="unadmitted live ({:.2f}M)".format(_mean(live)))
        ax.set_xlabel("Freeze after K checkpoints (expanding history)")
        ax.set_ylabel("Held-out admitted comm (million cycles)")
        ax.set_title("{} — freeze-K vs always-replan (admitted ASTRA-sim)".format(name))
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "{}_freeze_vs_replan_time.png".format(name), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        for policy in POLICIES:
            ys = []
            for freeze in xs:
                ov = [float(r["overflow_ratio"]) for r in sweep_rows
                      if r["source"] == name and r["kind"] == "freeze"
                      and r["reservation"] == policy and int(r["freeze"]) == freeze]
                ys.append(100.0 * _mean(ov))
            ax.plot(xs, ys, marker="o", color=COLORS[policy], label="freeze {}".format(policy))
        for policy, style in (("mean", "--"), ("p95", ":")):
            ov = [float(r["overflow_ratio"]) for r in sweep_rows
                  if r["source"] == name and r["kind"] == "replan" and r["reservation"] == policy]
            ax.axhline(100.0 * _mean(ov), color=COLORS[policy], linestyle=style,
                       label="replan {}".format(policy))
        ax.set_xlabel("Freeze after K checkpoints")
        ax.set_ylabel("Held-out overflow ratio (%)")
        ax.set_title("{} — freeze-K vs always-replan (overflow)".format(name))
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "{}_freeze_vs_replan_overflow.png".format(name), dpi=160)
        plt.close(fig)


def _draw_clos(fig_dir: Path, clos_rows, bundles):
    fig_dir.mkdir(parents=True, exist_ok=True)
    for bundle in bundles:
        name = bundle["name"]
        layers = sorted({r["layer"] for r in clos_rows if r["source"] == name})
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        x = np.arange(len(POLICIES))
        width = 0.36
        planned = []
        ecmp_mean = []
        ecmp_std = []
        for policy in POLICIES:
            ll_total = 0.0
            seed_totals = [0.0] * ECMP_SEEDS
            for layer in layers:
                ll = next(r for r in clos_rows
                          if r["source"] == name and r["reservation"] == policy
                          and r["variant"] == "admitted_last" and r["router"] == "least-loaded"
                          and r["layer"] == layer)
                ll_total += ll["iteration_time_s"]
                for seed in range(ECMP_SEEDS):
                    row = next(r for r in clos_rows
                               if r["source"] == name and r["reservation"] == policy
                               and r["variant"] == "admitted_last" and r["router"] == "ecmp"
                               and r["layer"] == layer and int(r["seed"]) == seed)
                    seed_totals[seed] += row["iteration_time_s"]
            planned.append(ll_total * 1e3)
            ecmp_mean.append(_mean(seed_totals) * 1e3)
            ecmp_std.append(float(np.std(seed_totals)) * 1e3)
        ax.bar(x - width / 2, planned, width, label="Clos least-loaded", color=COLORS["planned"])
        ax.bar(x + width / 2, ecmp_mean, width, yerr=ecmp_std, capsize=3,
               label="Clos ECMP (mean ± std, {} seeds)".format(ECMP_SEEDS), color=COLORS["ecmp"])
        ax.set_xticks(x, list(POLICIES))
        ax.set_ylabel("Whole-model Clos time (ms)")
        ax.set_title("{} — Puppeteer Clos planned vs ECMP (admitted last ckpt)".format(name))
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "{}_planned_vs_ecmp.png".format(name), dpi=160)
        plt.close(fig)


def render_report(summaries: Dict) -> str:
    lines = [
        "# Claim v1 — frozen early plan, no reconfig",
        "",
        "Early profiling produces one per-layer reservation `E`. Later steps send",
        "`min(M, E)` (overflow dropped, no deflection). The claim: that frozen plan",
        "stays close to an always-replan envelope and beats uniform / Clos-ECMP",
        "without mid-training reconfig.",
        "",
        "## Method",
        "",
        "- Sources: FLAME-MoE-290M (8 MoE layers, 250k tokens, 11 ckpts, freeze after 4)",
        "  and OLMoE-1B-7B (16 layers, 205k tokens, 5 ckpts, freeze after 2).",
        "- Reservation policies: uniform, mean, P95, worst.",
        "- ASTRA-sim: analytical congestion-aware **Switch** 8×100 GB/s. Whole-model",
        "  time is the **sum** of per-layer dispatch+combine runs (Exp N convention).",
        "  **Does not consume Clos paths.**",
        "- Congestion-free floor: same live `M`, 1 PB/s Switch.",
        "- Always-replan: expanding window `E_t = collapse(M_0..M_{t-1})`, causal.",
        "- Clos-vs-ECMP: Puppeteer 2-pod / 2-leaf / 2-spine Clos. Least-loaded vs",
        "  64-seed oblivious per-flow hash. Same admitted graphs.",
        "",
        summaries.get("note", ""),
        "",
    ]
    adm = summaries.get("admitted_live", {})
    lines += ["## 1. Admitted-live ASTRA-sim", ""]
    for source, block in adm.items():
        lines.append("### {}".format(source))
        lines.append("")
        lines.append("Live avg: `{:.0f}` cycles. Floor (last ckpt): `{}`.".format(
            block["live_avg_cycles"], block["floor_cycles"]))
        lines.append("")
        lines.append("| Policy | Admitted cycles | vs live | overflow | waste | token drop | slot drop |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for policy, row in block["policies"].items():
            lines.append(
                "| {} | {:.0f} | {:+.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% |".format(
                    policy,
                    row["admitted_avg_cycles"],
                    row["overhead_vs_live_pct"],
                    row["overflow_ratio_pct"],
                    row["waste_ratio_pct"],
                    row["token_drop_pct"],
                    row["slot_drop_pct"],
                )
            )
        lines.append("")

    sweep = summaries.get("freeze_sweep", {})
    lines += ["## 2. Freeze-point vs always-replan", ""]
    for source, block in sweep.items():
        lines.append("### {}".format(source))
        lines.append("")
        lines.append("| Policy | Default-freeze cycles | Replan cycles | gap | default overflow |")
        lines.append("|---|---:|---:|---:|---:|")
        for policy, row in block["policies"].items():
            frozen = row["default_freeze"]
            lines.append(
                "| {} | {:.0f} | {:.0f} | {:+.2f}% | {:.2f}% |".format(
                    policy,
                    frozen["admitted_avg_cycles"],
                    row["replan_avg_cycles"],
                    row["gap_default_vs_replan_pct"],
                    frozen["overflow_ratio_pct"],
                )
            )
        lines.append("")
        lines.append("Freeze-K admitted time (million cycles):")
        lines.append("")
        header = "| K | " + " | ".join(POLICIES) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(POLICIES) + 1))
        n = block["n_checkpoints"]
        for freeze in range(1, n):
            cells = [str(freeze)]
            for policy in POLICIES:
                curve = block["policies"][policy]["freeze_curve"]
                hit = next(c for c in curve if c["freeze"] == freeze)
                cells.append("{:.2f}".format(hit["admitted_avg_cycles"] / 1e6))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    clos = summaries.get("clos_ecmp", {})
    lines += ["## 3. Puppeteer Clos planned vs ECMP", ""]
    lines.append("Whole-model time = sum over layers of (dispatch + combine) Puppeteer iteration time.")
    lines.append("Variant `admitted_last` is live last-checkpoint traffic clipped to the frozen `E`.")
    lines.append("")
    for source, block in clos.items():
        lines.append("### {}".format(source))
        lines.append("")
        lines.append("| Policy | Planned (ms) | ECMP mean (ms) | ECMP std | speedup |")
        lines.append("|---|---:|---:|---:|---:|")
        for policy, variants in block["policies"].items():
            row = variants["admitted_last"]
            lines.append(
                "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f}× |".format(
                    policy,
                    row["planned_s"] * 1e3,
                    row["ecmp_mean_s"] * 1e3,
                    row["ecmp_std_s"] * 1e3,
                    row["speedup_vs_ecmp"],
                )
            )
        lines.append("")

    lines += [
        "## What this does and does not show",
        "",
        "- Shows: a frozen envelope can be *enforced* on later steps with a measured",
        "  drop cost, stays near causal replan, and Clos least-loaded beats Clos ECMP",
        "  on the same graph.",
        "- Does not show: packet-level queues, ASTRA-sim using Clos paths, training",
        "  loss under drops, or mid-run reconfig / deflection (later work).",
        "",
    ]
    return "\n".join(lines) + "\n"
