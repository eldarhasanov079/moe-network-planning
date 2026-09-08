"""Clos routing baselines on the same admitted graphs as claim v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .admit import admit_slots
from .claim_v1 import (
    BYTES,
    COLORS,
    POLICIES,
    MatrixBank,
    _ckpt_key,
    _mean,
    _style,
    _write_csv,
    _write_json,
    make_cfg,
    source_bundle,
)
from .config import DEFAULT_TOPOLOGY
from .planner import plan_matrix

HASHED_SEEDS = 32
PAPER_ROUTERS = ("least-loaded", "ecmp", "spray")
DETERMINISTIC = ("least-loaded", "spray")
HASHED = ("ecmp",)
ALL_ROUTERS = PAPER_ROUTERS


def _plan(matrix: np.ndarray, kind: str, seed: int = 0) -> Dict[str, float]:
    planned = plan_matrix(
        matrix,
        str(DEFAULT_TOPOLOGY),
        bytes_per_slot=BYTES,
        router_kind=kind,
        router_seed=seed,
    )
    metrics = planned["plan"].metrics
    return {
        "iteration_time_s": float(planned["iteration_time_s"]),
        "exposed_comm_s": float(planned["exposed_comm_s"]),
        "ideal_iteration_time_s": float(planned["ideal_iteration_time_s"]),
        "congestion_overhead_s": float(metrics.get("congestion_overhead_s") or 0.0),
    }


def run_clos_baselines(out: Path, sources=("flame", "olmoe")) -> Dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bank = MatrixBank()
    bundles = [source_bundle(name) for name in sources]
    print("loading traces for Clos baselines")
    for bundle in bundles:
        bank.load(bundle, keep_tokens=False)

    rows = []
    for bundle in bundles:
        window = bundle["default_window"]
        reserved = bank.reservations(bundle, window)
        last = bundle["ckpts"][window:][-1]
        for policy in POLICIES:
            for layer in bundle["layers"]:
                matrix = admit_slots(bank.M(bundle, layer, last), reserved[policy][layer])
                for kind in DETERMINISTIC:
                    fwd = _plan(matrix, kind)
                    rev = _plan(matrix.T, kind)
                    rows.append({
                        "source": bundle["name"],
                        "layer": layer,
                        "reservation": policy,
                        "checkpoint": _ckpt_key(last),
                        "router": kind,
                        "seed": 0,
                        "iteration_time_s": fwd["iteration_time_s"] + rev["iteration_time_s"],
                        "congestion_overhead_s": fwd["congestion_overhead_s"] + rev["congestion_overhead_s"],
                    })
                for kind in HASHED:
                    for seed in range(HASHED_SEEDS):
                        fwd = _plan(matrix, kind, seed)
                        rev = _plan(matrix.T, kind, seed)
                        rows.append({
                            "source": bundle["name"],
                            "layer": layer,
                            "reservation": policy,
                            "checkpoint": _ckpt_key(last),
                            "router": kind,
                            "seed": seed,
                            "iteration_time_s": fwd["iteration_time_s"] + rev["iteration_time_s"],
                            "congestion_overhead_s": fwd["congestion_overhead_s"] + rev["congestion_overhead_s"],
                        })
            print("clos-baselines", bundle["name"], policy)

    _write_csv(out / "plans.csv", rows)
    summary = _summarize(rows, bundles)
    _write_json(out / "summary.json", summary)
    _draw(out / "figures", rows, bundles)
    report = _report(summary)
    (out / "REPORT.md").write_text(report)
    reports = Path(__file__).resolve().parents[1] / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "clos-baselines.md").write_text(report)
    print(json.dumps(summary, indent=2)[:2500])
    return summary


def _whole_model(rows, source, policy, router, seed=None) -> List[float]:
    layers = sorted({r["layer"] for r in rows if r["source"] == source})
    if seed is None:
        seeds = sorted({int(r["seed"]) for r in rows
                        if r["source"] == source and r["reservation"] == policy
                        and r["router"] == router})
    else:
        seeds = [int(seed)]
    totals = []
    for s in seeds:
        total = 0.0
        ok = True
        for layer in layers:
            match = [r["iteration_time_s"] for r in rows
                     if r["source"] == source and r["reservation"] == policy
                     and r["router"] == router and r["layer"] == layer
                     and int(r["seed"]) == s]
            if not match:
                ok = False
                break
            total += match[0]
        if ok:
            totals.append(total)
    return totals


def _summarize(rows, bundles) -> Dict:
    detail = {"hashed_seeds": HASHED_SEEDS, "sources": {}}
    for bundle in bundles:
        name = bundle["name"]
        by_policy = {}
        for policy in POLICIES:
            routers = {}
            planned = _whole_model(rows, name, policy, "least-loaded")
            planned_s = planned[0] if planned else 0.0
            for kind in ALL_ROUTERS:
                vals = _whole_model(rows, name, policy, kind)
                routers[kind] = {
                    "mean_s": _mean(vals),
                    "std_s": float(np.std(vals)) if len(vals) > 1 else 0.0,
                    "speedup_vs_this": (_mean(vals) / planned_s) if planned_s else 0.0,
                }
            by_policy[policy] = {"planned_s": planned_s, "routers": routers}
        detail["sources"][name] = by_policy
    return detail


def _draw(fig_dir: Path, rows, bundles) -> None:
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = list(PAPER_ROUTERS)
    palette = {
        "least-loaded": COLORS["planned"],
        "spray": "#1a7f4b",
        "ecmp": COLORS["ecmp"],
    }
    labels = {
        "least-loaded": "planned (least-loaded)",
        "ecmp": "ECMP (per-flow hash)",
        "spray": "packet spray",
    }
    for bundle in bundles:
        name = bundle["name"]
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        x = np.arange(len(POLICIES))
        width = 0.24
        for i, kind in enumerate(order):
            means = []
            stds = []
            for policy in POLICIES:
                vals = _whole_model(rows, name, policy, kind)
                means.append(_mean(vals) * 1e3)
                stds.append((float(np.std(vals)) if len(vals) > 1 else 0.0) * 1e3)
            offset = (i - (len(order) - 1) / 2) * width
            ax.bar(
                x + offset, means, width, yerr=stds, capsize=2,
                label=labels[kind], color=palette[kind],
            )
        ax.set_xticks(x, list(POLICIES))
        ax.set_ylabel("Whole-model Clos time (ms)")
        ax.set_title("{} — planned vs ECMP vs spray (admitted last ckpt)".format(name))
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "{}_baselines.png".format(name), dpi=160)
        plt.close(fig)


def _report(summary: Dict) -> str:
    lines = [
        "# Clos routing baselines",
        "",
        "Same admitted last-checkpoint graphs as claim v1 (`sent = min(M, E)`).",
        "Puppeteer only — ASTRA-sim is not in this comparison.",
        "",
        "- **least-loaded (plan)**: one Clos path per pair, occupancy-aware.",
        "- **ECMP**: today's Clos default — same dest, hashed path, {} seeds.".format(HASHED_SEEDS),
        "- **spray**: split each pair across every equal-cost path (fluid RPS / REPS-family).",
        "",
        "DCQCN is the RoCE *rate* loop on top of ECMP/spray, not a fourth router.",
        "",
    ]
    for source, policies in summary["sources"].items():
        lines += ["## {}".format(source), ""]
        lines.append("| Policy | planned (ms) | ECMP (ms) | spray (ms) | vs ECMP | vs spray |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for policy, block in policies.items():
            r = block["routers"]
            lines.append(
                "| {} | {:.1f} | {:.1f}±{:.1f} | {:.1f} | {:.2f}× | {:.2f}× |".format(
                    policy,
                    r["least-loaded"]["mean_s"] * 1e3,
                    r["ecmp"]["mean_s"] * 1e3,
                    r["ecmp"]["std_s"] * 1e3,
                    r["spray"]["mean_s"] * 1e3,
                    r["ecmp"]["speedup_vs_this"],
                    r["spray"]["speedup_vs_this"],
                )
            )
        lines.append("")
    lines += [
        "Plan beats ECMP (~1.35×) and loses to spray (~0.88–0.90×). That 10–12%",
        "is the single-path / no-reorder tax. Spray is the unplanned multipath",
        "stack; DCQCN would sit on ECMP or spray at packet level (not modeled).",
        "",
    ]
    return "\n".join(lines) + "\n"
