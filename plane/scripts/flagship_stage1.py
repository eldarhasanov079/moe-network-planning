"""Stage 1 flagship figures from claim-v1 + clos-baselines."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLAIM = ROOT / "plane/output/claim-v1/summary.json"
CLOS = ROOT / "plane/output/clos-baselines/summary.json"
PLANS = ROOT / "plane/output/clos-baselines/plans.csv"
OUT = ROOT / "plane/output/stage1-flagship"
REPORTS = ROOT / "plane/reports"

POLICIES = ("uniform", "mean", "p95", "worst")
LINK_GBPS = 100.0
C = {
    "uniform": "#4a4a4a",
    "mean": "#2f6fed",
    "p95": "#c45c26",
    "worst": "#1a7f4b",
    "live": "#111",
    "planned": "#2f6fed",
    "ecmp": "#c45c26",
    "spray": "#1a7f4b",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222",
        "axes.grid": True,
        "grid.color": "#ddd",
        "grid.linewidth": 0.5,
        "font.size": 10,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _label_bars(ax, bars, texts, dy=0.0, fontsize=7.5) -> None:
    for bar, text in zip(bars, texts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            text,
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _hbar_groups(ax, categories, series, colors, xlabel, fmt="{:.0f}") -> None:
    n = len(categories)
    k = len(series)
    y = np.arange(n)
    h = 0.72 / k
    xmax = max(max(vals) for vals in series.values())
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - (k - 1) / 2) * h
        bars = ax.barh(y + offset, vals, h, color=colors[name], label=name)
        for bar, val in zip(bars, vals):
            ax.text(
                val + xmax * 0.012,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(val),
                va="center",
                ha="left",
                fontsize=9,
            )
    ax.set_yticks(y, categories)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, xmax * 1.22)
    ax.invert_yaxis()


def _clos_congestion() -> dict:
    """Whole-model congestion_s by source / policy / router (mean over seeds)."""
    layers = defaultdict(set)
    acc = defaultdict(list)
    with PLANS.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["source"], row["reservation"], row["router"], row["layer"], int(row["seed"]))
            acc[key].append(float(row["congestion_overhead_s"]))
            layers[row["source"]].add(row["layer"])
    per_seed = defaultdict(float)
    for (src, pol, rt, layer, seed), vals in acc.items():
        per_seed[(src, pol, rt, seed)] += vals[0]
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (src, pol, rt, seed), total in per_seed.items():
        out[src][pol][rt].append(total)
    means = {}
    for src in out:
        means[src] = {}
        for pol in POLICIES:
            means[src][pol] = {
                rt: float(np.mean(out[src][pol][rt])) for rt in ("least-loaded", "ecmp", "spray")
            }
    return means


def main() -> None:
    claim = json.loads(CLAIM.read_text())
    clos = json.loads(CLOS.read_text())
    cong = _clos_congestion()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    _style()
    _fig1(claim)
    _fig2(claim)
    _fig3(clos)
    _fig4(claim, cong)
    print("wrote", OUT / "figures")


def _fig1(claim) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    for col, src, title in ((0, "flame", "FLAME-290M"), (1, "olmoe", "OLMoE-1B-7B")):
        block = claim["admitted_live"][src]
        tok = [block["policies"][p]["token_drop_pct"] for p in POLICIES]
        slot = [block["policies"][p]["slot_drop_pct"] for p in POLICIES]
        ax = axes[0][col]
        x = np.arange(len(POLICIES))
        w = 0.38
        b1 = ax.bar(x - w / 2, tok, w, color="#c45c26", label="tokens losing ≥1 dest")
        b2 = ax.bar(x + w / 2, slot, w, color="#4a4a4a", label="slots dropped")
        _label_bars(ax, b1, ["{:.1f}".format(v) for v in tok])
        _label_bars(ax, b2, ["{:.2f}".format(v) for v in slot])
        ax.set_xticks(x, POLICIES)
        ax.set_ylabel("Drop if overflow discarded (%)")
        ax.set_title("{} — admission".format(title))
        ax.set_ylim(0, max(tok) * 1.18)
        if col == 1:
            ax.legend(fontsize=8, loc="upper right")

        ax = axes[1][col]
        live = block["live_avg_cycles"] / 1e6
        adm = [block["policies"][p]["admitted_avg_cycles"] / 1e6 for p in POLICIES]
        bars = ax.bar(POLICIES, adm, color=[C[p] for p in POLICIES])
        ax.axhline(live, color=C["live"], linestyle="--", linewidth=1.4, label="unadmitted live")
        ax.set_ylabel("Whole-model comm (M cycles)")
        ax.set_title("{} — admitted vs unadmitted live".format(title))
        lo = min(adm + [live]) * 0.988
        hi = max(adm + [live]) * 1.014
        ax.set_ylim(lo, hi)
        ax.text(
            3.45, live, "unadmitted live\n{:.2f}M".format(live),
            ha="left", va="center", fontsize=7.5, color=C["live"],
        )
        ax.legend(fontsize=8, loc="lower right")
        for bar, val, p in zip(bars, adm, POLICIES):
            ov = block["policies"][p]["overhead_vs_live_pct"]
            ax.text(
                bar.get_x() + bar.get_width() / 2, val,
                "{:.2f}M\n{:+.2f}%".format(val, ov),
                ha="center", va="bottom", fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig1_admission.png", dpi=170)
    plt.close(fig)


def _fig2(claim) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=False)
    for ax, src, title, default_k in (
        (axes[0], "flame", "FLAME-290M", 4),
        (axes[1], "olmoe", "OLMoE-1B-7B", 2),
    ):
        pols = claim["freeze_sweep"][src]["policies"]
        for name in ("uniform", "mean", "p95"):
            curve = pols[name]["freeze_curve"]
            xs = [c["freeze"] for c in curve]
            ys = [c["overflow_ratio_pct"] for c in curve]
            ax.plot(xs, ys, marker="o", color=C[name], label=name)
            for x, y in zip(xs, ys):
                ax.annotate(
                    "{:.2f}".format(y), (x, y),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=6.5, color=C[name],
                )
        ax.axvline(default_k, color="#888", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Freeze after K checkpoints")
        ax.set_title("{} — held-out overflow (not vs live)".format(title))
        ax.set_xticks(xs)
        ymax = max(
            c["overflow_ratio_pct"]
            for n in ("uniform", "mean", "p95")
            for c in pols[n]["freeze_curve"]
        )
        ax.set_ylim(0, ymax * 1.22)
        ax.text(default_k, ymax * 1.12, "default K={}".format(default_k),
                ha="center", fontsize=8, color="#666")
    axes[0].set_ylabel("Overflow of later M vs frozen E (%)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig2_freeze.png", dpi=170)
    plt.close(fig)


def _fig3(clos) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for ax, src, title in ((axes[0], "flame", "FLAME-290M"), (axes[1], "olmoe", "OLMoE-1B-7B")):
        planned, ecmp, spray = [], [], []
        for p in POLICIES:
            r = clos["sources"][src][p]["routers"]
            planned.append(r["least-loaded"]["mean_s"] * 1e3)
            ecmp.append(r["ecmp"]["mean_s"] * 1e3)
            spray.append(r["spray"]["mean_s"] * 1e3)
        _hbar_groups(
            ax,
            list(POLICIES),
            {"planned (least-loaded)": planned, "ECMP (32 seeds)": ecmp, "packet spray": spray},
            {
                "planned (least-loaded)": C["planned"],
                "ECMP (32 seeds)": C["ecmp"],
                "packet spray": C["spray"],
            },
            "Clos completion time (ms)",
        )
        ax.set_title("{} — same dest, different path".format(title))
        if src == "olmoe":
            ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig3_clos.png", dpi=170)
    plt.close(fig)


def _fig4(claim, cong) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for ax, src, title in ((axes[0], "flame", "FLAME-290M"), (axes[1], "olmoe", "OLMoE-1B-7B")):
        block = claim["admitted_live"][src]
        ov = [block["policies"][p]["overflow_ratio_pct"] for p in POLICIES]
        waste = [block["policies"][p]["waste_ratio_pct"] for p in POLICIES]
        _hbar_groups(
            ax,
            list(POLICIES),
            {"overflow (unreserved / M)": ov, "waste (idle reserved / E)": waste},
            {"overflow (unreserved / M)": "#a33", "waste (idle reserved / E)": "#2f6fed"},
            "Slot share (%)",
            fmt="{:.2f}",
        )
        ax.set_title("{} — reserved-bandwidth cost".format(title))
        if src == "olmoe":
            ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig4a_reserve.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for ax, src, title in ((axes[0], "flame", "FLAME-290M"), (axes[1], "olmoe", "OLMoE-1B-7B")):
        plan = [cong[src][p]["least-loaded"] * 1e3 for p in POLICIES]
        ecmp = [cong[src][p]["ecmp"] * 1e3 for p in POLICIES]
        spray = [cong[src][p]["spray"] * 1e3 for p in POLICIES]
        _hbar_groups(
            ax,
            list(POLICIES),
            {"planned": plan, "ECMP": ecmp, "spray": spray},
            {"planned": C["planned"], "ECMP": C["ecmp"], "spray": C["spray"]},
            "Congestion wait (ms)  ·  100 GB/s Clos links",
        )
        ax.set_title("{} — link contention".format(title))
        if src == "olmoe":
            ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig4b_links.png", dpi=170)
    plt.close(fig)

    old = OUT / "figures" / "fig4_cost.png"
    if old.exists():
        old.unlink()


if __name__ == "__main__":
    main()
