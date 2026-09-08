"""Redraw Chapter 3 figures from existing CSVs as PDF + SVG."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import plotting  # noqa: E402

P = plotting.PALETTE


def save_both(fig, stem: str) -> None:
    for ext in ("pdf", "svg"):
        plotting.save(fig, OUT / f"{stem}.{ext}", close=False)
    plt.close(fig)


def fig_ladder() -> None:
    b = pd.read_csv(ROOT / "exp_B_routing_stabilization/results/stabilization_layer_avg.csv")
    e = pd.read_csv(ROOT / "exp_E_volume_stabilization/results/volume_stability_avg.csv")
    h_m = pd.read_csv(ROOT / "exp_H_link_stabilization/results/matrix_stability_avg.csv")
    h_l = pd.read_csv(ROOT / "exp_H_link_stabilization/results/link_stability_avg.csv")

    d = pd.read_csv(ROOT / "exp_D_olmoe_saturation/results/olmoe_stabilization_avg.csv")
    e_o = pd.read_csv(ROOT / "exp_E_volume_stabilization/results/olmoe_volume_stability.csv")
    h_mo = pd.read_csv(ROOT / "exp_H_link_stabilization/results/olmoe_matrix_stability_avg.csv")
    h_lo = pd.read_csv(ROOT / "exp_H_link_stabilization/results/olmoe_link_stability_avg.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)

    ax = axes[0]
    x = np.arange(len(b))
    ax.plot(x, b["top6_overlap"], "^:", color=P[1], lw=2.0, label="routing top-6 overlap")
    ax.plot(x, e["cosine_vs_final"], "s--", color=P[2], lw=2.0, label="volume cosine")
    ax.plot(x, h_m["matrix_cosine_vs_final"], "o-", color=P[0], lw=2.2, label="matrix cosine")
    ax.plot(x, h_l["fabric_hot_overlap_vs_final"], "D-", color=P[4], lw=1.6, label="hot-link overlap")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in b["checkpoint"]], rotation=45, fontsize=8)
    ax.set_xlabel("FLAME-290M checkpoint (iteration)")
    ax.set_ylabel("Agreement with final model")
    ax.set_title("FLAME-MoE-290M")
    ax.set_ylim(0.2, 1.05)
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    x = np.arange(len(e_o))
    ax.plot(x[:len(d)], d["top8_overlap"], "^:", color=P[1], lw=2.0, label="routing top-8 overlap")
    ax.plot(x, e_o["volume_cosine_vs_final"], "s--", color=P[2], lw=2.0, label="volume cosine")
    ax.plot(x, h_mo["matrix_cosine_vs_final"], "o-", color=P[0], lw=2.2, label="matrix cosine")
    ax.plot(x, h_lo["fabric_hot_overlap_vs_final"], "D-", color=P[4], lw=1.6, label="hot-link overlap")
    ax.set_xticks(x)
    labels = [str(int(c)) if str(c) != "final" else "final" for c in e_o["checkpoint"]]
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_xlabel("OLMoE-1B-7B checkpoint (steps)")
    ax.set_title("OLMoE-1B-7B")
    ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("Predictability ladder: routing < volume ≤ matrix", fontweight="bold")
    save_both(fig, "fig3_1_predictability_ladder")


def fig_volume_vs_routing() -> None:
    b = pd.read_csv(ROOT / "exp_B_routing_stabilization/results/stabilization_layer_avg.csv")
    e = pd.read_csv(ROOT / "exp_E_volume_stabilization/results/volume_stability_avg.csv")
    e_o = pd.read_csv(ROOT / "exp_E_volume_stabilization/results/olmoe_volume_stability.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)

    ax = axes[0]
    ax.plot(e["checkpoint"], e["cosine_vs_final"], "o-", color=P[0], lw=2.2, label="volume cosine vs final")
    ax.plot(b["checkpoint"], b["top6_overlap"], "s--", color=P[1], lw=2.0, label="routing top-6 overlap")
    ax.plot(b["checkpoint"], b["top1_match_rate"], "^:", color=P[3], lw=1.8, label="routing top-1 match")
    ax.set_xlabel("FLAME-290M iteration")
    ax.set_ylabel("Agreement with final model")
    ax.set_title("FLAME-MoE-290M")
    ax.set_ylim(0.25, 1.05)
    ax.legend(fontsize=8)

    ax = axes[1]
    ckpt = e_o["checkpoint"].astype(str)
    x = np.arange(len(e_o))
    ax.plot(x, e_o["volume_cosine_vs_final"], "o-", color=P[0], lw=2.2, label="volume cosine vs final")
    ax.plot(x, e_o["routing_top8_overlap_vs_final"], "s--", color=P[1], lw=2.0, label="routing top-8 overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(ckpt, rotation=30, fontsize=8)
    ax.set_xlabel("OLMoE-1B-7B checkpoint")
    ax.set_title("OLMoE-1B-7B")
    ax.legend(fontsize=8)

    fig.suptitle("Aggregate volume settles before per-token routing", fontweight="bold")
    save_both(fig, "fig3_2_volume_vs_routing")


def fig_envelope() -> None:
    f = pd.read_csv(ROOT / "exp_H_link_stabilization/results/envelope_quality.csv")
    o = pd.read_csv(ROOT / "exp_H_link_stabilization/results/olmoe_envelope_quality.csv")

    flame_offsets = {
        "mean": (6, 5),
        "P90": (-8, 10),
        "P95": (-8, -15),
        "P99": (5, 8),
        "worst": (5, -13),
    }
    olmoe_offsets = {
        "mean": (-16, -15),
        "P90": (-8, -16),
        "P95": (-8, 8),
        "P99": (-24, -15),
        "worst": (5, -2),
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(f["link_waste_ratio"] * 100, f["link_overflow_ratio"] * 100,
            "o-", color=P[0], lw=1.8, ms=8, label="FLAME-290M")
    for _, r in f.iterrows():
        ax.annotate(r["policy"], (r["link_waste_ratio"] * 100, r["link_overflow_ratio"] * 100),
                    textcoords="offset points", xytext=flame_offsets[r["policy"]],
                    fontsize=8, color=P[0])
    ax.plot(o["link_waste_ratio"] * 100, o["link_overflow_ratio"] * 100,
            "s--", color=P[1], lw=1.8, ms=8, label="OLMoE-1B-7B")
    for _, r in o.iterrows():
        ax.annotate(r["policy"], (r["link_waste_ratio"] * 100, r["link_overflow_ratio"] * 100),
                    textcoords="offset points", xytext=olmoe_offsets[r["policy"]],
                    fontsize=8, color=P[1])
    ax.set_xlabel("Reservation waste (% of envelope unused)")
    ax.set_ylabel("Overflow (% of traffic exceeding envelope)")
    ax.set_title("Reservation envelope on held-out later checkpoints")
    ax.legend(fontsize=9)
    save_both(fig, "fig3_3_envelope_tradeoff")


def fig_cross_size() -> None:
    g = pd.read_csv(ROOT / "exp_G_cross_model_generalization/results/cross_model_stabilization.csv")
    short = {"flame-moe-290m": "290M", "flame-moe-721m": "721M", "flame-moe-1.7b": "1.7B"}
    colors = {"flame-moe-290m": P[0], "flame-moe-721m": P[2], "flame-moe-1.7b": P[1]}

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for m, c in colors.items():
        d = g[g["model"] == m]
        ax.plot(d["progress"], d["volume_cosine_vs_final"], "s--", color=c, lw=1.6, alpha=0.85)
        ax.plot(d["progress"], d["routing_top6_overlap"], "o-", color=c, lw=2.0, label=f"{short[m]} routing")
    spike = g[(g["model"] == "flame-moe-1.7b") & (g["checkpoint"] == 3300)].iloc[0]
    ax.annotate("1.7B collapse\niter 3300",
                xy=(spike["progress"], spike["volume_cosine_vs_final"]),
                xytext=(spike["progress"] + 0.08, 0.55),
                fontsize=8, color=P[1],
                arrowprops=dict(arrowstyle="->", color=P[1], lw=1))
    ax.set_xlabel("Training progress (iteration / final iteration)")
    ax.set_ylabel("Agreement with final model")
    ax.set_title("Volume (dashed) above routing (solid) at 290M / 721M / 1.7B")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    save_both(fig, "fig3_4_cross_size_collapse")


def fig_hetero() -> None:
    f = pd.read_csv(ROOT / "exp_L_heterogeneous_sources/results/hetero_sweep.csv")
    o = pd.read_csv(ROOT / "exp_L_heterogeneous_sources/results/olmoe_hetero_sweep.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    ax.plot(f["alpha"], f["rank1_residual"], "o-", color=P[0], lw=2.0, label="FLAME-290M")
    ax.plot(o["alpha"], o["rank1_residual"], "s--", color=P[1], lw=2.0, label="OLMoE-1B-7B")
    ax.axhline(0.1, color="black", ls=":", lw=1, label="≈0.1 crossover (report)")
    ax.set_xlabel("Source specialisation α (modelled)")
    ax.set_ylabel("Rank-1 residual of dispatch matrix")
    ax.set_title("Matrix structure appears with specialised sources")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(f["alpha"], f["matrix_cosine_vs_final"], "o-", color=P[0], lw=2.0, label="FLAME-290M")
    ax.plot(o["alpha"], o["matrix_cosine_vs_final"], "s--", color=P[1], lw=2.0, label="OLMoE-1B-7B")
    ax.set_xlabel("Source specialisation α (modelled)")
    ax.set_ylabel("Matrix cosine vs final")
    ax.set_title("Heterogeneity does not break freeze-ability")
    ax.set_ylim(0.9994, 1.00005)
    ax.legend(fontsize=8)

    fig.suptitle("When the matrix is more than volume — and still plannable", fontweight="bold")
    save_both(fig, "fig3_5_heterogeneous_sources")


def fig_cross_layer() -> None:
    f = pd.read_csv(ROOT / "exp_J_first_layer_prediction/results/prediction_quality_avg.csv")
    o = pd.read_csv(ROOT / "exp_J_first_layer_prediction/results/olmoe_prediction_quality_avg.csv")
    order = ["uniform", "firstlayer_identity", "firstlayer_transpose", "history_mean"]
    labels = ["uniform", "first layer", "first-layer\ntranspose", "per-layer\nhistory"]
    f = f.set_index("predictor").loc[order]
    o = o.set_index("predictor").loc[order]

    x = np.arange(len(order))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - w / 2, f["hot_link_overlap"], width=w, color=P[0], label="FLAME-290M")
    ax.bar(x + w / 2, o["hot_link_overlap"], width=w, color=P[1], label="OLMoE-1B-7B")
    ax.axhline(0.25, color="black", ls=":", lw=1, label="chance (hottest 25%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hot-link overlap with later layer")
    ax.set_ylim(0, 1.0)
    ax.set_title("First layer does not predict later-layer hot links")
    ax.legend(fontsize=8)
    save_both(fig, "fig3_6_cross_layer_negative")


def main() -> None:
    plotting.apply_style()
    fig_ladder()
    fig_volume_vs_routing()
    fig_envelope()
    fig_cross_size()
    fig_hetero()
    fig_cross_layer()
    print(f"wrote PDF+SVG under {OUT}")


if __name__ == "__main__":
    main()
