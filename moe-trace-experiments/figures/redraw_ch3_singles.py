"""Chapter 3 singles: one model, one claim per figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
LEVELS = OUT / "levels"
SINGLES = OUT / "singles"
sys.path.insert(0, str(ROOT))

from common import plotting  # noqa: E402

P = plotting.PALETTE
SIZE = (6.4, 4.4)


def save(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        plotting.save(fig, directory / f"{stem}.{ext}", close=False)
    plt.close(fig)


def _ckpt_labels(series) -> list[str]:
    out = []
    for c in series:
        s = str(c)
        if s == "final":
            out.append("final")
        else:
            try:
                out.append(str(int(float(s))))
            except ValueError:
                out.append(s)
    return out


def _line(ax, x, y, *, color, marker, ls, label, lw=2.0):
    ax.plot(x, y, marker=marker, ls=ls, color=color, lw=lw, label=label)


def level1_routing() -> None:
    flame = pd.read_csv(ROOT / "exp_B_routing_stabilization/results/stabilization_layer_avg.csv")
    olmoe = pd.read_csv(ROOT / "exp_D_olmoe_saturation/results/olmoe_stabilization_avg.csv")

    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(flame))
    _line(ax, x, 100 * flame["top1_match_rate"], color=P[0], marker="o", ls="-", label="top-1 match")
    _line(ax, x, 100 * flame["top6_overlap"], color=P[1], marker="s", ls="--", label="top-6 set overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(_ckpt_labels(flame["checkpoint"]), rotation=45, fontsize=8)
    ax.set_xlabel("FLAME-MoE-290M checkpoint (iteration)")
    ax.set_ylabel("Agreement with final snapshot (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Token-routing agreement in FLAME-MoE-290M")
    ax.legend(fontsize=9)
    save(fig, LEVELS, "L1_routing_flame")

    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(olmoe))
    _line(ax, x, 100 * olmoe["top1_match_rate"], color=P[0], marker="o", ls="-", label="top-1 match")
    _line(ax, x, 100 * olmoe["top8_overlap"], color=P[1], marker="s", ls="--", label="top-8 set overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(_ckpt_labels(olmoe["checkpoint"]), rotation=30, fontsize=8)
    ax.set_xlabel("OLMoE-1B-7B checkpoint (steps)")
    ax.set_ylabel("Agreement with final snapshot (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Token-routing agreement in OLMoE-1B-7B")
    ax.legend(fontsize=9)
    save(fig, LEVELS, "L1_routing_olmoe")


def level2_volume() -> None:
    flame = pd.read_csv(ROOT / "exp_E_volume_stabilization/results/volume_stability_avg.csv")
    olmoe = pd.read_csv(ROOT / "exp_E_volume_stabilization/results/olmoe_volume_stability.csv")

    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(flame))
    _line(ax, x, flame["cosine_vs_final"], color=P[0], marker="o", ls="-", label="cosine vs final")
    ax2 = ax.twinx()
    ax2.plot(x, flame["norm_l1_vs_final"], "s--", color=P[3], lw=2.0, label="normalised ℓ1 vs final")
    ax2.set_ylabel("Normalised ℓ1 vs final")
    ax2.grid(False)
    ax.set_xticks(x)
    ax.set_xticklabels(_ckpt_labels(flame["checkpoint"]), rotation=45, fontsize=8)
    ax.set_xlabel("FLAME-MoE-290M checkpoint (iteration)")
    ax.set_ylabel("Cosine vs final")
    ax.set_ylim(0.99, 1.001)
    ax.set_title("Expert-volume stability in FLAME-MoE-290M")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    save(fig, LEVELS, "L2_volume_flame")

    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(olmoe))
    _line(ax, x, olmoe["volume_cosine_vs_final"], color=P[0], marker="o", ls="-", label="cosine vs final")
    ax2 = ax.twinx()
    ax2.plot(x, olmoe["volume_norm_l1_vs_final"], "s--", color=P[3], lw=2.0, label="normalised ℓ1 vs final")
    ax2.set_ylabel("Normalised ℓ1 vs final")
    ax2.grid(False)
    ax.set_xticks(x)
    ax.set_xticklabels(_ckpt_labels(olmoe["checkpoint"]), rotation=30, fontsize=8)
    ax.set_xlabel("OLMoE-1B-7B checkpoint (steps)")
    ax.set_ylabel("Cosine vs final")
    ax.set_ylim(0.93, 1.005)
    ax.set_title("Expert-volume stability in OLMoE-1B-7B")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    save(fig, LEVELS, "L2_volume_olmoe")


def level3_matrix() -> None:
    flame = pd.read_csv(ROOT / "exp_H_link_stabilization/results/matrix_stability_avg.csv")
    olmoe = pd.read_csv(ROOT / "exp_H_link_stabilization/results/olmoe_matrix_stability_avg.csv")

    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(flame))
    _line(ax, x, flame["matrix_cosine_vs_final"], color=P[0], marker="o", ls="-", label="matrix cosine vs final")
    ax2 = ax.twinx()
    ax2.plot(x, flame["rank1_residual"], "D--", color=P[4], lw=2.0, label="rank-1 residual")
    ax2.set_ylabel("Rank-1 residual")
    ax2.set_ylim(0, 0.08)
    ax2.grid(False)
    ax.set_xticks(x)
    ax.set_xticklabels(_ckpt_labels(flame["checkpoint"]), rotation=45, fontsize=8)
    ax.set_xlabel("FLAME-MoE-290M checkpoint (iteration)")
    ax.set_ylabel("Matrix cosine vs final")
    ax.set_ylim(0.9985, 1.0002)
    ax.set_title("Destination-matrix stability in FLAME-MoE-290M")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    save(fig, LEVELS, "L3_matrix_flame")

    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(olmoe))
    _line(ax, x, olmoe["matrix_cosine_vs_final"], color=P[0], marker="o", ls="-", label="matrix cosine vs final")
    ax2 = ax.twinx()
    ax2.plot(x, olmoe["rank1_residual"], "D--", color=P[4], lw=2.0, label="rank-1 residual")
    ax2.set_ylabel("Rank-1 residual")
    ax2.set_ylim(0, 0.08)
    ax2.grid(False)
    ax.set_xticks(x)
    ax.set_xticklabels(_ckpt_labels(olmoe["checkpoint"]), rotation=30, fontsize=8)
    ax.set_xlabel("OLMoE-1B-7B checkpoint (steps)")
    ax.set_ylabel("Matrix cosine vs final")
    ax.set_ylim(0.990, 1.001)
    ax.set_title("Destination-matrix stability in OLMoE-1B-7B")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    save(fig, LEVELS, "L3_matrix_olmoe")


def level4_link() -> None:
    flame = pd.read_csv(ROOT / "exp_H_link_stabilization/results/link_stability_avg.csv")
    olmoe = pd.read_csv(ROOT / "exp_H_link_stabilization/results/olmoe_link_stability_avg.csv")

    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(flame))
    _line(ax, x, flame["fabric_max_over_mean"], color=P[0], marker="o", ls="-", label="full-mesh max/mean")
    _line(ax, x, flame["uplink_max_over_mean"], color=P[3], marker="s", ls="--", label="2-tier uplink max/mean")
    ax.axhline(1.0, color="black", ls=":", lw=1, label="perfectly balanced")
    ax.set_xticks(x)
    ax.set_xticklabels(_ckpt_labels(flame["checkpoint"]), rotation=45, fontsize=8)
    ax.set_xlabel("FLAME-MoE-290M checkpoint (iteration)")
    ax.set_ylabel("Link-load skew (max / mean)")
    ax.set_ylim(0.98, 1.20)
    ax.set_title("Link-load balance in FLAME-MoE-290M")
    ax.legend(fontsize=8)
    save(fig, LEVELS, "L4_link_flame")

    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(olmoe))
    _line(ax, x, olmoe["fabric_max_over_mean"], color=P[0], marker="o", ls="-", label="full-mesh max/mean")
    _line(ax, x, olmoe["uplink_max_over_mean"], color=P[3], marker="s", ls="--", label="2-tier uplink max/mean")
    ax.axhline(1.0, color="black", ls=":", lw=1, label="perfectly balanced")
    ax.set_xticks(x)
    ax.set_xticklabels(_ckpt_labels(olmoe["checkpoint"]), rotation=30, fontsize=8)
    ax.set_xlabel("OLMoE-1B-7B checkpoint (steps)")
    ax.set_ylabel("Link-load skew (max / mean)")
    ax.set_ylim(0.98, 1.40)
    ax.set_title("Link-load balance in OLMoE-1B-7B")
    ax.legend(fontsize=8)
    save(fig, LEVELS, "L4_link_olmoe")


def split_old_combined() -> None:
    b = pd.read_csv(ROOT / "exp_B_routing_stabilization/results/stabilization_layer_avg.csv")
    e = pd.read_csv(ROOT / "exp_E_volume_stabilization/results/volume_stability_avg.csv")
    h_m = pd.read_csv(ROOT / "exp_H_link_stabilization/results/matrix_stability_avg.csv")
    h_l = pd.read_csv(ROOT / "exp_H_link_stabilization/results/link_stability_avg.csv")
    d = pd.read_csv(ROOT / "exp_D_olmoe_saturation/results/olmoe_stabilization_avg.csv")
    e_o = pd.read_csv(ROOT / "exp_E_volume_stabilization/results/olmoe_volume_stability.csv")
    h_mo = pd.read_csv(ROOT / "exp_H_link_stabilization/results/olmoe_matrix_stability_avg.csv")
    h_lo = pd.read_csv(ROOT / "exp_H_link_stabilization/results/olmoe_link_stability_avg.csv")

    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(b))
    ax.plot(x, b["top6_overlap"], "^:", color=P[1], lw=2.0, label="routing top-6 overlap")
    ax.plot(x, e["cosine_vs_final"], "s--", color=P[2], lw=2.0, label="volume cosine")
    ax.plot(x, h_m["matrix_cosine_vs_final"], "o-", color=P[0], lw=2.2, label="matrix cosine")
    ax.plot(x, h_l["fabric_hot_overlap_vs_final"], "D-", color=P[4], lw=1.6, label="hot-link overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(_ckpt_labels(b["checkpoint"]), rotation=45, fontsize=8)
    ax.set_xlabel("FLAME-MoE-290M checkpoint (iteration)")
    ax.set_ylabel("Agreement with final model")
    ax.set_ylim(0.2, 1.05)
    ax.set_title("Predictability ladder (FLAME-MoE-290M)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, SINGLES, "fig3_1_ladder_flame")

    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(e_o))
    ax.plot(x[: len(d)], d["top8_overlap"], "^:", color=P[1], lw=2.0, label="routing top-8 overlap")
    ax.plot(x, e_o["volume_cosine_vs_final"], "s--", color=P[2], lw=2.0, label="volume cosine")
    ax.plot(x, h_mo["matrix_cosine_vs_final"], "o-", color=P[0], lw=2.2, label="matrix cosine")
    ax.plot(x, h_lo["fabric_hot_overlap_vs_final"], "D-", color=P[4], lw=1.6, label="hot-link overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(_ckpt_labels(e_o["checkpoint"]), rotation=30, fontsize=8)
    ax.set_xlabel("OLMoE-1B-7B checkpoint (steps)")
    ax.set_ylabel("Agreement with final model")
    ax.set_ylim(0.2, 1.05)
    ax.set_title("Predictability ladder (OLMoE-1B-7B)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, SINGLES, "fig3_1_ladder_olmoe")

    fig, ax = plt.subplots(figsize=SIZE)
    ax.plot(e["checkpoint"], e["cosine_vs_final"], "o-", color=P[0], lw=2.2, label="volume cosine vs final")
    ax.plot(b["checkpoint"], b["top6_overlap"], "s--", color=P[1], lw=2.0, label="routing top-6 overlap")
    ax.plot(b["checkpoint"], b["top1_match_rate"], "^:", color=P[3], lw=1.8, label="routing top-1 match")
    ax.set_xlabel("FLAME-MoE-290M iteration")
    ax.set_ylabel("Agreement with final model")
    ax.set_ylim(0.25, 1.05)
    ax.set_title("Volume vs routing (FLAME-MoE-290M)")
    ax.legend(fontsize=8)
    save(fig, SINGLES, "fig3_2_volume_vs_routing_flame")

    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(e_o))
    ax.plot(x, e_o["volume_cosine_vs_final"], "o-", color=P[0], lw=2.2, label="volume cosine vs final")
    ax.plot(x, e_o["routing_top8_overlap_vs_final"], "s--", color=P[1], lw=2.0, label="routing top-8 overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(_ckpt_labels(e_o["checkpoint"]), rotation=30, fontsize=8)
    ax.set_xlabel("OLMoE-1B-7B checkpoint (steps)")
    ax.set_ylabel("Agreement with final model")
    ax.set_ylim(0.25, 1.05)
    ax.set_title("Volume vs routing (OLMoE-1B-7B)")
    ax.legend(fontsize=8)
    save(fig, SINGLES, "fig3_2_volume_vs_routing_olmoe")

    f = pd.read_csv(ROOT / "exp_H_link_stabilization/results/envelope_quality.csv")
    o = pd.read_csv(ROOT / "exp_H_link_stabilization/results/olmoe_envelope_quality.csv")
    for df, stem, title in (
        (f, "fig3_3_envelope_flame", "Envelope trade-off (FLAME-MoE-290M)"),
        (o, "fig3_3_envelope_olmoe", "Envelope trade-off (OLMoE-1B-7B)"),
    ):
        fig, ax = plt.subplots(figsize=SIZE)
        ax.plot(df["link_waste_ratio"] * 100, df["link_overflow_ratio"] * 100,
                "o-", color=P[0], lw=1.8, ms=8)
        for _, r in df.iterrows():
            ax.annotate(r["policy"], (r["link_waste_ratio"] * 100, r["link_overflow_ratio"] * 100),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlabel("Reservation waste (% unused)")
        ax.set_ylabel("Overflow (% exceeding envelope)")
        ax.set_title(title)
        save(fig, SINGLES, stem)

    hf = pd.read_csv(ROOT / "exp_L_heterogeneous_sources/results/hetero_sweep.csv")
    ho = pd.read_csv(ROOT / "exp_L_heterogeneous_sources/results/olmoe_hetero_sweep.csv")
    for df, stem, title in (
        (hf, "fig3_5_hetero_residual_flame", "Rank-1 residual vs α (FLAME-MoE-290M)"),
        (ho, "fig3_5_hetero_residual_olmoe", "Rank-1 residual vs α (OLMoE-1B-7B)"),
        (hf, "fig3_5_hetero_cosine_flame", "Matrix cosine vs α (FLAME-MoE-290M)"),
        (ho, "fig3_5_hetero_cosine_olmoe", "Matrix cosine vs α (OLMoE-1B-7B)"),
    ):
        fig, ax = plt.subplots(figsize=SIZE)
        ycol = "rank1_residual" if "residual" in stem else "matrix_cosine_vs_final"
        ax.plot(df["alpha"], df[ycol], "o-", color=P[0], lw=2.0)
        ax.set_xlabel("Source specialisation α (modelled)")
        ax.set_ylabel("Rank-1 residual" if "residual" in stem else "Matrix cosine vs final")
        ax.set_title(title)
        save(fig, SINGLES, stem)

    fj = pd.read_csv(ROOT / "exp_J_first_layer_prediction/results/prediction_quality_avg.csv")
    oj = pd.read_csv(ROOT / "exp_J_first_layer_prediction/results/olmoe_prediction_quality_avg.csv")
    order = ["uniform", "firstlayer_identity", "firstlayer_transpose", "history_mean"]
    labels = ["uniform", "first layer", "first-layer\ntranspose", "per-layer\nhistory"]
    for df, stem, title in (
        (fj, "fig3_6_cross_layer_flame", "Cross-layer hot-link overlap (FLAME-MoE-290M)"),
        (oj, "fig3_6_cross_layer_olmoe", "Cross-layer hot-link overlap (OLMoE-1B-7B)"),
    ):
        d = df.set_index("predictor").loc[order]
        fig, ax = plt.subplots(figsize=SIZE)
        ax.bar(np.arange(len(order)), d["hot_link_overlap"], color=P[0], width=0.55)
        ax.axhline(0.25, color="black", ls=":", lw=1, label="chance (hottest 25%)")
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Hot-link overlap with later layer")
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.legend(fontsize=8)
        save(fig, SINGLES, stem)


def main() -> None:
    plotting.apply_style()
    plt.rcParams["savefig.dpi"] = 200
    level1_routing()
    level2_volume()
    level3_matrix()
    level4_link()
    split_old_combined()
    print(f"levels  -> {LEVELS}")
    print(f"singles -> {SINGLES}")


if __name__ == "__main__":
    main()
