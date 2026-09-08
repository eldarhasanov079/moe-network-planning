"""Experiment C - Expert Load Skew (FLAME-MoE-290M)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common"))

import config  # noqa: E402
from common import plotting  # noqa: E402
from common.flame_loader import load_layer  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def expert_counts(indices: np.ndarray, num_experts: int) -> np.ndarray:
    """Total times each expert id appears across all (token, slot) entries."""
    return np.bincount(indices.ravel(), minlength=num_experts).astype(np.float64)


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative load vector (0 = even, ->1 = skewed)."""
    x = np.sort(x)
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def skew_metrics(counts: np.ndarray) -> dict:
    total = counts.sum()
    mean = counts.mean()
    sorted_desc = np.sort(counts)[::-1]
    top8_share = sorted_desc[:8].sum() / total
    return {
        "max_over_mean": counts.max() / mean,
        "min_over_mean": counts.min() / mean,
        "coef_variation": counts.std() / mean,
        "gini": gini(counts),
        "top8_traffic_share_pct": 100.0 * top8_share,
        "n_experts_used": int(np.sum(counts > 0)),
        "expected_share_per_expert_pct": 100.0 / len(counts),
    }


def main() -> None:
    plotting.apply_style()
    import matplotlib.pyplot as plt

    final = config.FLAME_FINAL_CHECKPOINT
    layers = config.FLAME_LAYERS
    ne = config.FLAME_NUM_EXPERTS
    n_rows = config.SAMPLE_MARGIN_FINAL  # reuse Exp A's cached sample

    print("[C] Computing per-layer expert load at final checkpoint...")
    counts_by_layer = {}
    metric_rows = []
    for layer in layers:
        _, indices = load_layer(final, layer, n_rows)
        counts = expert_counts(indices, ne)
        counts_by_layer[layer] = counts
        row = {"layer": layer}
        row.update(skew_metrics(counts))
        metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS / "skew_metrics_by_layer.csv", index=False)

    share_rows = []
    for layer in layers:
        c = counts_by_layer[layer]
        share = 100.0 * c / c.sum()
        for e in range(ne):
            share_rows.append({"layer": layer, "expert_id": e, "traffic_share_pct": share[e]})
    pd.DataFrame(share_rows).to_csv(RESULTS / "expert_load_shares.csv", index=False)

    rep_layer = layers[len(layers) // 2]  # layer_06
    print(f"[C] Computing skew evolution for {rep_layer} across checkpoints...")
    evo_rows = []
    for ckpt in config.FLAME_CHECKPOINTS:
        _, indices = load_layer(ckpt, rep_layer, config.SAMPLE_MARGIN_TREND)
        counts = expert_counts(indices, ne)
        row = {"checkpoint": ckpt}
        row.update(skew_metrics(counts))
        evo_rows.append(row)
    evo_df = pd.DataFrame(evo_rows)
    evo_df.to_csv(RESULTS / "skew_evolution.csv", index=False)

    even = 100.0 / ne  # even-routing reference line (%)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    for ax, layer in zip(axes.ravel(), layers):
        c = counts_by_layer[layer]
        share = 100.0 * np.sort(c)[::-1] / c.sum()
        ax.bar(range(ne), share, color=plotting.PALETTE[0], width=1.0)
        ax.axhline(even, color="#d1495b", ls="--", lw=1)
        ax.set_title(layer, fontsize=11)
    fig.supxlabel("Experts (sorted by load, busiest first)")
    fig.supylabel("Traffic share (%)")
    fig.suptitle("Per-layer expert load distribution (final checkpoint) — red dashed = even routing",
                 fontweight="bold")
    plotting.save(fig, RESULTS / "fig1_load_distribution_by_layer.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for j, layer in enumerate(layers):
        c = counts_by_layer[layer]
        share = 100.0 * np.sort(c)[::-1] / c.sum()
        ax.plot(range(ne), share, lw=1.8, color=plotting.PALETTE[j % len(plotting.PALETTE)],
                label=layer)
    ax.axhline(even, color="black", ls="--", lw=1, label="even (1.56%)")
    ax.set_xlabel("Experts (sorted by load, busiest first)")
    ax.set_ylabel("Traffic share (%)")
    ax.set_title("Sorted expert-load curves — how concentrated is traffic?")
    ax.legend(ncol=2, fontsize=8)
    plotting.save(fig, RESULTS / "fig2_sorted_load_curves.png")

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    x = range(len(layers))
    ax1.bar(x, metrics_df["max_over_mean"], color=plotting.PALETTE[3], alpha=0.85,
            label="max/mean load")
    ax1.set_ylabel("Max / mean expert load", color=plotting.PALETTE[3])
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(layers, rotation=45, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, metrics_df["gini"], "o-", color=plotting.PALETTE[4], lw=2, label="Gini")
    ax2.set_ylabel("Gini coefficient", color=plotting.PALETTE[4])
    ax2.grid(False)
    ax1.set_title("Expert-load skew by layer (final checkpoint)")
    plotting.save(fig, RESULTS / "fig3_skew_metrics_by_layer.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(evo_df["checkpoint"], evo_df["max_over_mean"], "o-",
            color=plotting.PALETTE[3], lw=2, label="max/mean load")
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel("Max / mean expert load")
    ax2 = ax.twinx()
    ax2.plot(evo_df["checkpoint"], evo_df["top8_traffic_share_pct"], "s--",
             color=plotting.PALETTE[2], lw=2, label="top-8 traffic share (%)")
    ax2.set_ylabel("Top-8 traffic share (%)")
    ax2.grid(False)
    ax.set_title(f"Does load skew change during training?  ({rep_layer})")
    plotting.save(fig, RESULTS / "fig4_skew_evolution.png")

    worst = metrics_df.loc[metrics_df["max_over_mean"].idxmax()]
    headline = {
        "model": config.FLAME_MODEL,
        "final_checkpoint": final,
        "num_experts": ne,
        "even_share_pct": even,
        "max_over_mean_worst_layer": worst["layer"],
        "max_over_mean_worst_value": worst["max_over_mean"],
        "mean_top8_share_pct": metrics_df["top8_traffic_share_pct"].mean(),
        "mean_gini": metrics_df["gini"].mean(),
    }
    pd.DataFrame([headline]).to_csv(RESULTS / "headline.csv", index=False)

    print("\n=== Experiment C headline ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    print(f"\nHEADLINE: On {config.FLAME_MODEL}, the busiest expert in {worst['layer']} "
          f"receives {worst['max_over_mean']:.1f}x the average expert's traffic; "
          f"the top-8 of 64 experts capture {headline['mean_top8_share_pct']:.0f}% of traffic on average.")


if __name__ == "__main__":
    main()
