"""Experiment A - Router Margin Analysis (FLAME-MoE-290M)."""

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


def margins_for(checkpoint: int, layer: str, n_rows: int) -> np.ndarray:
    """margin = scores[:,0] - scores[:,1] for each token."""
    scores, _ = load_layer(checkpoint, layer, n_rows)
    return (scores[:, 0] - scores[:, 1]).astype(np.float64)


def main() -> None:
    plotting.apply_style()
    import matplotlib.pyplot as plt

    final = config.FLAME_FINAL_CHECKPOINT
    layers = config.FLAME_LAYERS

    print("[A] Loading final-checkpoint margins per layer...")
    per_layer = {}
    for layer in layers:
        per_layer[layer] = margins_for(final, layer, config.SAMPLE_MARGIN_FINAL)
    all_margins = np.concatenate(list(per_layer.values()))

    rows = []
    for thr in config.MARGIN_THRESHOLDS:
        rows.append({
            "scope": "ALL_LAYERS",
            "threshold": thr,
            "pct_tokens_below": 100.0 * np.mean(all_margins < thr),
        })
    for layer in layers:
        m = per_layer[layer]
        for thr in config.MARGIN_THRESHOLDS:
            rows.append({
                "scope": layer,
                "threshold": thr,
                "pct_tokens_below": 100.0 * np.mean(m < thr),
            })
    thr_df = pd.DataFrame(rows)
    thr_df.to_csv(RESULTS / "margin_thresholds.csv", index=False)

    summ_rows = []
    for layer in layers:
        m = per_layer[layer]
        summ_rows.append({
            "layer": layer,
            "n_tokens": len(m),
            "mean_margin": m.mean(),
            "median_margin": np.median(m),
            "p95_margin": np.percentile(m, 95),
            "pct_below_0.03": 100.0 * np.mean(m < 0.03),
        })
    summ_df = pd.DataFrame(summ_rows)
    summ_df.to_csv(RESULTS / "margin_by_layer.csv", index=False)

    deflect_df = pd.DataFrame([{
        "metric": "score_loss_deflect_to_2nd",
        "mean": all_margins.mean(),
        "median": np.median(all_margins),
        "p95": np.percentile(all_margins, 95),
        "p99": np.percentile(all_margins, 99),
    }])
    deflect_df.to_csv(RESULTS / "deflection_cost.csv", index=False)

    print("[A] Loading margin trend across checkpoints...")
    trend_rows = []
    trend_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
    for ckpt in config.FLAME_CHECKPOINTS:
        for layer in trend_layers:
            m = margins_for(ckpt, layer, config.SAMPLE_MARGIN_TREND)
            trend_rows.append({
                "checkpoint": ckpt,
                "layer": layer,
                "mean_margin": m.mean(),
                "pct_below_0.03": 100.0 * np.mean(m < 0.03),
            })
    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(RESULTS / "margin_trend.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(all_margins, bins=120, range=(0, 0.4), color=plotting.PALETTE[0], alpha=0.85)
    for thr in config.MARGIN_THRESHOLDS:
        ax.axvline(thr, color="#d1495b", ls="--", lw=1, alpha=0.7)
        ax.text(thr, ax.get_ylim()[1] * 0.92, f"{thr}", rotation=90,
                va="top", ha="right", fontsize=8, color="#d1495b")
    ax.set_xlabel("Router margin  (top-1 prob − top-2 prob)")
    ax.set_ylabel("Token count")
    ax.set_title(f"FLAME-MoE-290M router margin (final ckpt, all {len(layers)} layers)")
    plotting.save(fig, RESULTS / "fig1_margin_hist_overall.png")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for ax, layer in zip(axes.ravel(), layers):
        ax.hist(per_layer[layer], bins=80, range=(0, 0.4),
                color=plotting.PALETTE[1], alpha=0.85)
        ax.axvline(0.03, color="#2c6fbb", ls="--", lw=1)
        ax.set_title(layer, fontsize=11)
    fig.supxlabel("Router margin")
    fig.supylabel("Token count")
    fig.suptitle("Per-layer router margin distribution (final checkpoint)", fontweight="bold")
    plotting.save(fig, RESULTS / "fig2_margin_hist_by_layer.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    all_only = thr_df[thr_df["scope"] == "ALL_LAYERS"]
    bars = ax.bar([str(t) for t in all_only["threshold"]],
                  all_only["pct_tokens_below"], color=plotting.PALETTE[2], alpha=0.9)
    for b, v in zip(bars, all_only["pct_tokens_below"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%",
                ha="center", fontsize=10)
    ax.set_xlabel("Margin threshold")
    ax.set_ylabel("% of tokens with margin below threshold")
    ax.set_title("Share of low-confidence (deflectable) routing decisions")
    plotting.save(fig, RESULTS / "fig3_threshold_share.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(layers)), summ_df["mean_margin"], "o-",
            color=plotting.PALETTE[0], lw=2)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_ylabel("Mean router margin")
    ax.set_title("Mean router margin by layer (final checkpoint)")
    plotting.save(fig, RESULTS / "fig4_margin_by_layer.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for j, layer in enumerate(trend_layers):
        sub = trend_df[trend_df["layer"] == layer]
        ax.plot(sub["checkpoint"], sub["mean_margin"], "o-",
                color=plotting.PALETTE[j], lw=2, label=layer)
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel("Mean router margin")
    ax.set_title("Does routing confidence grow as training stabilises?")
    ax.legend(title="layer")
    plotting.save(fig, RESULTS / "fig5_margin_trend.png")

    pct003 = 100.0 * np.mean(all_margins < 0.03)
    pct005 = 100.0 * np.mean(all_margins < 0.05)
    headline = {
        "model": config.FLAME_MODEL,
        "final_checkpoint": final,
        "n_layers": len(layers),
        "n_tokens_per_layer": config.SAMPLE_MARGIN_FINAL,
        "pct_below_0.01": 100.0 * np.mean(all_margins < 0.01),
        "pct_below_0.03": pct003,
        "pct_below_0.05": pct005,
        "pct_below_0.10": 100.0 * np.mean(all_margins < 0.10),
        "mean_margin": all_margins.mean(),
        "median_margin": float(np.median(all_margins)),
        "p95_deflection_cost": float(np.percentile(all_margins, 95)),
    }
    pd.DataFrame([headline]).to_csv(RESULTS / "headline.csv", index=False)

    print("\n=== Experiment A headline ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    print(f"\nHEADLINE: On {config.FLAME_MODEL}, {pct003:.1f}% of routed tokens "
          f"have routing margin < 0.03 (final checkpoint).")


if __name__ == "__main__":
    main()
