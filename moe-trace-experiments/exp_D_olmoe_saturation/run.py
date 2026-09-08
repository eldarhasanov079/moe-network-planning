"""Experiment D - OLMoE Router Saturation & Load Skew (cross-model check)."""

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
from common.olmoe_loader import load_checkpoint, OLMOE_CHECKPOINTS  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Per-row top-k set overlap fraction between (N,k) arrays."""
    eq = (a[:, :, None] == b[:, None, :]).any(axis=2)
    return eq.sum(axis=1) / k


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def main() -> None:
    plotting.apply_style()
    import matplotlib.pyplot as plt

    n = config.SAMPLE_OLMOE
    nl = config.OLMOE_NUM_LAYERS
    ne = config.OLMOE_NUM_EXPERTS
    k = config.OLMOE_TOP_K

    data = {}
    for ckpt in OLMOE_CHECKPOINTS:
        data[ckpt] = load_checkpoint(ckpt, n)
    final = data["final"]
    m = min(arr.shape[0] for arr in data.values())

    print("[D] Computing OLMoE stabilization vs final...")
    rows = []
    early = [c for c in OLMOE_CHECKPOINTS if c != "final"]
    for ckpt in early:
        arr = data[ckpt]
        for L in range(nl):
            a = arr[:m, L, :]
            b = final[:m, L, :]
            top1 = float(np.mean(a[:, 0] == b[:, 0]))
            ov = float(np.mean(topk_overlap(a, b, k)))
            rows.append({"checkpoint": ckpt, "layer": L,
                         "top1_match_rate": top1, "top8_overlap": ov})
    stab = pd.DataFrame(rows)
    stab.to_csv(RESULTS / "olmoe_stabilization.csv", index=False)
    stab_avg = stab.groupby("checkpoint")[["top1_match_rate", "top8_overlap"]].mean()
    stab_avg = stab_avg.reindex(early).reset_index()
    stab_avg.to_csv(RESULTS / "olmoe_stabilization_avg.csv", index=False)

    print("[D] Computing OLMoE expert load skew at final...")
    skew_rows = []
    counts_by_layer = {}
    for L in range(nl):
        counts = np.bincount(final[:, L, :].ravel(), minlength=ne).astype(np.float64)
        counts_by_layer[L] = counts
        sorted_desc = np.sort(counts)[::-1]
        skew_rows.append({
            "layer": L,
            "max_over_mean": counts.max() / counts.mean(),
            "gini": gini(counts),
            "top8_traffic_share_pct": 100.0 * sorted_desc[:8].sum() / counts.sum(),
            "n_experts_used": int(np.sum(counts > 0)),
        })
    skew = pd.DataFrame(skew_rows)
    skew.to_csv(RESULTS / "olmoe_load_skew.csv", index=False)

    xlabels = [f"{c}" for c in early]
    xpos = range(len(early))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(xpos, 100 * stab_avg["top1_match_rate"], "o-", color=plotting.PALETTE[0],
            lw=2.2, label="top-1 match")
    ax.plot(xpos, 100 * stab_avg["top8_overlap"], "s-", color=plotting.PALETTE[1],
            lw=2.2, label="top-8 overlap")
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Checkpoint (training steps) — vs final")
    ax.set_ylabel("Agreement with final model (%)")
    ax.set_title("OLMoE routing stabilization (avg over 16 layers)")
    ax.legend()
    plotting.save(fig, RESULTS / "fig1_olmoe_stabilization.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    pivot = stab.pivot(index="layer", columns="checkpoint", values="top8_overlap")
    pivot = pivot[early]  # column order
    im = ax.imshow(100 * pivot.values, aspect="auto", cmap="viridis",
                   vmin=0, vmax=100)
    ax.set_xticks(range(len(early)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(nl))
    ax.set_yticklabels(range(nl))
    ax.set_xlabel("Checkpoint (steps)")
    ax.set_ylabel("MoE layer")
    ax.set_title("OLMoE top-8 overlap with final (%)")
    fig.colorbar(im, ax=ax, label="overlap %")
    plotting.save(fig, RESULTS / "fig2_olmoe_stabilization_heatmap.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1.bar(range(nl), skew["gini"], color=plotting.PALETTE[4], alpha=0.85)
    ax1.set_xlabel("MoE layer")
    ax1.set_ylabel("Gini coefficient")
    ax1.set_title("OLMoE expert-load skew by layer (final)")
    worst_L = int(skew.loc[skew["gini"].idxmax(), "layer"])
    c = counts_by_layer[worst_L]
    share = 100.0 * np.sort(c)[::-1] / c.sum()
    ax2.bar(range(ne), share, color=plotting.PALETTE[0], width=1.0)
    ax2.axhline(100.0 / ne, color="#d1495b", ls="--", lw=1, label="even (1.56%)")
    ax2.set_xlabel("Experts (sorted by load)")
    ax2.set_ylabel("Traffic share (%)")
    ax2.set_title(f"Load distribution, most-skewed layer (L{worst_L})")
    ax2.legend()
    plotting.save(fig, RESULTS / "fig3_olmoe_load_skew.png")

    first = stab_avg.iloc[0]
    headline = {
        "model": "OLMoE-1B-7B",
        "n_tokens": int(m),
        "earliest_checkpoint_steps": early[0],
        "earliest_top1_match_pct": 100 * first["top1_match_rate"],
        "earliest_top8_overlap_pct": 100 * first["top8_overlap"],
        "mean_gini_final": skew["gini"].mean(),
        "mean_top8_share_final_pct": skew["top8_traffic_share_pct"].mean(),
        "even_share_pct": 100.0 / ne,
    }
    pd.DataFrame([headline]).to_csv(RESULTS / "headline.csv", index=False)

    print("\n=== Experiment D headline ===")
    for kk, v in headline.items():
        print(f"  {kk}: {v}")


if __name__ == "__main__":
    main()
