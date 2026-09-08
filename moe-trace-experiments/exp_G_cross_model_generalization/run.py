"""Experiment G - Cross-Model Generalization (FLAME 290M / 721M / 1.7B)."""

from __future__ import annotations

import heapq
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

MODELS = ["flame-moe-290m", "flame-moe-721m", "flame-moe-1.7b"]
SAMPLE = 250_000
NE = 64
TOPK = 6
MARGIN_THRESHOLDS = (0.03, 0.05)


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def topk_overlap(a, b, k):
    eq = (a[:, :, None] == b[:, None, :]).any(axis=2)
    return eq.sum(axis=1) / k


def rebalance_excess_removed(top1, top2, margin, ne):
    """Greedy single-step deflection; return % of primary-load excess removed."""
    L = np.bincount(top1, minlength=ne).astype(np.int64)
    mean = len(top1) / ne
    init_peak = L.max() / mean
    pools = [[] for _ in range(ne)]
    for t in range(len(top1)):
        pools[top1[t]].append((float(margin[t]), int(top2[t])))
    for e in range(ne):
        heapq.heapify(pools[e])
    while True:
        e = int(np.argmax(L))
        moved = False
        pool = pools[e]
        while pool:
            m, d = heapq.heappop(pool)
            if L[d] < L[e] - 1:
                L[e] -= 1
                L[d] += 1
                moved = True
                break
        if not moved:
            break
    floor_peak = L.max() / mean
    if init_peak <= 1.0:
        return init_peak, floor_peak, 0.0
    return init_peak, floor_peak, 100.0 * (init_peak - floor_peak) / (init_peak - 1.0)


def analyse_model(model: str):
    meta = config.FLAME_MODELS[model]
    layers = meta["layers"]
    checkpoints = meta["checkpoints"]
    final = checkpoints[-1]
    print(f"\n[G] === {model}: {len(layers)} layers x {len(checkpoints)} checkpoints ===", flush=True)

    final_scores, final_indices = {}, {}
    for layer in layers:
        s, i = load_layer(final, layer, SAMPLE, model=model)
        final_scores[layer] = s
        final_indices[layer] = i

    all_margins = []
    ginis, primary_peaks, excess_removed = [], [], []
    for layer in layers:
        s, i = final_scores[layer], final_indices[layer]
        all_margins.append((s[:, 0] - s[:, 1]).astype(np.float64))
        counts = np.bincount(i.ravel(), minlength=NE).astype(np.float64)
        ginis.append(gini(counts))
        top1 = i[:, 0].astype(np.int64)
        top2 = i[:, 1].astype(np.int64)
        margin = (s[:, 0] - s[:, 1]).astype(np.float64)
        init_p, _, exc = rebalance_excess_removed(top1, top2, margin, NE)
        primary_peaks.append(init_p)
        excess_removed.append(exc)
    all_margins = np.concatenate(all_margins)

    final_metrics = {
        "model": model,
        "n_layers": len(layers),
        "pct_margin_below_0.03": 100.0 * np.mean(all_margins < 0.03),
        "pct_margin_below_0.05": 100.0 * np.mean(all_margins < 0.05),
        "mean_margin": float(all_margins.mean()),
        "mean_gini_top6_membership": float(np.mean(ginis)),
        "mean_primary_top1_peak": float(np.mean(primary_peaks)),
        "mean_pct_excess_removed": float(np.mean(excess_removed)),
    }

    stab_rows = []
    for ckpt in checkpoints:
        cos_list, l1_list, t1_list, t6_list = [], [], [], []
        for layer in layers:
            s, i = load_layer(ckpt, layer, SAMPLE, model=model)
            p = np.bincount(i.ravel(), minlength=NE).astype(np.float64)
            p /= p.sum()
            pf = np.bincount(final_indices[layer].ravel(), minlength=NE).astype(np.float64)
            pf /= pf.sum()
            cos_list.append(cosine(p, pf))
            l1_list.append(float(np.abs(p - pf).sum()))
            m = min(len(i), len(final_indices[layer]))
            t1_list.append(float(np.mean(i[:m, 0] == final_indices[layer][:m, 0])))
            t6_list.append(float(np.mean(topk_overlap(i[:m], final_indices[layer][:m], TOPK))))
        stab_rows.append({
            "model": model,
            "checkpoint": ckpt,
            "progress": ckpt / final,
            "volume_cosine_vs_final": float(np.mean(cos_list)),
            "volume_norm_l1_vs_final": float(np.mean(l1_list)),
            "routing_top1_match": float(np.mean(t1_list)),
            "routing_top6_overlap": float(np.mean(t6_list)),
        })
    return final_metrics, pd.DataFrame(stab_rows)


def main():
    plotting.apply_style()
    import matplotlib.pyplot as plt

    final_rows = []
    stab_all = []
    for model in MODELS:
        fm, sdf = analyse_model(model)
        final_rows.append(fm)
        stab_all.append(sdf)

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(RESULTS / "cross_model_final_metrics.csv", index=False)
    stab_df = pd.concat(stab_all, ignore_index=True)
    stab_df.to_csv(RESULTS / "cross_model_stabilization.csv", index=False)

    short = {"flame-moe-290m": "290M", "flame-moe-721m": "721M", "flame-moe-1.7b": "1.7B"}
    colors = {m: plotting.PALETTE[j] for j, m in enumerate(MODELS)}

    anomalies = stab_df[stab_df["volume_cosine_vs_final"] < 0.9]

    def annotate_anomalies(ax, ycol):
        for _, r in anomalies.iterrows():
            ax.annotate(f"router-collapse\nspike (iter {int(r['checkpoint'])})",
                        xy=(r["progress"], r[ycol]),
                        xytext=(r["progress"] + 0.04, r[ycol] + (0.15 if ycol.endswith("cosine_vs_final") else 0.0)),
                        fontsize=8, color="#d1495b",
                        arrowprops=dict(arrowstyle="->", color="#d1495b", lw=1))

    # ============================ FIGURES ============================
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in MODELS:
        d = stab_df[stab_df["model"] == m]
        ax.plot(d["progress"], d["volume_cosine_vs_final"], "o-",
                color=colors[m], lw=2, label=short[m])
    annotate_anomalies(ax, "volume_cosine_vs_final")
    ax.set_xlabel("Training progress (iteration / final iteration)")
    ax.set_ylabel("Volume cosine vs final (layer-avg)")
    ax.set_title("Volume-level stabilization generalises across model size")
    ax.legend(title="model")
    plotting.save(fig, RESULTS / "fig1_volume_stab_cross_model.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in MODELS:
        d = stab_df[stab_df["model"] == m]
        ax.plot(d["progress"], d["volume_norm_l1_vs_final"], "o-",
                color=colors[m], lw=2, label=short[m])
    ax.set_xlabel("Training progress (iteration / final iteration)")
    ax.set_ylabel("Volume normalised L1 vs final (layer-avg)")
    ax.set_title("Volume distance from final shrinks for all model sizes")
    ax.legend(title="model")
    plotting.save(fig, RESULTS / "fig2_volume_l1_cross_model.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in MODELS:
        d = stab_df[stab_df["model"] == m]
        ax.plot(d["progress"], d["routing_top6_overlap"], "o-",
                color=colors[m], lw=2, label=f"{short[m]} routing top-6")
        ax.plot(d["progress"], d["volume_cosine_vs_final"], "s--",
                color=colors[m], lw=1.5, alpha=0.7)
    annotate_anomalies(ax, "routing_top6_overlap")
    ax.set_xlabel("Training progress (iteration / final iteration)")
    ax.set_ylabel("Agreement with final model")
    ax.set_title("Volume (dashed) stabilises above routing (solid) at every size")
    ax.legend(fontsize=8)
    plotting.save(fig, RESULTS / "fig3_routing_vs_volume_cross_model.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    xs = np.arange(len(MODELS))
    labels = [short[m] for m in MODELS]
    axes[0].bar(xs - 0.2, final_df["pct_margin_below_0.03"], width=0.4,
                color=plotting.PALETTE[0], label="<0.03")
    axes[0].bar(xs + 0.2, final_df["pct_margin_below_0.05"], width=0.4,
                color=plotting.PALETTE[2], label="<0.05")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("% of tokens")
    axes[0].set_title("Low-margin tokens (deflection headroom)")
    axes[0].legend()
    axes[1].bar(xs - 0.2, final_df["mean_gini_top6_membership"], width=0.4,
                color=plotting.PALETTE[4], label="Gini (top-6 membership)")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Gini")
    ax1b = axes[1].twinx()
    ax1b.plot(xs, final_df["mean_primary_top1_peak"], "o-", color=plotting.PALETTE[1],
              lw=2, label="primary top-1 peak (max/mean)")
    ax1b.set_ylabel("primary peak (max/mean)")
    ax1b.grid(False)
    axes[1].set_title("Load skew vs model size")
    axes[2].bar(xs, final_df["mean_pct_excess_removed"], width=0.5,
                color=plotting.PALETTE[3])
    axes[2].set_xticks(xs); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("% primary-load excess removed")
    axes[2].set_title("Margin-gated deflection effectiveness")
    fig.suptitle("Final-checkpoint headline metrics across FLAME model sizes", fontweight="bold")
    plotting.save(fig, RESULTS / "fig4_final_metrics_cross_model.png")

    # ============================ PRINT ============================
    print("\n=== Experiment G: cross-model final metrics ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
