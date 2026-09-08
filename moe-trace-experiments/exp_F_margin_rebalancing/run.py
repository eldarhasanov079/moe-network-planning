"""Experiment F - Margin-Gated Load Rebalancing (FLAME-MoE-290M)."""

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


def greedy_peak_shave(top1: np.ndarray, top2: np.ndarray, margin: np.ndarray,
                      ne: int):
    """Trace the Pareto frontier of (cumulative score loss) vs (peak load)."""
    L = np.bincount(top1, minlength=ne).astype(np.int64)
    n_tokens = len(top1)
    mean = n_tokens / ne

    pools: list[list] = [[] for _ in range(ne)]
    for t in range(n_tokens):
        e = top1[t]
        pools[e].append((float(margin[t]), int(top2[t])))
    for e in range(ne):
        heapq.heapify(pools[e])

    cum_cost = 0.0
    n_moved = 0
    peaks = [L.max() / mean]
    costs = [0.0]
    moved = [0]
    avg_cost_moved = [0.0]

    while True:
        e = int(np.argmax(L))
        pool = pools[e]
        did_move = False
        while pool:
            m, d = heapq.heappop(pool)
            if L[d] < L[e] - 1:
                L[e] -= 1
                L[d] += 1
                cum_cost += m
                n_moved += 1
                peaks.append(L.max() / mean)
                costs.append(cum_cost)
                moved.append(n_moved)
                avg_cost_moved.append(cum_cost / n_moved)
                did_move = True
                break
        if not did_move:
            break

    return {
        "peak_ratio": np.asarray(peaks),
        "cum_cost": np.asarray(costs),
        "n_moved": np.asarray(moved),
        "frac_moved": np.asarray(moved) / n_tokens,
        "avg_score_loss_per_moved": np.asarray(avg_cost_moved),
        "avg_score_loss_per_token": np.asarray(costs) / n_tokens,
        "n_tokens": n_tokens,
    }


def excess_removed_pct(initial: float, achieved: float) -> float:
    """% of the peak's *excess over a perfectly balanced layer* that was removed."""
    if initial <= 1.0:
        return 0.0
    return 100.0 * (initial - achieved) / (initial - 1.0)


def metrics_at_budgets(curve: dict, budgets=(0.001, 0.002, 0.005, 0.010, 0.020)):
    """For each avg-score-loss-per-token budget, the best (lowest) peak reached."""
    cost = curve["avg_score_loss_per_token"]
    pr = curve["peak_ratio"]
    init = float(pr[0])
    rows = []
    for b in budgets:
        idx = np.where(cost <= b)[0]
        i = idx[-1] if len(idx) else 0
        rows.append({
            "score_loss_budget_per_token": b,
            "achieved_peak_ratio": float(pr[i]),
            "excess_removed_pct": excess_removed_pct(init, float(pr[i])),
            "frac_tokens_moved_pct": 100.0 * curve["frac_moved"][i],
        })
    return rows


def main() -> None:
    plotting.apply_style()
    import matplotlib.pyplot as plt

    final = config.FLAME_FINAL_CHECKPOINT
    layers = config.FLAME_LAYERS
    ne = config.FLAME_NUM_EXPERTS
    n = config.SAMPLE_MARGIN_FINAL  # 500k cached from Exp A

    print("[F] Running greedy margin-gated peak-shaving per layer...")
    curves = {}
    budget_rows = []
    init_rows = []
    for layer in layers:
        scores, indices = load_layer(final, layer, n)
        top1 = indices[:, 0].astype(np.int64)
        top2 = indices[:, 1].astype(np.int64)
        margin = (scores[:, 0] - scores[:, 1]).astype(np.float64)
        curve = greedy_peak_shave(top1, top2, margin, ne)
        curves[layer] = curve

        init_p = float(curve["peak_ratio"][0])
        floor_p = float(curve["peak_ratio"][-1])
        init_rows.append({
            "layer": layer,
            "initial_peak_ratio": init_p,
            "floor_peak_ratio": floor_p,
            "excess_removed_at_floor_pct": excess_removed_pct(init_p, floor_p),
            "total_frac_moved_pct": 100.0 * curve["frac_moved"][-1],
            "total_avg_score_loss_per_token": float(curve["avg_score_loss_per_token"][-1]),
        })
        for r in metrics_at_budgets(curve):
            budget_rows.append({"layer": layer, **r})

    pd.DataFrame(init_rows).to_csv(RESULTS / "rebalancing_summary.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(RESULTS / "rebalancing_budgets.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, layer in enumerate(layers):
        c = curves[layer]
        ax.plot(100 * c["frac_moved"], c["peak_ratio"],
                color=plotting.PALETTE[j % len(plotting.PALETTE)], lw=1.8, label=layer)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="perfectly balanced")
    ax.set_xlabel("% of tokens deflected (lowest-margin first)")
    ax.set_ylabel("Peak expert load  (max / mean)")
    ax.set_title("Margin-gated peak shaving: hotspot reduction vs tokens moved")
    ax.legend(ncol=2, fontsize=8)
    plotting.save(fig, RESULTS / "fig1_pareto_frac_moved.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, layer in enumerate(layers):
        c = curves[layer]
        ax.plot(c["avg_score_loss_per_token"], c["peak_ratio"],
                color=plotting.PALETTE[j % len(plotting.PALETTE)], lw=1.8, label=layer)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xlabel("Average router-score loss per token (whole layer)")
    ax.set_ylabel("Peak expert load  (max / mean)")
    ax.set_title("Margin-gated peak shaving: hotspot reduction vs quality cost")
    ax.legend(ncol=2, fontsize=8)
    plotting.save(fig, RESULTS / "fig2_pareto_score_loss.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(layers))
    init = [curves[l]["peak_ratio"][0] for l in layers]
    floor = [curves[l]["peak_ratio"][-1] for l in layers]
    ax.bar(x - 0.2, init, width=0.4, color=plotting.PALETTE[1], label="initial peak (max/mean)")
    ax.bar(x + 0.2, floor, width=0.4, color=plotting.PALETTE[2],
           label="best achievable peak (all eligible moves)")
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_ylabel("Peak expert load (max / mean)")
    ax.set_title("How flat can deflection make each layer?")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / "fig3_initial_vs_floor.png")

    show = "layer_09"
    c = curves[show]
    init_p = float(c["peak_ratio"][0])
    floor_p = float(c["peak_ratio"][-1])
    cheap = metrics_at_budgets(c, budgets=(0.005,))[0]
    headline = {
        "model": config.FLAME_MODEL,
        "showcase_layer": show,
        "initial_peak_ratio": init_p,
        "floor_peak_ratio": floor_p,
        "excess_removed_at_floor_pct": excess_removed_pct(init_p, floor_p),
        "total_frac_moved_pct": 100.0 * c["frac_moved"][-1],
        "total_avg_score_loss_per_token": float(c["avg_score_loss_per_token"][-1]),
        "cheap_budget_per_token": 0.005,
        "cheap_achieved_peak_ratio": cheap["achieved_peak_ratio"],
        "cheap_excess_removed_pct": cheap["excess_removed_pct"],
        "cheap_frac_moved_pct": cheap["frac_tokens_moved_pct"],
    }
    pd.DataFrame([headline]).to_csv(RESULTS / "headline.csv", index=False)

    print("\n=== Experiment F headline ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    print(f"\nHEADLINE: On {config.FLAME_MODEL} {show}, single-step margin-gated deflection "
          f"cuts the busiest primary-expert from {init_p:.2f}x to {floor_p:.2f}x mean "
          f"(removing {excess_removed_pct(init_p, floor_p):.0f}% of the excess) by moving "
          f"{100*c['frac_moved'][-1]:.1f}% of tokens at avg score loss "
          f"{float(c['avg_score_loss_per_token'][-1]):.4f}/token.")


if __name__ == "__main__":
    main()
