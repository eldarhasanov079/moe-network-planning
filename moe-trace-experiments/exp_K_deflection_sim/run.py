"""Experiment K - Margin-Aware Deflection in the All-to-All simulator (Approach #3, Exp 3B)."""

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
from common import placement as pl  # noqa: E402
from common.flame_loader import load_layer  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
RELIEF_TARGET = 1.05


def device_load_vec(top1_dev, D, n):
    Ld = np.bincount(top1_dev, minlength=D).astype(np.float64)
    return Ld


def _trace(peaks, fracs, costs, Ld, mean, affected, cost, n):
    peaks.append(Ld.max() / mean); fracs.append(affected / n); costs.append(cost)


def margin_deflection(top1_dev, top2_dev, margin, D, n, *, random_seed=None):
    """Deflect tokens off the busiest device to their top-2 device."""
    Ld = device_load_vec(top1_dev, D, n)
    mean = Ld.sum() / D
    if random_seed is None:
        pools = [[] for _ in range(D)]
        for t in range(n):
            pools[int(top1_dev[t])].append((float(margin[t]), t))
        for d in range(D):
            heapq.heapify(pools[d])
        pop = lambda d: heapq.heappop(pools[d])           # noqa: E731
        empty = lambda d: not pools[d]                    # noqa: E731
    else:
        rng = np.random.default_rng(random_seed)
        order = [[] for _ in range(D)]
        for t in range(n):
            order[int(top1_dev[t])].append(t)
        for d in range(D):
            rng.shuffle(order[d])
        ptr = [0] * D
        def pop(d):
            t = order[d][ptr[d]]; ptr[d] += 1
            return (float(margin[t]), t)
        def empty(d):
            return ptr[d] >= len(order[d])

    peaks = [Ld.max() / mean]; fracs = [0.0]; costs = [0.0]
    affected = 0; cost = 0.0
    while True:
        d = int(np.argmax(Ld)); did = False
        while not empty(d):
            m, t = pop(d)
            dst = int(top2_dev[t])
            if dst == d:
                continue
            if Ld[dst] < Ld[d] - 1:
                Ld[d] -= 1; Ld[dst] += 1
                affected += 1; cost += m
                _trace(peaks, fracs, costs, Ld, mean, affected, cost, n)
                did = True
                break
        if not did:
            break
    return {"peak": np.asarray(peaks), "frac": np.asarray(fracs), "scoreloss": np.asarray(costs)}


def token_dropping(top1_dev, top1_score, D, n):
    """Drop the cheapest token on the busiest device (token lost)."""
    Ld = device_load_vec(top1_dev, D, n)
    mean0 = Ld.sum() / D
    pools = [[] for _ in range(D)]
    for t in range(n):
        pools[int(top1_dev[t])].append((float(top1_score[t]), t))
    for d in range(D):
        heapq.heapify(pools[d])
    peaks = [Ld.max() / mean0]; fracs = [0.0]; costs = [0.0]
    affected = 0; cost = 0.0
    while Ld.max() > mean0:
        d = int(np.argmax(Ld))
        if not pools[d]:
            break
        sc, t = heapq.heappop(pools[d])
        Ld[d] -= 1; affected += 1; cost += sc
        _trace(peaks, fracs, costs, Ld, mean0, affected, cost, n)
    return {"peak": np.asarray(peaks), "frac": np.asarray(fracs), "scoreloss": np.asarray(costs)}


def at_relief(par, target):
    """First index reaching peak <= target; return (frac_affected, cumulative_scoreloss)."""
    idx = np.where(par["peak"] <= target)[0]
    i = idx[0] if len(idx) else len(par["peak"]) - 1
    return float(par["frac"][i]), float(par["scoreloss"][i])


def main():
    plotting.apply_style()
    import matplotlib.pyplot as plt

    final = config.FLAME_FINAL_CHECKPOINT
    layers = config.FLAME_LAYERS
    ne = config.FLAME_NUM_EXPERTS
    D = config.NET_DEVICES
    n_sample = config.SAMPLE_STABILIZATION
    place = pl.expert_to_device(ne, D, config.NET_PLACEMENT, config.NET_RANDOM_SEED)

    print(f"[K] Margin-aware deflection in-sim (FLAME-290M, final={final}, D={D}, "
          f"relief target peak<={RELIEF_TARGET})")
    rows = []; pareto_by_layer = {}
    for layer in layers:
        scores, indices = load_layer(final, layer, n_sample)
        n = len(indices)
        top1 = indices[:, 0].astype(np.int64); top2 = indices[:, 1].astype(np.int64)
        margin = (scores[:, 0] - scores[:, 1]).astype(np.float64)
        s1 = scores[:, 0].astype(np.float64)
        top1_dev = place[top1]; top2_dev = place[top2]

        ma = margin_deflection(top1_dev, top2_dev, margin, D, n, random_seed=None)
        rnd = margin_deflection(top1_dev, top2_dev, margin, D, n, random_seed=0)
        drp = token_dropping(top1_dev, s1, D, n)
        pareto_by_layer[layer] = {"margin": ma, "random": rnd, "dropping": drp}

        ma_f, ma_c = at_relief(ma, RELIEF_TARGET)
        rn_f, rn_c = at_relief(rnd, RELIEF_TARGET)
        dr_f, dr_c = at_relief(drp, RELIEF_TARGET)
        rows.append({
            "layer": layer, "init_peak": float(ma["peak"][0]),
            "margin_frac_moved_pct": 100 * ma_f, "margin_scoreloss": ma_c,
            "random_frac_moved_pct": 100 * rn_f, "random_scoreloss": rn_c,
            "dropping_frac_dropped_pct": 100 * dr_f, "dropping_scoreloss": dr_c,
            "dropping_vs_margin_scoreloss_x": (dr_c / ma_c) if ma_c > 0 else np.nan,
            "random_vs_margin_scoreloss_x": (rn_c / ma_c) if ma_c > 0 else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "deflection_summary.csv", index=False)

    showcase = layers[int(np.argmax(df["init_peak"].values))]   # most-skewed layer
    sc = pareto_by_layer[showcase]
    palette = plotting.PALETTE

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(sc["margin"]["scoreloss"], sc["margin"]["peak"], "-", color=palette[0], lw=2,
            label="margin-aware deflection (#3)")
    ax.plot(sc["random"]["scoreloss"], sc["random"]["peak"], "--", color=palette[3], lw=2,
            label="random deflection")
    ax.plot(sc["dropping"]["scoreloss"], sc["dropping"]["peak"], ":", color=palette[1], lw=2,
            label="token dropping")
    ax.axhline(RELIEF_TARGET, color="gray", ls="-.", lw=1, label=f"relief target {RELIEF_TARGET}")
    ax.axhline(1.0, color="black", ls=":", lw=1)
    ax.set_xlabel("Cumulative router-score lost"); ax.set_ylabel("Peak device load (max/mean)")
    ax.set_title(f"Cost of relief: margin-aware vs baselines ({showcase}, FLAME-290M)")
    ax.set_xlim(left=0); ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / "fig1_pareto_scoreloss.png")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for key, st, col, lab in [("margin", "-", palette[0], "margin-aware deflection (#3)"),
                              ("random", "--", palette[3], "random deflection"),
                              ("dropping", ":", palette[1], "token dropping (tokens LOST)")]:
        ax.plot(100 * sc[key]["frac"], sc[key]["peak"], st, color=col, lw=2, label=lab)
    ax.axhline(1.0, color="black", ls=":", lw=1)
    ax.set_xlabel("% of tokens affected"); ax.set_ylabel("Peak device load (max/mean)")
    ax.set_title(f"Relief vs tokens touched ({showcase}, FLAME-290M)")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / "fig2_pareto_frac.png")

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = np.arange(len(layers))
    ax.bar(x - 0.25, df["margin_scoreloss"], width=0.25, color=palette[0], label="margin-aware")
    ax.bar(x, df["random_scoreloss"], width=0.25, color=palette[3], label="random")
    ax.bar(x + 0.25, df["dropping_scoreloss"], width=0.25, color=palette[1], label="token-dropping")
    ax.set_xticks(x); ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"Router-score lost to reach peak<={RELIEF_TARGET}")
    ax.set_title("Margin-aware deflection relieves congestion at the lowest score cost")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / "fig3_scoreloss_per_layer.png")

    srow = df[df["layer"] == showcase].iloc[0]
    headline = {
        "model": config.FLAME_MODEL, "final_checkpoint": final, "devices": D,
        "relief_target_peak": RELIEF_TARGET, "showcase_layer": showcase,
        "init_peak": float(srow["init_peak"]),
        "margin_frac_moved_pct": float(srow["margin_frac_moved_pct"]),
        "margin_scoreloss": float(srow["margin_scoreloss"]),
        "random_scoreloss": float(srow["random_scoreloss"]),
        "dropping_scoreloss": float(srow["dropping_scoreloss"]),
        "dropping_vs_margin_scoreloss_x": float(srow["dropping_vs_margin_scoreloss_x"]),
        "mean_init_peak": float(df["init_peak"].mean()),
    }
    pd.DataFrame([headline]).to_csv(RESULTS / "headline.csv", index=False)
    print("\n=== Experiment K headline ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    print(f"\nHEADLINE: on the most-skewed layer ({showcase}, init peak "
          f"{float(srow['init_peak']):.2f}x), margin-aware deflection reaches peak<={RELIEF_TARGET} "
          f"by moving {float(srow['margin_frac_moved_pct']):.2f}% of tokens at score-loss "
          f"{float(srow['margin_scoreloss']):.1f}, vs token-dropping which loses "
          f"{float(srow['dropping_vs_margin_scoreloss_x']):.0f}x more score mass (and discards "
          f"those tokens). Trained traffic is already balanced (mean init peak "
          f"{float(df['init_peak'].mean()):.2f}x), so absolute headroom is modest.")


if __name__ == "__main__":
    main()
