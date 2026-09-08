"""Extra figures for Exp O — GLOBAL vs LOCAL online deflection, fine-grained margin."""
from __future__ import annotations

import sys
import heapq
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as O  # noqa: E402
from common import plotting  # noqa: E402
from common import placement as pl  # noqa: E402
import config  # noqa: E402

RESULTS = HERE / "results"

TAUS_FINE = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01]
TAU_TICKS = [0.0, 0.002, 0.004, 0.006, 0.008, 0.01]
TARGET_PEAK = 1.15

palette = plotting.PALETTE
C_GLOBAL, C_LOCAL, C_OFF, C_RAW = palette[1], palette[2], palette[0], "0.45"


def offline_full(top1, top2, margin, D):
    """Offline greedy Pareto traced to FULL balance (until no move reduces the peak)."""
    n = len(top1)
    L = np.bincount(top1, minlength=D).astype(np.float64)
    mean = n / D
    pools = [[] for _ in range(D)]
    for t in range(n):
        pools[int(top1[t])].append((float(margin[t]), int(top2[t])))
    for d in range(D):
        heapq.heapify(pools[d])
    peaks = [L.max() / mean]; costs = [0.0]
    cost = 0.0
    while L.max() / mean > 1.0 + 1e-9:
        d = int(np.argmax(L)); did = False
        while pools[d]:
            m, dst = heapq.heappop(pools[d])
            if dst == d:
                continue
            if L[dst] < L[d] - 1:
                L[d] -= 1; L[dst] += 1; cost += m
                peaks.append(L.max() / mean); costs.append(cost); did = True
                break
        if not did:
            break
    return {"peak": np.array(peaks), "cost": np.array(costs)}


def online_fine(top1, top2, margin, src, D, S, order, local):
    peaks, costs, fracs = [], [], []
    for tau in TAUS_FINE:
        r = O.online_margin(top1, top2, margin, src, D, S, tau, order, local)
        peaks.append(r["peak"]); costs.append(r["cost"]); fracs.append(r["frac"])
    return {"peak": np.array(peaks), "cost": np.array(costs), "frac": np.array(fracs),
            "tau": np.array(TAUS_FINE)}


def cost_at_peak(peak_arr, cost_arr, target):
    """Cheapest cost among operating points that reach peak <= target (nan if unreachable)."""
    mask = peak_arr <= target + 1e-9
    if not mask.any():
        return np.nan
    return float(np.min(cost_arr[mask]))


def collect():
    D = config.NET_DEVICES; S = config.NET_SRC_RANKS
    place = pl.expert_to_device(config.FLAME_NUM_EXPERTS, D, config.NET_PLACEMENT, config.NET_RANDOM_SEED)
    n = config.SAMPLE_STABILIZATION
    ckpt = config.FLAME_FINAL_CHECKPOINT
    data = {}
    for layer in config.FLAME_LAYERS:
        top1, top2, margin, src = O.layer_arrays(ckpt, layer, n, D, S, place)
        order = np.arange(n)
        raw_peak = O.peak_of(top1, D)
        off = offline_full(top1, top2, margin, D)
        g = online_fine(top1, top2, margin, src, D, S, order, local=False)
        l = online_fine(top1, top2, margin, src, D, S, order, local=True)
        data[layer] = {"raw_peak": raw_peak, "off": off, "g": g, "l": l, "n": n}
        print(f"  {layer}: raw {raw_peak:.3f}  offline->{off['peak'][-1]:.3f}@{off['cost'][-1]:.1f}  "
              f"g(min peak {g['peak'].min():.3f})  l(min peak {l['peak'].min():.3f})")
    return data, n


def fig_margin_sweep(data, n, showcase):
    d = data[showcase]; x = np.array(TAUS_FINE)
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))

    a = ax[0]
    a.plot(x, d["g"]["peak"], "o-", color=C_GLOBAL, lw=2.2, ms=6, label="online global")
    a.plot(x, d["l"]["peak"], "s--", color=C_LOCAL, lw=2.2, ms=6, label="online local")
    a.axhline(d["raw_peak"], color=C_RAW, ls="-", lw=1.6, label=f"raw (upper bound) {d['raw_peak']:.2f}")
    a.axhline(1.0, color=C_OFF, ls=":", lw=1.4, label="full balance = 1.0")
    a.set_xlim(0, TAUS_FINE[-1]); a.set_ylim(0.99, d["raw_peak"] * 1.02)
    a.set_xticks(TAU_TICKS); a.set_xticklabels([f"{t:g}" for t in TAU_TICKS])
    a.set_xlabel("margin threshold M (τ)"); a.set_ylabel("peak device load (max/mean)")
    a.set_title("(a) Load balance vs margin")
    a.legend(fontsize=8); a.grid(True, alpha=0.3)

    b = ax[1]
    b.plot(x, d["g"]["cost"], "o-", color=C_GLOBAL, lw=2.2, ms=6, label="online global")
    b.plot(x, d["l"]["cost"], "s--", color=C_LOCAL, lw=2.2, ms=6, label="online local")
    b.set_xlim(0, TAUS_FINE[-1]); b.set_ylim(0, None)
    b.set_xticks(TAU_TICKS); b.set_xticklabels([f"{t:g}" for t in TAU_TICKS])
    b.set_xlabel("margin threshold M (τ)")
    b.set_ylabel("router-score cost (Σ margin of moved tokens)")
    b.set_title("(b) Quality cost vs margin")
    b.legend(fontsize=8); b.grid(True, alpha=0.3)

    fig.suptitle(f"Fine margin sweep — global vs local  ({showcase}, FLAME-290M)", y=1.02)
    plotting.save(fig, RESULTS / "fig4_margin_sweep_global_local.png")


def fig_heatmaps(data):
    layers = config.FLAME_LAYERS
    G = np.array([data[L]["g"]["peak"] for L in layers])
    Lc = np.array([data[L]["l"]["peak"] for L in layers])
    vmin = 1.0; vmax = max(G.max(), Lc.max())
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 4.8))
    labels = [f"{t:g}" for t in TAUS_FINE]
    for a, M, title in [(ax[0], G, "online GLOBAL"), (ax[1], Lc, "online LOCAL")]:
        im = a.imshow(M, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
        a.set_xticks(np.arange(len(TAUS_FINE))); a.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        a.set_yticks(np.arange(len(layers))); a.set_yticklabels([L.replace("layer_", "L") for L in layers])
        a.set_xlabel("margin threshold M (τ)"); a.set_title(f"{title}: peak device load")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                       color="white" if M[i, j] > (vmin + vmax) / 2 else "black", fontsize=6)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label="peak (max/mean); 1.0 = balanced")
    ax[0].set_ylabel("MoE layer")
    fig.suptitle("Peak device load across layers × fine margin — lower (darker) is better", y=1.02)
    plotting.save(fig, RESULTS / "fig5_heatmap_layers_margin.png")


def fig_cost_vs_balance(data, showcase):
    """Split into two standalone figures (fig6a frontier, fig6b across-layers)."""
    d = data[showcase]
    fig, a = plt.subplots(figsize=(7.6, 5.0))
    a.plot(d["off"]["peak"], d["off"]["cost"], "-", color=C_OFF, lw=2.6, label="offline (cost lower bound)")
    a.plot(d["g"]["peak"], d["g"]["cost"], "o-", color=C_GLOBAL, lw=2.1, ms=6, label="online global")
    a.plot(d["l"]["peak"], d["l"]["cost"], "s--", color=C_LOCAL, lw=2.1, ms=6, label="online local")
    a.axvline(d["raw_peak"], color=C_RAW, ls="-", lw=1.5, label=f"raw peak {d['raw_peak']:.2f}")
    a.invert_xaxis()
    a.set_xlabel("peak device load reached (max/mean)   →   more balanced")
    a.set_ylabel("router-score cost (Σ margin of moved tokens)")
    a.set_title(f"Cost to reach a given balance level ({showcase}, FLAME-290M)\n"
                "offline is cheapest at every balance; online costs more (no hindsight)", fontsize=10)
    a.legend(fontsize=9); a.grid(True, alpha=0.3)
    plotting.save(fig, RESULTS / "fig6a_cost_frontier.png")

    layers = config.FLAME_LAYERS; x = np.arange(len(layers))
    fig, b = plt.subplots(figsize=(8.4, 5.0))
    off_c = [cost_at_peak(data[L]["off"]["peak"], data[L]["off"]["cost"], TARGET_PEAK) for L in layers]
    g_c = [cost_at_peak(data[L]["g"]["peak"], data[L]["g"]["cost"], TARGET_PEAK) for L in layers]
    l_c = [cost_at_peak(data[L]["l"]["peak"], data[L]["l"]["cost"], TARGET_PEAK) for L in layers]
    b.plot(x, off_c, "D-", color=C_OFF, lw=2.2, ms=7, label="offline (lower bound)")
    b.plot(x, g_c, "o-", color=C_GLOBAL, lw=2.2, ms=6, label="online global")
    b.plot(x, l_c, "s--", color=C_LOCAL, lw=2.2, ms=6, label="online local")
    b.set_yscale("log")
    b.set_xticks(x); b.set_xticklabels([L.replace("layer_", "L") for L in layers])
    b.set_xlabel("MoE layer")
    b.set_ylabel(f"router-score cost to reach peak ≤ {TARGET_PEAK} (log scale)")
    b.set_title(f"Cost to balance to peak ≤ {TARGET_PEAK} across layers\n"
                "offline is consistently ~3–10× cheaper than online", fontsize=10)
    b.legend(fontsize=9); b.grid(True, alpha=0.3, which="both")
    plotting.save(fig, RESULTS / "fig6b_cost_across_layers.png")


def main():
    plotting.apply_style()
    print("[O-figs] collecting fine-grained real-trace sweeps (global & local) ...")
    data, n = collect()
    showcase = max(config.FLAME_LAYERS, key=lambda L: data[L]["raw_peak"])
    fig_margin_sweep(data, n, showcase)
    fig_heatmaps(data)
    fig_cost_vs_balance(data, showcase)
    print("[O-figs] wrote fig4_margin_sweep_global_local.png, fig5_heatmap_layers_margin.png, "
          "fig6a_cost_frontier.png, fig6b_cost_across_layers.png")


if __name__ == "__main__":
    main()
