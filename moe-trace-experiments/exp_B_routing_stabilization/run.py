"""Experiment B - Routing Stabilization Over Training (FLAME-MoE-290M)."""

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


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Per-row overlap fraction between two (N, k) index arrays."""
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    eq = (a[:, :, None] == b[:, None, :]).any(axis=2)
    out = eq.sum(axis=1) / k
    return out


def main() -> None:
    plotting.apply_style()
    import matplotlib.pyplot as plt

    layers = config.FLAME_LAYERS
    checkpoints = config.FLAME_CHECKPOINTS
    final = config.FLAME_FINAL_CHECKPOINT
    k = config.FLAME_TOP_K
    n = config.SAMPLE_STABILIZATION

    print("[B] Loading final-checkpoint routing per layer...")
    final_idx = {}
    for layer in layers:
        _, idx = load_layer(final, layer, n)
        final_idx[layer] = idx

    print("[B] Comparing each checkpoint to the final checkpoint...")
    rows = []
    for ckpt in checkpoints:
        for layer in layers:
            _, idx = load_layer(ckpt, layer, n)
            fin = final_idx[layer]
            m = min(len(idx), len(fin))
            idx, fin = idx[:m], fin[:m]
            top1_match = float(np.mean(idx[:, 0] == fin[:, 0]))
            top6_overlap = float(np.mean(topk_overlap(idx, fin, k)))
            rows.append({
                "checkpoint": ckpt,
                "layer": layer,
                "top1_match_rate": top1_match,
                "top6_overlap": top6_overlap,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "stabilization.csv", index=False)

    avg = df.groupby("checkpoint")[["top1_match_rate", "top6_overlap"]].mean().reset_index()
    avg.to_csv(RESULTS / "stabilization_layer_avg.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, layer in enumerate(layers):
        sub = df[df["layer"] == layer]
        ax.plot(sub["checkpoint"], 100 * sub["top1_match_rate"], "o-",
                color=plotting.PALETTE[j % len(plotting.PALETTE)], lw=1.8, label=layer)
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel("% tokens whose top-1 expert = final top-1")
    ax.set_title("Top-1 routing agreement with the final model")
    ax.legend(ncol=2, fontsize=8)
    plotting.save(fig, RESULTS / "fig1_top1_match_by_layer.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, layer in enumerate(layers):
        sub = df[df["layer"] == layer]
        ax.plot(sub["checkpoint"], 100 * sub["top6_overlap"], "o-",
                color=plotting.PALETTE[j % len(plotting.PALETTE)], lw=1.8, label=layer)
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel("Top-6 expert-set overlap with final (%)")
    ax.set_title("Router saturation: top-6 set overlap with the final model")
    ax.legend(ncol=2, fontsize=8)
    plotting.save(fig, RESULTS / "fig2_top6_overlap_by_layer.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(avg["checkpoint"], 100 * avg["top1_match_rate"], "o-",
            color=plotting.PALETTE[0], lw=2.2, label="top-1 match")
    ax.plot(avg["checkpoint"], 100 * avg["top6_overlap"], "s-",
            color=plotting.PALETTE[1], lw=2.2, label="top-6 overlap")
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel("Agreement with final model (%)")
    ax.set_title("Routing stabilization (averaged over all 8 layers)")
    ax.legend()
    plotting.save(fig, RESULTS / "fig3_stabilization_avg.png")

    first = avg.iloc[0]
    frac_through = 100.0 * checkpoints[0] / final
    headline = {
        "model": config.FLAME_MODEL,
        "final_checkpoint": final,
        "first_checkpoint": checkpoints[0],
        "first_ckpt_pct_through_training": frac_through,
        "first_ckpt_top1_match_pct": 100 * first["top1_match_rate"],
        "first_ckpt_top6_overlap_pct": 100 * first["top6_overlap"],
        "n_tokens_per_layer": n,
    }
    pd.DataFrame([headline]).to_csv(RESULTS / "headline.csv", index=False)

    print("\n=== Experiment B headline ===")
    for k_, v in headline.items():
        print(f"  {k_}: {v}")
    print(f"\nHEADLINE: By iter {checkpoints[0]} (~{frac_through:.0f}% through capture), "
          f"top-6 routing already overlaps the final model by "
          f"{100*first['top6_overlap']:.0f}% on average.")


if __name__ == "__main__":
    main()
