"""Rebuild Exp E per-layer volume figures as individual PDF/SVG/PNG."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "per_layer"
ORIG = Path(__file__).resolve().parent / "expE_original"
E_RESULTS = ROOT / "exp_E_volume_stabilization" / "results"
sys.path.insert(0, str(ROOT))

from common import plotting  # noqa: E402

P = plotting.PALETTE
SIZE = (8.0, 5.0)


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        plotting.save(fig, OUT / f"{stem}.{ext}", close=False)
    plt.close(fig)


def copy_originals() -> None:
    ORIG.mkdir(parents=True, exist_ok=True)
    names = [
        "fig2_norm_l1_vs_final.png",
        "fig3_consecutive_change.png",
        "fig6_olmoe_l1_vs_final_by_layer.png",
        "fig7_olmoe_consecutive_change_by_layer.png",
    ]
    for name in names:
        src = E_RESULTS / name
        if src.exists():
            shutil.copy2(src, ORIG / name)


def flame_l1_vs_final() -> None:
    df = pd.read_csv(E_RESULTS / "volume_stability.csv")
    layers = list(df["layer"].unique())
    fig, ax = plt.subplots(figsize=SIZE)
    for j, layer in enumerate(layers):
        sub = df[df["layer"] == layer]
        ax.plot(sub["checkpoint"], sub["norm_l1_vs_final"], "o-",
                color=P[j % len(P)], lw=1.8, label=layer)
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel(r"Normalised L1 distance vs final  ($\Sigma|\Delta p|$)")
    ax.set_title("Expert-volume distance by layer in FLAME-MoE-290M")
    ax.legend(ncol=2, fontsize=8)
    save(fig, "flame_volume_l1_vs_final_by_layer")


def flame_consecutive() -> None:
    df = pd.read_csv(E_RESULTS / "volume_stability.csv")
    layers = list(df["layer"].unique())
    fig, ax = plt.subplots(figsize=SIZE)
    for j, layer in enumerate(layers):
        sub = df[df["layer"] == layer].iloc[:-1]
        ax.plot(sub["checkpoint"], sub["norm_l1_vs_next"], "o-",
                color=P[j % len(P)], lw=1.8, label=layer)
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel("Normalised L1 between consecutive checkpoints")
    ax.set_title("Step-to-step volume change shrinks as training proceeds")
    ax.legend(ncol=2, fontsize=8)
    save(fig, "flame_volume_consecutive_by_layer")


def olmoe_l1_vs_final() -> None:
    df = pd.read_csv(E_RESULTS / "olmoe_volume_stability_by_layer.csv")
    layers = sorted(df["layer"].unique())
    ckpts = list(df["checkpoint"].unique())
    xpos = np.arange(len(ckpts))
    labels = [str(int(c)) if str(c) != "final" else "final" for c in ckpts]
    nl = len(layers)
    norm = Normalize(vmin=0, vmax=nl - 1)
    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=SIZE)
    for L in layers:
        ys = df[df["layer"] == L]["norm_l1_vs_final"].values
        ax.plot(xpos, ys, "o-", color=cmap(norm(int(L))), lw=1.5, alpha=0.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("OLMoE checkpoint (training steps)")
    ax.set_ylabel(r"Normalised L1 distance vs final  ($\Sigma|\Delta p|$)")
    ax.set_title("Expert-volume distance by layer in OLMoE-1B-7B")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="MoE layer")
    save(fig, "olmoe_volume_l1_vs_final_by_layer")


def olmoe_consecutive() -> None:
    """Recompute consecutive L1 from cached 205k traces (same method as Exp E)."""
    import config
    from common.olmoe_loader import OLMOE_CHECKPOINTS, load_checkpoint as olmoe_load

    ne = config.OLMOE_NUM_EXPERTS
    nl = config.OLMOE_NUM_LAYERS
    n = config.SAMPLE_OLMOE
    data = {c: olmoe_load(c, n) for c in OLMOE_CHECKPOINTS}
    m = min(arr.shape[0] for arr in data.values())
    dist = {}
    for ckpt in OLMOE_CHECKPOINTS:
        arr = data[ckpt]
        per_layer = []
        for L in range(nl):
            c = np.bincount(arr[:m, L, :].ravel(), minlength=ne).astype(np.float64)
            per_layer.append(c / c.sum())
        dist[ckpt] = per_layer

    pairs = list(zip(OLMOE_CHECKPOINTS[:-1], OLMOE_CHECKPOINTS[1:]))
    xpos = np.arange(len(pairs))
    labels = [f"{a}→{b}" for a, b in pairs]
    norm = Normalize(vmin=0, vmax=nl - 1)
    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=SIZE)
    for L in range(nl):
        ys = [float(np.abs(dist[a][L] - dist[b][L]).sum()) for a, b in pairs]
        ax.plot(xpos, ys, "o-", color=cmap(norm(L)), lw=1.5, alpha=0.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Checkpoint transition")
    ax.set_ylabel("Normalised L1 between consecutive checkpoints")
    ax.set_title("OLMoE step-to-step volume change shrinks (per layer)")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="MoE layer")
    save(fig, "olmoe_volume_consecutive_by_layer")


def main() -> None:
    plotting.apply_style()
    plt.rcParams["savefig.dpi"] = 200
    copy_originals()
    flame_l1_vs_final()
    flame_consecutive()
    olmoe_l1_vs_final()
    olmoe_consecutive()
    print(f"per-layer -> {OUT}")
    print(f"originals -> {ORIG}")


if __name__ == "__main__":
    main()
