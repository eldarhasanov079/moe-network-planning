"""Level 3 visual: earliest vs final dispatch-matrix heatmaps (one layer each)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "l3_l4"
H_RESULTS = ROOT / "exp_H_link_stabilization" / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common"))

import config  # noqa: E402
from common import placement as pl  # noqa: E402
from common import plotting  # noqa: E402
from common import traffic_matrix as tmx  # noqa: E402
from common.flame_loader import load_layer  # noqa: E402
from common.olmoe_loader import load_checkpoint as olmoe_load  # noqa: E402

SIZE = (8.4, 4.0)


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        plotting.save(fig, OUT / f"{stem}.{ext}", close=False)
    plt.close(fig)


def matrix(indices: np.ndarray) -> np.ndarray:
    place = pl.expert_to_device(
        config.FLAME_NUM_EXPERTS, config.NET_DEVICES,
        config.NET_PLACEMENT, config.NET_RANDOM_SEED,
    )
    src = pl.token_source_ranks(len(indices), config.NET_SRC_RANKS, config.NET_SHARDING)
    M = tmx.build_dispatch_matrix(
        indices, place, src, config.NET_SRC_RANKS, config.NET_DEVICES,
        dedup_device=config.NET_DEDUP_DEVICE,
    )
    return M / M.sum()


def heatmap_pair(M0, Mf, *, xlabel, titles, stem, title):
    vmax = max(M0.max(), Mf.max())
    fig, axes = plt.subplots(1, 2, figsize=SIZE, constrained_layout=True)
    for ax, M, t in zip(axes, (M0, Mf), titles):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("Destination device")
        ax.set_ylabel("Source rank")
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="Share of dispatch slots")
    fig.suptitle(title, fontweight="bold")
    save(fig, stem)


def copy_l4() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("fig3_link_load_skew.png", "L4_skew_flame_original.png"),
        ("olmoe_fig3_link_load_skew.png", "L4_skew_olmoe_original.png"),
        ("olmoe_fig4_hot_link_overlap.png", "L4_hotlinks_olmoe_original.png"),
    ]
    for src, dst in pairs:
        shutil.copy2(H_RESULTS / src, OUT / dst)


def main() -> None:
    plotting.apply_style()
    plt.rcParams["savefig.dpi"] = 200
    copy_l4()

    n = config.SAMPLE_STABILIZATION
    flame_layer = "layer_06"
    M0 = matrix(load_layer(540, flame_layer, n)[1])
    Mf = matrix(load_layer(5473, flame_layer, n)[1])
    heatmap_pair(
        M0, Mf,
        xlabel="device",
        titles=("iteration 540", "iteration 5473 (final)"),
        stem="L3_matrix_heatmap_flame",
        title=f"Dispatch matrix for {flame_layer} in FLAME-MoE-290M",
    )

    n_o = config.SAMPLE_OLMOE
    layer = 8
    early = olmoe_load("5000", n_o)
    final = olmoe_load("final", n_o)
    m = min(early.shape[0], final.shape[0])
    M0 = matrix(early[:m, layer, :])
    Mf = matrix(final[:m, layer, :])
    heatmap_pair(
        M0, Mf,
        xlabel="device",
        titles=("5,000 steps", "final"),
        stem="L3_matrix_heatmap_olmoe",
        title=f"Dispatch matrix for layer {layer} in OLMoE-1B-7B",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
