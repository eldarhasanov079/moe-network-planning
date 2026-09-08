"""Experiment E - Volume-Level Traffic Stabilization (FLAME-MoE-290M)."""

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
from common.olmoe_loader import load_checkpoint as olmoe_load, OLMOE_CHECKPOINTS  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def load_distribution(checkpoint: int, layer: str, n_rows: int, ne: int) -> np.ndarray:
    """Per-expert traffic volume as a probability distribution (sums to 1)."""
    _, indices = load_layer(checkpoint, layer, n_rows)
    counts = np.bincount(indices.ravel(), minlength=ne).astype(np.float64)
    return counts / counts.sum()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def norm_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Per-row top-k set overlap fraction between two (N, k) index arrays."""
    eq = (a[:, :, None] == b[:, None, :]).any(axis=2)
    return eq.sum(axis=1) / k


def run_olmoe() -> None:
    """Cross-ARCHITECTURE check: same volume-stabilization analysis on OLMoE-1B-7B."""
    import matplotlib.pyplot as plt
    ne = config.OLMOE_NUM_EXPERTS
    nl = config.OLMOE_NUM_LAYERS
    k = config.OLMOE_TOP_K
    n = config.SAMPLE_OLMOE

    print("\n[E] OLMoE cross-architecture volume stabilization...")
    data = {c: olmoe_load(c, n) for c in OLMOE_CHECKPOINTS}  # (n_tokens, 16, 8)
    final = data["final"]
    m = min(arr.shape[0] for arr in data.values())
    early = [c for c in OLMOE_CHECKPOINTS if c != "final"]

    dist = {}
    for ckpt in OLMOE_CHECKPOINTS:
        arr = data[ckpt]
        per_layer = []
        for L in range(nl):
            c = np.bincount(arr[:m, L, :].ravel(), minlength=ne).astype(np.float64)
            per_layer.append(c / c.sum())
        dist[ckpt] = per_layer
    final_dist = dist["final"]

    rows = []
    l1_vs_final_per_layer = {}
    for ckpt in OLMOE_CHECKPOINTS:
        arr = data[ckpt]
        cos_list, l1_list, t_list = [], [], []
        for L in range(nl):
            p = dist[ckpt][L]
            cos_list.append(cosine(p, final_dist[L]))
            l1_list.append(norm_l1(p, final_dist[L]))
            t_list.append(float(np.mean(topk_overlap(arr[:m, L, :], final[:m, L, :], k))))
        l1_vs_final_per_layer[ckpt] = np.asarray(l1_list)
        rows.append({
            "checkpoint": ckpt,
            "volume_cosine_vs_final": float(np.mean(cos_list)),
            "volume_norm_l1_vs_final": float(np.mean(l1_list)),
            "routing_top8_overlap_vs_final": float(np.mean(t_list)),
        })
    odf = pd.DataFrame(rows)
    odf.to_csv(RESULTS / "olmoe_volume_stability.csv", index=False)

    consec_pairs = list(zip(OLMOE_CHECKPOINTS[:-1], OLMOE_CHECKPOINTS[1:]))
    consec_l1 = {}  # (a,b) -> array(nl)
    for a, b in consec_pairs:
        consec_l1[(a, b)] = np.asarray([norm_l1(dist[a][L], dist[b][L]) for L in range(nl)])

    detail = []
    for ckpt in OLMOE_CHECKPOINTS:
        for L in range(nl):
            detail.append({"checkpoint": ckpt, "layer": L,
                           "norm_l1_vs_final": float(l1_vs_final_per_layer[ckpt][L])})
    pd.DataFrame(detail).to_csv(RESULTS / "olmoe_volume_stability_by_layer.csv", index=False)

    xpos = range(len(OLMOE_CHECKPOINTS))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(xpos, odf["volume_cosine_vs_final"], "o-", color=plotting.PALETTE[0],
            lw=2.2, label="volume cosine vs final")
    ax.plot(xpos, odf["routing_top8_overlap_vs_final"], "s--", color=plotting.PALETTE[1],
            lw=2.0, label="routing top-8 overlap vs final")
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(OLMOE_CHECKPOINTS)
    ax.set_xlabel("OLMoE checkpoint (training steps)")
    ax.set_ylabel("Agreement with final model (layer-avg)")
    ax.set_title("OLMoE-1B-7B: volume stabilises above routing (cross-architecture check)")
    ax.legend()
    plotting.save(fig, RESULTS / "fig5_olmoe_volume_stab.png")

    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=nl - 1)
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(8, 5))
    for L in range(nl):
        ys = [l1_vs_final_per_layer[ckpt][L] for ckpt in OLMOE_CHECKPOINTS]
        ax.plot(xpos, ys, "o-", color=cmap(norm(L)), lw=1.5, alpha=0.85)
    ax.set_xticks(list(xpos)); ax.set_xticklabels(OLMOE_CHECKPOINTS)
    ax.set_xlabel("OLMoE checkpoint (training steps)")
    ax.set_ylabel("Normalised L1 distance vs final  (Σ|Δp|)")
    ax.set_title("OLMoE per-layer volume distance from final")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="MoE layer")
    plotting.save(fig, RESULTS / "fig6_olmoe_l1_vs_final_by_layer.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    pair_pos = range(len(consec_pairs))
    pair_labels = [f"{a}\u2192{b}" for a, b in consec_pairs]
    for L in range(nl):
        ys = [consec_l1[(a, b)][L] for a, b in consec_pairs]
        ax.plot(pair_pos, ys, "o-", color=cmap(norm(L)), lw=1.5, alpha=0.85)
    ax.set_xticks(list(pair_pos)); ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_xlabel("Checkpoint transition")
    ax.set_ylabel("Normalised L1 between consecutive checkpoints")
    ax.set_title("OLMoE step-to-step volume change shrinks (per layer)")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="MoE layer")
    plotting.save(fig, RESULTS / "fig7_olmoe_consecutive_change_by_layer.png")

    print("=== Experiment E (OLMoE) ===")
    print(odf.to_string(index=False))
    e0 = odf.iloc[0]
    print(f"\nOLMoE HEADLINE: at the earliest checkpoint ({early[0]} steps) the per-expert "
          f"volume distribution already matches final with cosine {e0['volume_cosine_vs_final']:.4f} "
          f"(L1 {e0['volume_norm_l1_vs_final']:.4f}), while routing top-8 overlap is only "
          f"{e0['routing_top8_overlap_vs_final']:.3f} — same volume>routing pattern as FLAME.")


def main() -> None:
    plotting.apply_style()
    import matplotlib.pyplot as plt

    layers = config.FLAME_LAYERS
    checkpoints = config.FLAME_CHECKPOINTS
    final = config.FLAME_FINAL_CHECKPOINT
    ne = config.FLAME_NUM_EXPERTS
    n = config.SAMPLE_STABILIZATION

    print("[E] Building per-expert volume distributions...")
    dist = {layer: {} for layer in layers}
    for layer in layers:
        for ckpt in checkpoints:
            dist[layer][ckpt] = load_distribution(ckpt, layer, n, ne)

    rows = []
    for layer in layers:
        p_final = dist[layer][final]
        for i, ckpt in enumerate(checkpoints):
            p = dist[layer][ckpt]
            cos_final = cosine(p, p_final)
            l1_final = norm_l1(p, p_final)
            if i + 1 < len(checkpoints):
                p_next = dist[layer][checkpoints[i + 1]]
                cos_next = cosine(p, p_next)
                l1_next = norm_l1(p, p_next)
            else:
                cos_next, l1_next = np.nan, np.nan
            rows.append({
                "layer": layer, "checkpoint": ckpt,
                "cosine_vs_final": cos_final, "norm_l1_vs_final": l1_final,
                "cosine_vs_next": cos_next, "norm_l1_vs_next": l1_next,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "volume_stability.csv", index=False)

    avg = df.groupby("checkpoint")[["cosine_vs_final", "norm_l1_vs_final",
                                    "cosine_vs_next", "norm_l1_vs_next"]].mean().reset_index()
    avg.to_csv(RESULTS / "volume_stability_avg.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, layer in enumerate(layers):
        sub = df[df["layer"] == layer]
        ax.plot(sub["checkpoint"], sub["cosine_vs_final"], "o-",
                color=plotting.PALETTE[j % len(plotting.PALETTE)], lw=1.8, label=layer)
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel("Cosine similarity of volume vector vs final")
    ax.set_title("Volume-level stabilization: cosine(volume$_c$, volume$_{final}$)")
    ax.legend(ncol=2, fontsize=8)
    plotting.save(fig, RESULTS / "fig1_cosine_vs_final.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, layer in enumerate(layers):
        sub = df[df["layer"] == layer]
        ax.plot(sub["checkpoint"], sub["norm_l1_vs_final"], "o-",
                color=plotting.PALETTE[j % len(plotting.PALETTE)], lw=1.8, label=layer)
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel("Normalised L1 distance vs final  (Σ|Δp|)")
    ax.set_title("Volume-level stabilization: how far is each checkpoint from final?")
    ax.legend(ncol=2, fontsize=8)
    plotting.save(fig, RESULTS / "fig2_norm_l1_vs_final.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, layer in enumerate(layers):
        sub = df[df["layer"] == layer].iloc[:-1]
        ax.plot(sub["checkpoint"], sub["norm_l1_vs_next"], "o-",
                color=plotting.PALETTE[j % len(plotting.PALETTE)], lw=1.8, label=layer)
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel("Normalised L1 between consecutive checkpoints")
    ax.set_title("Step-to-step volume change shrinks as training proceeds")
    ax.legend(ncol=2, fontsize=8)
    plotting.save(fig, RESULTS / "fig3_consecutive_change.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(avg["checkpoint"], avg["cosine_vs_final"], "o-",
            color=plotting.PALETTE[0], lw=2.2, label="volume cosine vs final (Exp E)")
    expb = ROOT / "exp_B_routing_stabilization" / "results" / "stabilization_layer_avg.csv"
    if expb.exists():
        b = pd.read_csv(expb)
        ax.plot(b["checkpoint"], b["top6_overlap"], "s--",
                color=plotting.PALETTE[1], lw=2.2, label="routing top-6 overlap (Exp B)")
        ax.plot(b["checkpoint"], b["top1_match_rate"], "^:",
                color=plotting.PALETTE[3], lw=2.0, label="routing top-1 match (Exp B)")
    ax.set_xlabel("Training checkpoint (iteration)")
    ax.set_ylabel("Agreement with final model")
    ax.set_title("Aggregate VOLUME stabilises earlier/higher than per-token ROUTING")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / "fig4_volume_vs_routing.png")

    first = avg.iloc[0]
    headline = {
        "model": config.FLAME_MODEL,
        "metric": "per-expert traffic volume distribution",
        "n_tokens_per_layer": n,
        "first_ckpt": checkpoints[0],
        "first_ckpt_cosine_vs_final": first["cosine_vs_final"],
        "first_ckpt_norm_l1_vs_final": first["norm_l1_vs_final"],
        "final_minus1_cosine_vs_final": float(avg.iloc[-2]["cosine_vs_final"]),
        "final_minus1_norm_l1_vs_final": float(avg.iloc[-2]["norm_l1_vs_final"]),
    }
    pd.DataFrame([headline]).to_csv(RESULTS / "headline.csv", index=False)

    print("\n=== Experiment E headline ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    print(f"\nHEADLINE: From the very first checkpoint (iter {checkpoints[0]}), the per-expert "
          f"traffic-volume distribution already matches the final model with cosine "
          f"{first['cosine_vs_final']:.4f} (normalised L1 {first['norm_l1_vs_final']:.4f}). "
          f"Volume is far more stable than per-token routing.")

    run_olmoe()


if __name__ == "__main__":
    main()
