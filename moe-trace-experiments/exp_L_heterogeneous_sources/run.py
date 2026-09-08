"""Experiment L - Heterogeneous Sources: when does matrix planning beat volume planning?"""

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
from common import netmetrics as nm  # noqa: E402
from common import netsim as ns  # noqa: E402
from common import placement as pl  # noqa: E402
from common import traffic_matrix as tmx  # noqa: E402
from common.flame_loader import load_layer  # noqa: E402
from common.olmoe_loader import load_checkpoint as olmoe_load, OLMOE_CHECKPOINTS  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
ECMP_SEEDS = 128


def hetero_src(d_ref, S, alpha, seed):
    """token -> source rank."""
    n = len(d_ref)
    rng = np.random.default_rng(seed)
    affinity = (d_ref - config.HETERO_OFFSET) % S
    rand = rng.integers(0, S, size=n)
    use_aff = rng.random(n) < alpha
    return np.where(use_aff, affinity, rand).astype(np.int64)


def run_hetero(*, label, prefix, get_indices, layers, layer_labels, checkpoints,
               ckpt_labels, profile_slice, title_suffix):
    import matplotlib.pyplot as plt

    D = config.NET_DEVICES; S = config.NET_SRC_RANKS; R = config.NET_RAILS
    place = pl.expert_to_device(64, D, config.NET_PLACEMENT, config.NET_RANDOM_SEED)
    p0, p1 = profile_slice
    profile_ckpts = checkpoints[p0:p1]
    test_ckpts = checkpoints[p1:]
    ref_ckpt = profile_ckpts[0]
    final = checkpoints[-1]
    needed = list(dict.fromkeys(profile_ckpts + test_ckpts + [final]))

    print(f"\n[L] === {label} ===  D={D} rails={R}  ref={ckpt_labels[checkpoints.index(ref_ckpt)]}  "
          f"alphas={config.HETERO_ALPHAS}")

    idx = {lab: {} for lab in layer_labels}
    for layer, lab in zip(layers, layer_labels):
        for c in needed:
            idx[lab][c] = get_indices(layer, c)
    n_common = min(len(idx[lab][c]) for lab in layer_labels for c in needed)
    for lab in layer_labels:
        for c in needed:
            idx[lab][c] = idx[lab][c][:n_common]
    d_ref = {lab: place[idx[lab][ref_ckpt][:, 0].astype(np.int64)] for lab in layer_labels}

    rows = []
    for alpha in config.HETERO_ALPHAS:
        src = {lab: hetero_src(d_ref[lab], S, alpha, seed=config.NET_RANDOM_SEED)
               for lab in layer_labels}
        mats = {lab: {c: tmx.build_dispatch_matrix(idx[lab][c], place, src[lab], S, D,
                                                   dedup_device=config.NET_DEDUP_DEVICE)
                      for c in needed} for lab in layer_labels}

        resid, cos_final = [], []
        ecmp_t, vol_t, mat_t, orc_t = [], [], [], []
        for lab in layer_labels:
            mean_M = np.mean(np.stack([mats[lab][c] for c in profile_ckpts], 0), 0)
            A_matrix = ns.greedy_assign(mean_M, R)                  # knows full joint
            A_volume = ns.greedy_assign(nm.volume_approx(mean_M), R)  # knows only marginals
            Mf = mats[lab][final]
            for c in test_ckpts:
                M = mats[lab][c]
                resid.append(nm.rank1_residual(M))
                cos_final.append(nm.cosine(tmx.as_distribution(M), tmx.as_distribution(Mf)))
                ecmp_t.append(ns.ecmp_normalized_time(M, R, ECMP_SEEDS)[0])
                vol_t.append(ns.normalized_time(A_volume, M, R))
                mat_t.append(ns.normalized_time(A_matrix, M, R))
                orc_t.append(ns.normalized_time(ns.greedy_assign(M, R), M, R))
        rows.append({
            "alpha": alpha,
            "rank1_residual": float(np.mean(resid)),
            "matrix_cosine_vs_final": float(np.mean(cos_final)),
            "ecmp_time": float(np.mean(ecmp_t)),
            "volume_plan_time": float(np.mean(vol_t)),
            "matrix_plan_time": float(np.mean(mat_t)),
            "oracle_time": float(np.mean(orc_t)),
            "matrix_vs_volume_gain_pct": 100.0 * (np.mean(vol_t) - np.mean(mat_t)) / np.mean(vol_t),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{prefix}hetero_sweep.csv", index=False)

    palette = plotting.PALETTE
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(df["alpha"], df["rank1_residual"], "o-", color=palette[0], lw=2,
            label="rank-1 residual (structure beyond volume)")
    ax.plot(df["alpha"], 1 - df["matrix_cosine_vs_final"], "s--", color=palette[1], lw=2,
            label="1 - matrix cosine vs final (instability)")
    ax.set_xlabel("Source heterogeneity alpha"); ax.set_ylabel("Residual / instability")
    ax.set_title(f"Heterogeneity adds structure but stays stable{title_suffix}")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / f"{prefix}fig1_structure_stability.png")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(df["alpha"], df["ecmp_time"], "^--", color=palette[1], lw=2, label="ECMP")
    ax.plot(df["alpha"], df["volume_plan_time"], "D-.", color=palette[3], lw=2, label="volume plan (marginals only)")
    ax.plot(df["alpha"], df["matrix_plan_time"], "o-", color=palette[0], lw=2, label="matrix plan (full joint)")
    ax.plot(df["alpha"], df["oracle_time"], "s:", color=palette[2], lw=2, label="oracle")
    ax.axhline(1.0, color="black", ls=":", lw=1)
    ax.set_xlabel("Source heterogeneity alpha"); ax.set_ylabel("Normalized All-to-All time")
    ax.set_title(f"Matrix planning earns its keep as sources specialize{title_suffix}")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / f"{prefix}fig2_matrix_vs_volume_plan.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(df["alpha"], df["matrix_vs_volume_gain_pct"], "o-", color=palette[0], lw=2)
    ax.axhline(0.0, color="black", ls=":", lw=1)
    ax.set_xlabel("Source heterogeneity alpha")
    ax.set_ylabel("Matrix-plan speedup over volume-plan (%)")
    ax.set_title(f"Value of knowing the joint matrix{title_suffix}")
    plotting.save(fig, RESULTS / f"{prefix}fig3_matrix_gain.png")

    hi = df.iloc[-1]; lo = df.iloc[0]
    print(f"\n=== Experiment L headline ({label}) ===")
    print(df.to_string(index=False))
    print(f"\nHEADLINE [{label}]: at alpha=0 (homogeneous, as in the public traces) the matrix "
          f"residual is {lo['rank1_residual']:.3f} and matrix-plan ~ volume-plan "
          f"({lo['matrix_plan_time']:.2f}x vs {lo['volume_plan_time']:.2f}x, "
          f"{lo['matrix_vs_volume_gain_pct']:.1f}% gain). At alpha=1 (domain-specialized) the "
          f"residual rises to {hi['rank1_residual']:.3f}, the matrix stays stable "
          f"(cosine-vs-final {hi['matrix_cosine_vs_final']:.3f}), and matrix-plan beats "
          f"volume-plan by {hi['matrix_vs_volume_gain_pct']:.1f}% ({hi['matrix_plan_time']:.2f}x "
          f"vs {hi['volume_plan_time']:.2f}x, oracle {hi['oracle_time']:.2f}x).")
    return df


def run_flame():
    n = config.SAMPLE_STABILIZATION

    def get_indices(layer, ckpt):
        return load_layer(ckpt, layer, n)[1]

    return run_hetero(label="flame-moe-290m", prefix="", get_indices=get_indices,
                      layers=config.FLAME_LAYERS, layer_labels=list(config.FLAME_LAYERS),
                      checkpoints=config.FLAME_CHECKPOINTS,
                      ckpt_labels=[str(c) for c in config.FLAME_CHECKPOINTS],
                      profile_slice=(1, 6), title_suffix="  (FLAME-MoE-290M)")


def run_olmoe():
    nl = config.OLMOE_NUM_LAYERS
    n = config.SAMPLE_OLMOE
    print("\n[L] OLMoE: loading cached checkpoints (n=%d)..." % n)
    data = {c: olmoe_load(c, n) for c in OLMOE_CHECKPOINTS}
    m = min(arr.shape[0] for arr in data.values())

    def get_indices(layer, ckpt):
        return data[ckpt][:m, layer, :]

    return run_hetero(label="olmoe-1b-7b", prefix="olmoe_", get_indices=get_indices,
                      layers=list(range(nl)), layer_labels=[f"layer_{i:02d}" for i in range(nl)],
                      checkpoints=list(OLMOE_CHECKPOINTS), ckpt_labels=list(OLMOE_CHECKPOINTS),
                      profile_slice=(0, 3), title_suffix="  (OLMoE-1B-7B)")


def main():
    plotting.apply_style()
    flame = run_flame()
    olmoe = run_olmoe()
    pd.concat([flame.assign(model="flame-290m"), olmoe.assign(model="olmoe-1b-7b")]) \
        .to_csv(RESULTS / "hetero_sweep_both.csv", index=False)


if __name__ == "__main__":
    main()
