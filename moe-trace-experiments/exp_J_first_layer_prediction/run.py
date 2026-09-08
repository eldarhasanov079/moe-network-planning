"""Experiment J - First-Layer-Triggered Cross-Layer Prediction (Approach #2)."""

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
from common import topology as tp  # noqa: E402
from common import traffic_matrix as tmx  # noqa: E402
from common.flame_loader import load_layer  # noqa: E402
from common.olmoe_loader import load_checkpoint as olmoe_load, OLMOE_CHECKPOINTS  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def build_matrix_for(indices, place, sharding):
    src = pl.token_source_ranks(len(indices), config.NET_SRC_RANKS, sharding)
    return tmx.build_dispatch_matrix(indices, place, src, config.NET_SRC_RANKS,
                                     config.NET_DEVICES, dedup_device=config.NET_DEDUP_DEVICE)


def run_pred(*, label, prefix, get_indices, layers, layer_labels, checkpoints,
             ckpt_labels, profile_slice, title_suffix):
    import matplotlib.pyplot as plt

    D = config.NET_DEVICES
    R = config.NET_RAILS
    seeds = config.SIM_ECMP_SEEDS
    place = pl.expert_to_device(64, D, config.NET_PLACEMENT, config.NET_RANDOM_SEED)
    fabric = tp.FullMeshFabric(D)
    p0, p1 = profile_slice
    profile_ckpts = checkpoints[p0:p1]
    test_ckpts = checkpoints[p1:]
    trigger_idx = 0

    print(f"\n[J] === {label} ===  layers={len(layers)}  trigger={layer_labels[0]}  "
          f"test_ckpts={[ckpt_labels[checkpoints.index(c)] for c in test_ckpts]}")

    mats = {lab: {} for lab in layer_labels}
    needed = list(dict.fromkeys(profile_ckpts + test_ckpts))
    for layer, lab in zip(layers, layer_labels):
        for ckpt in needed:
            mats[lab][ckpt] = build_matrix_for(get_indices(layer, ckpt), place, config.NET_SHARDING)

    history = {lab: np.mean(np.stack([mats[lab][c] for c in profile_ckpts], 0), 0) for lab in layer_labels}

    trig_lab = layer_labels[trigger_idx]
    later_labels = layer_labels[1:]

    rows = []
    per_layer_cos = {lab: [] for lab in later_labels}
    for c in test_ckpts:
        trig = mats[trig_lab][c]
        for lab in later_labels:
            actual = mats[lab][c]
            a_dist = tmx.as_distribution(actual)
            a_link = fabric.link_load_array(actual)
            preds = {
                "firstlayer_identity": mats[trig_lab][c],
                "firstlayer_transpose": mats[trig_lab][c].T,
                "history_mean": history[lab],
                "uniform": np.ones_like(actual),
            }
            for name, P in preds.items():
                rows.append({
                    "checkpoint": ckpt_labels[checkpoints.index(c)], "layer": lab, "predictor": name,
                    "cosine": nm.cosine(tmx.as_distribution(P), a_dist),
                    "norm_l1": nm.norm_l1(tmx.as_distribution(P), a_dist),
                    "hot_link_overlap": nm.hot_link_overlap(fabric.link_load_array(P), a_link, 0.25),
                })
            per_layer_cos[lab].append(nm.cosine(tmx.as_distribution(trig), a_dist))
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(RESULTS / f"{prefix}prediction_quality.csv", index=False)
    pred_avg = pred_df.groupby("predictor")[["cosine", "norm_l1", "hot_link_overlap"]].mean().reset_index()
    pred_avg.to_csv(RESULTS / f"{prefix}prediction_quality_avg.csv", index=False)

    sim_rows = []
    for c in test_ckpts:
        trig = mats[trig_lab][c]
        A_trig = ns.greedy_assign(trig, R)
        for lab in later_labels:
            actual = mats[lab][c]
            A_hist = ns.greedy_assign(history[lab], R)
            A_oracle = ns.greedy_assign(actual, R)
            ecmp_mean, _ = ns.ecmp_normalized_time(actual, R, seeds)
            sim_rows.append({
                "checkpoint": ckpt_labels[checkpoints.index(c)], "layer": lab,
                "ecmp": ecmp_mean,
                "firstlayer_plan": ns.normalized_time(A_trig, actual, R),
                "history_plan": ns.normalized_time(A_hist, actual, R),
                "oracle": ns.normalized_time(A_oracle, actual, R),
            })
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(RESULTS / f"{prefix}sim_detail.csv", index=False)
    sim_avg = sim_df[["ecmp", "firstlayer_plan", "history_plan", "oracle"]].mean()
    sim_avg.to_frame("normalized_time").to_csv(RESULTS / f"{prefix}sim_summary.csv")

    # ---- FIGURES ----
    palette = plotting.PALETTE
    order = ["firstlayer_identity", "firstlayer_transpose", "history_mean", "uniform"]
    olabels = ["first-layer\n(identity, #2)", "first-layer\n(transpose)", "history mean\n(#1)", "uniform"]

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = np.arange(len(order))
    cvals = [pred_avg[pred_avg["predictor"] == p]["cosine"].iloc[0] for p in order]
    hvals = [pred_avg[pred_avg["predictor"] == p]["hot_link_overlap"].iloc[0] for p in order]
    ax.bar(x - 0.2, cvals, width=0.4, color=palette[0], label="matrix cosine")
    ax.bar(x + 0.2, hvals, width=0.4, color=palette[3], label="hot-link overlap")
    ax.set_xticks(x); ax.set_xticklabels(olabels, fontsize=8)
    ax.set_ylabel("Agreement with actual later-layer traffic")
    ax.set_title(f"Predicting a later layer's All-to-All{title_suffix}")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / f"{prefix}fig1_prediction_quality.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    keys = ["ecmp", "firstlayer_plan", "history_plan", "oracle"]
    klabels = ["ECMP", "first-layer plan (#2)", "history plan (#1)", "oracle"]
    yvals = [float(sim_avg[k]) for k in keys]
    ax.bar(np.arange(len(keys)), yvals, color=[palette[1], palette[0], palette[2], palette[4]])
    for xi, v in enumerate(yvals):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax.axhline(1.0, color="black", ls=":", lw=1)
    ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(klabels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Normalized All-to-All time")
    ax.set_title(f"Routing a later layer from the first layer vs history{title_suffix}")
    plotting.save(fig, RESULTS / f"{prefix}fig2_sim_plan_source.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    yvals = [float(np.mean(per_layer_cos[lab])) for lab in later_labels]
    ax.plot(np.arange(1, len(later_labels) + 1), yvals, "o-", color=palette[0], lw=1.8)
    ax.set_xlabel("Layer distance from trigger (first MoE layer)")
    ax.set_ylabel("Matrix cosine: first layer vs layer L")
    ax.set_title(f"Does the first layer resemble later layers?{title_suffix}")
    plotting.save(fig, RESULTS / f"{prefix}fig3_cosine_vs_distance.png")

    # ---- HEADLINE ----
    fi = pred_avg[pred_avg["predictor"] == "firstlayer_identity"].iloc[0]
    hm = pred_avg[pred_avg["predictor"] == "history_mean"].iloc[0]
    headline = {
        "model": label,
        "firstlayer_identity_cosine": float(fi["cosine"]),
        "firstlayer_identity_hotlink": float(fi["hot_link_overlap"]),
        "history_mean_cosine": float(hm["cosine"]),
        "sim_ecmp": float(sim_avg["ecmp"]),
        "sim_firstlayer_plan": float(sim_avg["firstlayer_plan"]),
        "sim_history_plan": float(sim_avg["history_plan"]),
        "sim_oracle": float(sim_avg["oracle"]),
    }
    pd.DataFrame([headline]).to_csv(RESULTS / f"{prefix}headline.csv", index=False)
    print(f"\n=== Experiment J headline ({label}) ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    print(f"HEADLINE [{label}]: first-layer matrix predicts a later layer with cosine "
          f"{float(fi['cosine']):.3f} (hot-link {float(fi['hot_link_overlap']):.2f}); "
          f"routing a later layer from the first-layer plan gives normalized time "
          f"{float(sim_avg['firstlayer_plan']):.2f}x vs ECMP {float(sim_avg['ecmp']):.2f}x, "
          f"history-plan {float(sim_avg['history_plan']):.2f}x, oracle {float(sim_avg['oracle']):.2f}x.")
    return pred_avg, sim_avg


def run_flame():
    n = config.SAMPLE_STABILIZATION

    def get_indices(layer, ckpt):
        return load_layer(ckpt, layer, n)[1]

    return run_pred(label="flame-moe-290m", prefix="", get_indices=get_indices,
                    layers=config.FLAME_LAYERS, layer_labels=list(config.FLAME_LAYERS),
                    checkpoints=config.FLAME_CHECKPOINTS,
                    ckpt_labels=[str(c) for c in config.FLAME_CHECKPOINTS],
                    profile_slice=(1, 6), title_suffix="  (FLAME-MoE-290M)")


def run_olmoe():
    nl = config.OLMOE_NUM_LAYERS
    n = config.SAMPLE_OLMOE
    print("\n[J] OLMoE: loading cached checkpoints (n=%d)..." % n)
    data = {c: olmoe_load(c, n) for c in OLMOE_CHECKPOINTS}
    m = min(arr.shape[0] for arr in data.values())

    def get_indices(layer, ckpt):
        return data[ckpt][:m, layer, :]

    return run_pred(label="olmoe-1b-7b", prefix="olmoe_", get_indices=get_indices,
                    layers=list(range(nl)), layer_labels=[f"layer_{i:02d}" for i in range(nl)],
                    checkpoints=list(OLMOE_CHECKPOINTS), ckpt_labels=list(OLMOE_CHECKPOINTS),
                    profile_slice=(0, 3), title_suffix="  (OLMoE-1B-7B)")


def main():
    plotting.apply_style()
    run_flame()
    run_olmoe()


if __name__ == "__main__":
    main()
