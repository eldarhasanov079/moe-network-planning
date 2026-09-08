"""Experiment I - Stabilization-Gated Planned-Routing Simulation (Approach #1, Exp 1C)."""

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

POLICY_ORDER = ["ecmp", "static_mean", "static_p95", "static_worst", "oracle"]
POLICY_LABELS = {
    "ecmp": "ECMP (oblivious)", "static_mean": "static mean (frozen)",
    "static_p95": "static P95 (frozen)", "static_worst": "static worst (frozen)",
    "oracle": "oracle (per-step)",
}


def build_matrix_for(indices, place, sharding):
    src = pl.token_source_ranks(len(indices), config.NET_SRC_RANKS, sharding)
    return tmx.build_dispatch_matrix(indices, place, src, config.NET_SRC_RANKS,
                                     config.NET_DEVICES, dedup_device=config.NET_DEDUP_DEVICE)


def envelopes_for_layer(hist):
    """Return decision matrices (mean / P95 / worst) over a profiling-window history."""
    stack = np.stack(hist, 0).astype(np.float64)
    return {
        "static_mean": stack.mean(0),
        "static_p95": nm.build_envelope(hist, 95),
        "static_worst": nm.build_envelope(hist, 100),
    }


def run_sim(*, label, prefix, get_indices, layers, layer_labels, checkpoints,
            ckpt_labels, final, profile_slice, title_suffix):
    import matplotlib.pyplot as plt

    R = config.NET_RAILS
    seeds = config.SIM_ECMP_SEEDS
    place = pl.expert_to_device(64, config.NET_DEVICES, config.NET_PLACEMENT, config.NET_RANDOM_SEED)
    p0, p1 = profile_slice
    profile_ckpts = checkpoints[p0:p1]
    test_ckpts = checkpoints[p1:]

    print(f"\n[I] === {label} ===  devices={config.NET_DEVICES}  rails={R}  "
          f"ECMP seeds={seeds}  layers={len(layers)}  "
          f"profile={[ckpt_labels[checkpoints.index(c)] for c in profile_ckpts]}  "
          f"test={[ckpt_labels[checkpoints.index(c)] for c in test_ckpts]}")

    mats = {lab: {} for lab in layer_labels}
    needed = list(dict.fromkeys(profile_ckpts + test_ckpts))
    for layer, lab in zip(layers, layer_labels):
        for ckpt in needed:
            mats[lab][ckpt] = build_matrix_for(get_indices(layer, ckpt), place, config.NET_SHARDING)

    frozen = {lab: {} for lab in layer_labels}
    for lab in layer_labels:
        decs = envelopes_for_layer([mats[lab][c] for c in profile_ckpts])
        for pol, dec in decs.items():
            frozen[lab][pol] = ns.greedy_assign(dec, R)

    per_rows = []
    over_time = {pol: {c: [] for c in test_ckpts} for pol in POLICY_ORDER}
    for lab in layer_labels:
        for c in test_ckpts:
            M = mats[lab][c]
            A_oracle = ns.greedy_assign(M, R)
            vals = {
                "static_mean": ns.normalized_time(frozen[lab]["static_mean"], M, R),
                "static_p95": ns.normalized_time(frozen[lab]["static_p95"], M, R),
                "static_worst": ns.normalized_time(frozen[lab]["static_worst"], M, R),
                "oracle": ns.normalized_time(A_oracle, M, R),
            }
            ecmp_mean, ecmp_p95 = ns.ecmp_normalized_time(M, R, seeds)
            vals["ecmp"] = ecmp_mean
            for pol in POLICY_ORDER:
                per_rows.append({"layer": lab, "checkpoint": ckpt_labels[checkpoints.index(c)],
                                 "policy": pol, "normalized_time": vals[pol]})
                over_time[pol][c].append(vals[pol])
            per_rows.append({"layer": lab, "checkpoint": ckpt_labels[checkpoints.index(c)],
                             "policy": "ecmp_p95", "normalized_time": ecmp_p95})
    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(RESULTS / f"{prefix}sim_detail.csv", index=False)

    summ = per_df.groupby("policy")["normalized_time"].mean()
    ecmp_norm = float(summ["ecmp"]); oracle_norm = float(summ["oracle"])
    rows = []
    for pol in POLICY_ORDER:
        nt = float(summ[pol])
        rows.append({
            "policy": pol, "label": POLICY_LABELS[pol], "normalized_time": nt,
            "speedup_vs_ecmp": ecmp_norm / nt if nt > 0 else np.nan,
            "gap_to_oracle_pct": 100.0 * (nt - oracle_norm) / oracle_norm if oracle_norm > 0 else 0.0,
        })
    summ_df = pd.DataFrame(rows)
    summ_df.to_csv(RESULTS / f"{prefix}sim_policy_summary.csv", index=False)

    sweep_rows = []
    for Rs in config.SIM_RAIL_SWEEP:
        e_list, p_list, o_list = [], [], []
        for lab in layer_labels:
            decs = envelopes_for_layer([mats[lab][c] for c in profile_ckpts])
            A_p95 = ns.greedy_assign(decs["static_p95"], Rs)
            for c in test_ckpts:
                M = mats[lab][c]
                o_list.append(ns.normalized_time(ns.greedy_assign(M, Rs), M, Rs))
                p_list.append(ns.normalized_time(A_p95, M, Rs))
                e_list.append(ns.ecmp_normalized_time(M, Rs, max(64, seeds // 4))[0])
        sweep_rows.append({"rails": Rs, "ecmp": float(np.mean(e_list)),
                           "static_p95": float(np.mean(p_list)), "oracle": float(np.mean(o_list))})
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RESULTS / f"{prefix}sim_rail_sweep.csv", index=False)

    # ---- FIGURES ----
    palette = plotting.PALETTE
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(POLICY_ORDER))
    ax.bar(x, summ_df["normalized_time"], color=[palette[i % len(palette)] for i in range(len(x))])
    ax.axhline(1.0, color="black", ls=":", lw=1, label="ideal (perfectly balanced)")
    for xi, nt in zip(x, summ_df["normalized_time"]):
        ax.annotate(f"{nt:.2f}", (xi, nt), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([POLICY_LABELS[p] for p in POLICY_ORDER],
                                         rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Normalized All-to-All time (max link load / ideal)")
    ax.set_title(f"Planned routing vs ECMP{title_suffix}")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / f"{prefix}fig1_time_by_policy.png")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    xt = np.arange(len(test_ckpts))
    test_labels = [ckpt_labels[checkpoints.index(c)] for c in test_ckpts]
    for pol, style, col in [("ecmp", "^--", palette[1]), ("static_p95", "o-", palette[0]),
                            ("oracle", "s:", palette[2])]:
        ys = [float(np.mean(over_time[pol][c])) for c in test_ckpts]
        ax.plot(xt, ys, style, color=col, lw=2, label=POLICY_LABELS[pol])
    ax.axhline(1.0, color="black", ls=":", lw=1)
    ax.set_xticks(xt); ax.set_xticklabels(test_labels, rotation=45 if len(test_labels) > 6 else 0, fontsize=8)
    ax.set_xlabel("Held-out checkpoint (plan frozen from earlier profiling window)")
    ax.set_ylabel("Normalized All-to-All time")
    ax.set_title(f"Frozen P95 plan stays near oracle on later checkpoints{title_suffix}")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / f"{prefix}fig2_frozen_plan_over_time.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(sweep_df["rails"], sweep_df["ecmp"], "^--", color=palette[1], lw=2, label="ECMP")
    ax.plot(sweep_df["rails"], sweep_df["static_p95"], "o-", color=palette[0], lw=2, label="static P95 (frozen)")
    ax.plot(sweep_df["rails"], sweep_df["oracle"], "s:", color=palette[2], lw=2, label="oracle")
    ax.axhline(1.0, color="black", ls=":", lw=1)
    ax.set_xlabel("Number of rails R"); ax.set_ylabel("Normalized All-to-All time")
    ax.set_title(f"Sensitivity to rail count{title_suffix}")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / f"{prefix}fig3_rail_sweep.png")

    # ---- HEADLINE ----
    p95 = summ_df[summ_df["policy"] == "static_p95"].iloc[0]
    headline = {
        "model": label, "devices": config.NET_DEVICES, "rails": R,
        "ecmp_normalized_time": ecmp_norm, "oracle_normalized_time": oracle_norm,
        "static_p95_normalized_time": float(p95["normalized_time"]),
        "static_p95_speedup_vs_ecmp": float(p95["speedup_vs_ecmp"]),
        "static_p95_gap_to_oracle_pct": float(p95["gap_to_oracle_pct"]),
    }
    pd.DataFrame([headline]).to_csv(RESULTS / f"{prefix}headline.csv", index=False)
    print(f"\n=== Experiment I headline ({label}) ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    print(f"HEADLINE [{label}]: frozen P95 plan cuts normalized All-to-All time from ECMP's "
          f"{ecmp_norm:.2f}x to {float(p95['normalized_time']):.2f}x "
          f"({float(p95['speedup_vs_ecmp']):.2f}x speedup), within "
          f"{float(p95['gap_to_oracle_pct']):.1f}% of the per-step oracle ({oracle_norm:.2f}x).")
    return summ_df


def run_flame():
    n = config.SAMPLE_STABILIZATION

    def get_indices(layer, ckpt):
        return load_layer(ckpt, layer, n)[1]

    return run_sim(label="flame-moe-290m", prefix="", get_indices=get_indices,
                   layers=config.FLAME_LAYERS, layer_labels=list(config.FLAME_LAYERS),
                   checkpoints=config.FLAME_CHECKPOINTS, ckpt_labels=[str(c) for c in config.FLAME_CHECKPOINTS],
                   final=config.FLAME_FINAL_CHECKPOINT, profile_slice=(1, 6),
                   title_suffix="  (FLAME-MoE-290M)")


def run_olmoe():
    nl = config.OLMOE_NUM_LAYERS
    n = config.SAMPLE_OLMOE
    print("\n[I] OLMoE: loading cached checkpoints (n=%d)..." % n)
    data = {c: olmoe_load(c, n) for c in OLMOE_CHECKPOINTS}
    m = min(arr.shape[0] for arr in data.values())

    def get_indices(layer, ckpt):
        return data[ckpt][:m, layer, :]

    return run_sim(label="olmoe-1b-7b", prefix="olmoe_", get_indices=get_indices,
                   layers=list(range(nl)), layer_labels=[f"layer_{i:02d}" for i in range(nl)],
                   checkpoints=list(OLMOE_CHECKPOINTS), ckpt_labels=list(OLMOE_CHECKPOINTS),
                   final="final", profile_slice=(0, 3), title_suffix="  (OLMoE-1B-7B)")


def main():
    plotting.apply_style()
    import matplotlib.pyplot as plt

    flame = run_flame()
    olmoe = run_olmoe()

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(POLICY_ORDER))
    w = 0.38
    ax.bar(x - w / 2, [flame[flame["policy"] == p]["normalized_time"].iloc[0] for p in POLICY_ORDER],
           width=w, color=plotting.PALETTE[0], label="FLAME-MoE-290M")
    ax.bar(x + w / 2, [olmoe[olmoe["policy"] == p]["normalized_time"].iloc[0] for p in POLICY_ORDER],
           width=w, color=plotting.PALETTE[3], label="OLMoE-1B-7B")
    ax.axhline(1.0, color="black", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([POLICY_LABELS[p] for p in POLICY_ORDER],
                                         rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Normalized All-to-All time")
    ax.set_title("Planned routing vs ECMP across both model families")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / "fig4_cross_model_summary.png")


if __name__ == "__main__":
    main()
