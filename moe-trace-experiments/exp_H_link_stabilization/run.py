"""Experiment H - Matrix- & Link-Level Stabilization + Envelope/Refund Quality."""

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
from common import placement as pl  # noqa: E402
from common import topology as tp  # noqa: E402
from common import traffic_matrix as tmx  # noqa: E402
from common.flame_loader import load_layer  # noqa: E402
from common.olmoe_loader import load_checkpoint as olmoe_load, OLMOE_CHECKPOINTS  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def build_matrix_for(indices: np.ndarray, place: np.ndarray, sharding: str) -> np.ndarray:
    """Dispatch matrix for one trace file under a given placement/sharding."""
    src = pl.token_source_ranks(len(indices), config.NET_SRC_RANKS, sharding)
    return tmx.build_dispatch_matrix(
        indices, place, src, config.NET_SRC_RANKS, config.NET_DEVICES,
        dedup_device=config.NET_DEDUP_DEVICE,
    )


def rank1_residual(M: np.ndarray) -> float:
    """Fraction of matrix 'energy' NOT explained by the rank-1 outer product"""
    M = M.astype(np.float64)
    tot = M.sum()
    if tot == 0:
        return 0.0
    P = M / tot
    r = P.sum(axis=1, keepdims=True)
    c = P.sum(axis=0, keepdims=True)
    rank1 = r @ c
    return float(np.linalg.norm(P - rank1) / np.linalg.norm(P))


def _per_layer_lineplot(ax, layer_labels, xpos, series_by_layer):
    """Plot one line per layer; switch to a colormap+colorbar when there are many."""
    if len(layer_labels) <= 10:
        for j, lab in enumerate(layer_labels):
            ax.plot(xpos, series_by_layer[lab], "o-",
                    color=plotting.PALETTE[j % len(plotting.PALETTE)], lw=1.8, label=lab)
        ax.legend(ncol=2, fontsize=8)
        return None
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=len(layer_labels) - 1)
    for j, lab in enumerate(layer_labels):
        ax.plot(xpos, series_by_layer[lab], "o-", color=cm.viridis(norm(j)), lw=1.4, alpha=0.85)
    return cm.ScalarMappable(norm=norm, cmap=cm.viridis)


def run_model(*, label, prefix, get_indices, layers, layer_labels, checkpoints,
              ckpt_labels, final, ne, profile_slice, overlay, title_suffix):
    """Compute matrix/link stabilization + envelope quality + robustness for one model."""
    import matplotlib.pyplot as plt

    D = config.NET_DEVICES
    place_default = pl.expert_to_device(ne, D, config.NET_PLACEMENT, config.NET_RANDOM_SEED)
    fabric = tp.FullMeshFabric(D)
    twotier = tp.TwoTierNodes(D, config.NET_NODE_SIZE)
    xpos = np.arange(len(checkpoints))

    print(f"\n[H] === {label} ===  experts={ne}  devices={D}  src_ranks={config.NET_SRC_RANKS}  "
          f"placement={config.NET_PLACEMENT}  sharding={config.NET_SHARDING}  "
          f"layers={len(layers)}  checkpoints={len(checkpoints)}")

    print(f"[H] {label}: building dispatch matrices (default placement)...")
    mats = {lab: {} for lab in layer_labels}
    for layer, lab in zip(layers, layer_labels):
        for ckpt in checkpoints:
            mats[lab][ckpt] = build_matrix_for(get_indices(layer, ckpt), place_default,
                                               config.NET_SHARDING)

    # ---- 1. Matrix-level stabilization --------------------------------
    rows = []
    for lab in layer_labels:
        p_final = tmx.as_distribution(mats[lab][final])
        for i, ckpt in enumerate(checkpoints):
            p = tmx.as_distribution(mats[lab][ckpt])
            rows.append({
                "layer": lab, "checkpoint": ckpt_labels[i],
                "matrix_cosine_vs_final": nm.cosine(p, p_final),
                "matrix_norm_l1_vs_final": nm.norm_l1(p, p_final),
                "rank1_residual": rank1_residual(mats[lab][ckpt]),
            })
    mat_df = pd.DataFrame(rows)
    mat_df.to_csv(RESULTS / f"{prefix}matrix_stability.csv", index=False)
    mat_avg = mat_df.groupby("checkpoint", sort=False)[
        ["matrix_cosine_vs_final", "matrix_norm_l1_vs_final", "rank1_residual"]
    ].mean().reset_index()
    mat_avg.to_csv(RESULTS / f"{prefix}matrix_stability_avg.csv", index=False)

    # ---- 2. Link-level stabilization ----------------------------------
    link_rows = []
    for lab in layer_labels:
        final_fab = fabric.link_load_array(mats[lab][final])
        final_up = twotier.uplink_load_array(mats[lab][final])
        for i, ckpt in enumerate(checkpoints):
            fab_loads = fabric.link_load_array(mats[lab][ckpt])
            up_loads = twotier.uplink_load_array(mats[lab][ckpt])
            link_rows.append({
                "layer": lab, "checkpoint": ckpt_labels[i],
                "fabric_max_over_mean": nm.link_stats(fab_loads)["max_over_mean"],
                "fabric_hot_overlap_vs_final": nm.hot_link_overlap(fab_loads, final_fab, 0.25),
                "uplink_max_over_mean": nm.link_stats(up_loads)["max_over_mean"],
                "uplink_hot_overlap_vs_final": nm.hot_link_overlap(up_loads, final_up, 0.5),
            })
    link_df = pd.DataFrame(link_rows)
    link_df.to_csv(RESULTS / f"{prefix}link_stability.csv", index=False)
    link_avg = link_df.groupby("checkpoint", sort=False)[
        ["fabric_max_over_mean", "fabric_hot_overlap_vs_final",
         "uplink_max_over_mean", "uplink_hot_overlap_vs_final"]
    ].mean().reset_index()
    link_avg.to_csv(RESULTS / f"{prefix}link_stability_avg.csv", index=False)

    p0, p1 = profile_slice
    profile_ckpts = checkpoints[p0:p1]
    test_ckpts = checkpoints[p1:]
    mask = ~np.eye(D, dtype=bool)
    policies = {"mean": None, "P90": 90, "P95": 95, "P99": 99, "worst": 100}
    print(f"[H] {label}: envelope profile={[ckpt_labels[checkpoints.index(c)] for c in profile_ckpts]} "
          f"-> test={[ckpt_labels[checkpoints.index(c)] for c in test_ckpts]}")
    env_rows = []
    for pol, pct in policies.items():
        agg = {k: [] for k in ("ov_m", "fe_m", "wa_m", "ov_l", "fe_l", "wa_l")}
        for lab in layer_labels:
            hist = [mats[lab][c].astype(np.float64) for c in profile_ckpts]
            E = np.mean(np.stack(hist, 0), axis=0) if pol == "mean" else nm.build_envelope(hist, pct)
            E_link = E[mask]
            for c in test_ckpts:
                M = mats[lab][c].astype(np.float64)
                mm = nm.envelope_metrics(M, E)
                lm = nm.envelope_metrics(M[mask], E_link)
                agg["ov_m"].append(mm["overflow_ratio"]); agg["fe_m"].append(mm["frac_entries_overflow"]); agg["wa_m"].append(mm["waste_ratio"])
                agg["ov_l"].append(lm["overflow_ratio"]); agg["fe_l"].append(lm["frac_entries_overflow"]); agg["wa_l"].append(lm["waste_ratio"])
        env_rows.append({
            "policy": pol,
            "matrix_overflow_ratio": float(np.mean(agg["ov_m"])),
            "matrix_frac_flows_overflow": float(np.mean(agg["fe_m"])),
            "matrix_waste_ratio": float(np.mean(agg["wa_m"])),
            "link_overflow_ratio": float(np.mean(agg["ov_l"])),
            "link_frac_flows_overflow": float(np.mean(agg["fe_l"])),
            "link_waste_ratio": float(np.mean(agg["wa_l"])),
        })
    env_df = pd.DataFrame(env_rows)
    env_df.to_csv(RESULTS / f"{prefix}envelope_quality.csv", index=False)

    robust_rows = []
    variants = [("contiguous", "contiguous"), ("roundrobin", "contiguous"),
                ("random", "contiguous"), ("contiguous", "strided")]
    for plc, shard in variants:
        place_v = pl.expert_to_device(ne, D, plc, config.NET_RANDOM_SEED)
        first_cos, final_resid, final_fab_skew = [], [], []
        for layer, lab in zip(layers, layer_labels):
            m0 = build_matrix_for(get_indices(layer, checkpoints[0]), place_v, shard)
            mf = build_matrix_for(get_indices(layer, final), place_v, shard)
            first_cos.append(nm.cosine(tmx.as_distribution(m0), tmx.as_distribution(mf)))
            final_resid.append(rank1_residual(mf))
            final_fab_skew.append(nm.link_stats(fabric.link_load_array(mf))["max_over_mean"])
        robust_rows.append({
            "placement": plc, "sharding": shard,
            "first_ckpt_matrix_cosine_vs_final": float(np.mean(first_cos)),
            "final_rank1_residual": float(np.mean(final_resid)),
            "final_fabric_max_over_mean": float(np.mean(final_fab_skew)),
        })
    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(RESULTS / f"{prefix}placement_robustness.csv", index=False)

    # ---- FIGURES ------------------------------------------------------
    palette = plotting.PALETTE

    fig, ax = plt.subplots(figsize=(8, 5))
    series = {lab: mat_df[mat_df["layer"] == lab]["matrix_cosine_vs_final"].values for lab in layer_labels}
    sm = _per_layer_lineplot(ax, layer_labels, xpos, series)
    ax.set_xticks(xpos); ax.set_xticklabels(ckpt_labels, rotation=45 if len(ckpt_labels) > 6 else 0, fontsize=8)
    ax.set_xlabel("Training checkpoint"); ax.set_ylabel("Cosine of dispatch matrix vs final")
    ax.set_title(f"Matrix-level stabilization{title_suffix}")
    if sm is not None:
        fig.colorbar(sm, ax=ax, label="MoE layer")
    plotting.save(fig, RESULTS / f"{prefix}fig1_matrix_cosine_vs_final.png")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(xpos, mat_avg["matrix_cosine_vs_final"], "o-", color=palette[0], lw=2.2,
            label="matrix cosine vs final (Exp H)")
    ax.plot(xpos, link_avg["fabric_hot_overlap_vs_final"], "D-", color=palette[4], lw=2.0,
            label="link hot-overlap vs final (Exp H)")
    if overlay.get("volume") is not None:
        ax.plot(xpos, overlay["volume"], "s--", color=palette[2], lw=2.0, label="volume cosine vs final (Exp E)")
    if overlay.get("routing") is not None:
        ax.plot(xpos, overlay["routing"], "^:", color=palette[1], lw=2.0, label=overlay["routing_label"])
    ax.set_xticks(xpos); ax.set_xticklabels(ckpt_labels, rotation=45 if len(ckpt_labels) > 6 else 0, fontsize=8)
    ax.set_xlabel("Training checkpoint"); ax.set_ylabel("Agreement with final model (layer-avg)")
    ax.set_title(f"Predictability ladder: routing < volume \u2264 matrix{title_suffix}")
    ax.legend(fontsize=8)
    plotting.save(fig, RESULTS / f"{prefix}fig2_predictability_ladder.png")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(xpos, link_avg["fabric_max_over_mean"], "o-", color=palette[0], lw=2.2, label="full-mesh link max/mean")
    ax.plot(xpos, link_avg["uplink_max_over_mean"], "s--", color=palette[3], lw=2.0, label="2-tier uplink max/mean")
    ax.axhline(1.0, color="black", ls=":", lw=1, label="perfectly balanced")
    ax.set_xticks(xpos); ax.set_xticklabels(ckpt_labels, rotation=45 if len(ckpt_labels) > 6 else 0, fontsize=8)
    ax.set_xlabel("Training checkpoint"); ax.set_ylabel("Link-load skew (max / mean)")
    ax.set_title(f"Link-load skew across training{title_suffix}")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / f"{prefix}fig3_link_load_skew.png")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(xpos, link_avg["fabric_hot_overlap_vs_final"], "o-", color=palette[0], lw=2.2, label="full-mesh hottest-25% overlap")
    ax.plot(xpos, link_avg["uplink_hot_overlap_vs_final"], "s--", color=palette[3], lw=2.0, label="2-tier uplink hottest-50% overlap")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xpos); ax.set_xticklabels(ckpt_labels, rotation=45 if len(ckpt_labels) > 6 else 0, fontsize=8)
    ax.set_xlabel("Training checkpoint"); ax.set_ylabel("Hot-link overlap with final")
    ax.set_title(f"Do the same links stay hot?{title_suffix}")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / f"{prefix}fig4_hot_link_overlap.png")

    fig, ax = plt.subplots(figsize=(7.2, 5))
    for _, r in env_df.iterrows():
        ax.scatter(r["link_waste_ratio"] * 100, r["link_overflow_ratio"] * 100, s=90, zorder=3)
        ax.annotate(r["policy"], (r["link_waste_ratio"] * 100, r["link_overflow_ratio"] * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=10)
    ax.plot(env_df["link_waste_ratio"] * 100, env_df["link_overflow_ratio"] * 100,
            color=palette[8], lw=1.2, ls="--", zorder=2)
    ax.set_xlabel("Reservation waste (% of envelope unused = refundable)")
    ax.set_ylabel("Overflow (% of traffic exceeding envelope)")
    ax.set_title(f"Envelope trade-off on held-out checkpoints (link){title_suffix}")
    plotting.save(fig, RESULTS / f"{prefix}fig5_envelope_tradeoff.png")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    labels = [f"{r.placement}\n/{r.sharding}" for r in robust_df.itertuples()]
    x = np.arange(len(robust_df))
    ax.bar(x - 0.2, robust_df["first_ckpt_matrix_cosine_vs_final"], width=0.4, color=palette[0],
           label="matrix cosine (first ckpt vs final)")
    ax.bar(x + 0.2, robust_df["final_rank1_residual"], width=0.4, color=palette[1],
           label="rank-1 residual (structure beyond volume)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Metric value")
    ax.set_title(f"Robustness to placement & sharding{title_suffix}")
    ax.legend(fontsize=8)
    plotting.save(fig, RESULTS / f"{prefix}fig6_placement_robustness.png")

    # ---- HEADLINE -----------------------------------------------------
    first_mat_cos = float(mat_avg.iloc[0]["matrix_cosine_vs_final"])
    first_resid = float(mat_avg.iloc[0]["rank1_residual"])
    fab_skew_final = float(link_df[link_df["checkpoint"] == ckpt_labels[-1]]["fabric_max_over_mean"].mean())
    fab_hot_first = float(link_avg.iloc[0]["fabric_hot_overlap_vs_final"])
    p95 = env_df[env_df["policy"] == "P95"].iloc[0]
    worst = env_df[env_df["policy"] == "worst"].iloc[0]
    headline = {
        "model": label, "devices": D, "src_ranks": config.NET_SRC_RANKS,
        "placement": config.NET_PLACEMENT, "sharding": config.NET_SHARDING,
        "first_ckpt": ckpt_labels[0],
        "first_ckpt_matrix_cosine_vs_final": first_mat_cos,
        "first_ckpt_rank1_residual": first_resid,
        "first_ckpt_fabric_hot_overlap_vs_final": fab_hot_first,
        "final_fabric_max_over_mean": fab_skew_final,
        "P95_link_overflow_ratio": float(p95["link_overflow_ratio"]),
        "P95_link_waste_ratio": float(p95["link_waste_ratio"]),
        "worstcase_link_waste_ratio": float(worst["link_waste_ratio"]),
    }
    pd.DataFrame([headline]).to_csv(RESULTS / f"{prefix}headline.csv", index=False)
    print(f"\n=== Experiment H headline ({label}) ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    print(f"HEADLINE [{label}]: matrix cosine {first_mat_cos:.4f} at first ckpt; "
          f"link max/mean {fab_skew_final:.2f}x; rank-1 residual {first_resid:.3f} "
          f"(structure beyond volume); P95 envelope {100*float(p95['link_overflow_ratio']):.1f}% "
          f"link overflow at {100*float(p95['link_waste_ratio']):.1f}% waste "
          f"(worst-case {100*float(worst['link_waste_ratio']):.1f}% waste).")
    return mat_avg, link_avg, env_df


def run_flame():
    layers = config.FLAME_LAYERS
    checkpoints = config.FLAME_CHECKPOINTS
    n = config.SAMPLE_STABILIZATION

    def get_indices(layer, ckpt):
        return load_layer(ckpt, layer, n)[1]

    overlay = {"volume": None, "routing": None, "routing_label": "routing top-6 overlap (Exp B)"}
    expe = ROOT / "exp_E_volume_stabilization" / "results" / "volume_stability_avg.csv"
    if expe.exists():
        e = pd.read_csv(expe).set_index("checkpoint")
        overlay["volume"] = [float(e.loc[c, "cosine_vs_final"]) for c in checkpoints]
    expb = ROOT / "exp_B_routing_stabilization" / "results" / "stabilization_layer_avg.csv"
    if expb.exists():
        b = pd.read_csv(expb).set_index("checkpoint")
        if "top6_overlap" in b.columns:
            overlay["routing"] = [float(b.loc[c, "top6_overlap"]) for c in checkpoints]

    return run_model(
        label="flame-moe-290m", prefix="", get_indices=get_indices,
        layers=layers, layer_labels=list(layers),
        checkpoints=checkpoints, ckpt_labels=[str(c) for c in checkpoints],
        final=config.FLAME_FINAL_CHECKPOINT, ne=config.FLAME_NUM_EXPERTS,
        profile_slice=(1, 6), overlay=overlay, title_suffix="  (FLAME-MoE-290M)",
    )


def run_olmoe():
    ne = config.OLMOE_NUM_EXPERTS
    nl = config.OLMOE_NUM_LAYERS
    n = config.SAMPLE_OLMOE

    print("\n[H] OLMoE: loading cached checkpoints (n=%d)..." % n)
    data = {c: olmoe_load(c, n) for c in OLMOE_CHECKPOINTS}   # (n_tokens, 16, 8)
    m = min(arr.shape[0] for arr in data.values())            # token alignment

    layers = list(range(nl))
    layer_labels = [f"layer_{i:02d}" for i in range(nl)]

    def get_indices(layer, ckpt):
        return data[ckpt][:m, layer, :]

    overlay = {"volume": None, "routing": None, "routing_label": "routing top-8 overlap (Exp E)"}
    ole = ROOT / "exp_E_volume_stabilization" / "results" / "olmoe_volume_stability.csv"
    if ole.exists():
        o = pd.read_csv(ole).set_index("checkpoint")
        idx = [int(c) if c != "final" else "final" for c in o.index]  # keys may be int-parsed
        o.index = [str(c) for c in idx]
        if all(c in o.index for c in OLMOE_CHECKPOINTS):
            overlay["volume"] = [float(o.loc[c, "volume_cosine_vs_final"]) for c in OLMOE_CHECKPOINTS]
            overlay["routing"] = [float(o.loc[c, "routing_top8_overlap_vs_final"]) for c in OLMOE_CHECKPOINTS]

    return run_model(
        label="olmoe-1b-7b", prefix="olmoe_", get_indices=get_indices,
        layers=layers, layer_labels=layer_labels,
        checkpoints=list(OLMOE_CHECKPOINTS), ckpt_labels=list(OLMOE_CHECKPOINTS),
        final="final", ne=ne, profile_slice=(0, 3), overlay=overlay,
        title_suffix="  (OLMoE-1B-7B)",
    )


def main():
    plotting.apply_style()
    run_flame()
    run_olmoe()


if __name__ == "__main__":
    main()
