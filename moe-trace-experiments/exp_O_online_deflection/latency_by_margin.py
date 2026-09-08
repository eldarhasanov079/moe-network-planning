"""Exp O — TOTAL latency vs baseline at FIXED margin thresholds (whole-model, no per-layer split)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as O  # noqa: E402
from common import plotting  # noqa: E402
from common import placement as pl  # noqa: E402
import config  # noqa: E402

TAUS = [0.001, 0.002, 0.004]
RESULTS = O.RESULTS
palette = plotting.PALETTE
C_RAW, C_LOCAL, C_GLOBAL, C_OFF = "0.55", palette[2], palette[1], palette[0]

HIDDEN_SIZE = 1024
PEAK_GFLOPS = 900.0
BW_GB_S = 100.0
ATTN_GATE_FLOPS_PER_TOKEN = 10 * HIDDEN_SIZE * HIDDEN_SIZE
EXPERT_FLOPS_PER_SLOT = 16 * HIDDEN_SIZE * HIDDEN_SIZE


def tk(tau):
    return f"t{int(round(tau * 10000)):04d}"          # 0.002 -> t0020


def _flops_to_cycles(flops, comm_ref_cycles, comm_ref_bytes):
    comm_s = comm_ref_bytes / (BW_GB_S * 1e9)
    compute_s = flops / (PEAK_GFLOPS * 1e9)
    if comm_s <= 0:
        return 0
    return max(1, int(comm_ref_cycles * (compute_s / comm_s)))


def e2e_wall(M, disp, comb, bytes_per_slot):
    """max-rank(attn+gating + expert) + dispatch + combine (Exp M model)."""
    total_bytes = float(M.sum() * bytes_per_slot)
    best = 0
    for r in range(M.shape[0]):
        local_tok = float(M[r, :].sum())
        recv_slots = float(M[:, r].sum())
        attn = _flops_to_cycles(local_tok * ATTN_GATE_FLOPS_PER_TOKEN, disp, total_bytes)
        expert = _flops_to_cycles(recv_slots * EXPERT_FLOPS_PER_SLOT, disp, total_bytes)
        best = max(best, attn + expert)
    return best + disp + comb


def base_meta(base):
    if base == "mraw":
        return ("raw", None)
    if base == "moff":
        return ("offline", None)
    pol = "online_global" if base.startswith("og_") else "online_local"
    tau = int(base.split("_t")[1]) / 10000.0
    return (pol, tau)


def build_and_run():
    D = config.NET_DEVICES; S = config.NET_SRC_RANKS
    place = pl.expert_to_device(config.FLAME_NUM_EXPERTS, D, config.NET_PLACEMENT, config.NET_RANDOM_SEED)
    n = config.SAMPLE_STABILIZATION
    ckpt = config.FLAME_FINAL_CHECKPOINT
    layers = config.FLAME_LAYERS
    order = np.arange(n)

    (O.WORK / "net").mkdir(parents=True, exist_ok=True)
    for topo, fname in O.TOPOLOGIES.items():
        (O.WORK / "net" / fname).write_text(
            f"topology: [ {topo} ]\nnpus_count: [ {D} ]\nbandwidth: [ 100.0 ]\nlatency: [ 500.0 ]\n")

    jobs, matrices, peaks = [], {}, []
    print(f"[lat-margin] building dispatch+combine matrices at fixed tau {TAUS} for {len(layers)} layers")
    for layer in layers:
        top1, top2, margin, src = O.layer_arrays(ckpt, layer, n, D, S, place)
        tag = layer.replace("layer_", "L")
        specs = [("mraw", top1), ("moff", O.offline_full_chosen(top1, top2, margin, D))]
        for tau in TAUS:
            g = O.online_margin(top1, top2, margin, src, D, S, tau, order, local=False)
            l = O.online_margin(top1, top2, margin, src, D, S, tau, order, local=True)
            specs += [(f"og_{tk(tau)}", g["chosen"]), (f"ol_{tk(tau)}", l["chosen"])]
            peaks.append({"layer": layer, "policy": "online_global", "tau": tau, "peak": g["peak"], "frac": g["frac"]})
            peaks.append({"layer": layer, "policy": "online_local", "tau": tau, "peak": l["peak"], "frac": l["frac"]})
        for base, chosen in specs:
            M = O.matrix_from_chosen(chosen, src, D, S)
            matrices[(layer, base)] = M
            dname, cname = f"{tag}_{base}_d", f"{tag}_{base}_c"
            np.save(O.WORK / "npy" / f"{dname}.npy", M)
            np.save(O.WORK / "npy" / f"{cname}.npy", M.T.copy())
            jobs.append({"name": dname, "matrix": str(O.WORK / "npy" / f"{dname}.npy")})
            jobs.append({"name": cname, "matrix": str(O.WORK / "npy" / f"{cname}.npy")})

    manifest = {"out_dir": str(O.WORK), "bytes_per_slot": O.BYTES_PER_SLOT, "mode": "comm_only", "jobs": jobs}
    (O.WORK / "manifest_margin.json").write_text(json.dumps(manifest))
    print(f"[lat-margin] generating Chakra ET ({len(jobs)} matrices) ...")
    g = subprocess.run([str(O.CHAKRA_PY), str(O.GEN_ET), "--manifest", str(O.WORK / "manifest_margin.json")],
                       capture_output=True, text=True)
    if g.returncode != 0:
        raise RuntimeError("gen_et failed:\n" + g.stdout[-800:] + g.stderr[-800:])

    print("[lat-margin] running ASTRA-sim across 3 topologies (dispatch + combine) ...")
    rows = []
    bases = ["mraw", "moff"] + [f"og_{tk(t)}" for t in TAUS] + [f"ol_{tk(t)}" for t in TAUS]
    for topo in O.TOPOLOGIES:
        for layer in layers:
            tag = layer.replace("layer_", "L")
            for base in bases:
                disp = O.run_astrasim(f"{tag}_{base}_d", topo)
                comb = O.run_astrasim(f"{tag}_{base}_c", topo)
                pol, tau = base_meta(base)
                rows.append({"topology": topo, "layer": layer, "policy": pol, "tau": tau,
                             "a2a": disp + comb,
                             "wall": e2e_wall(matrices[(layer, base)], disp, comb, O.BYTES_PER_SLOT)})
        print(f"  {topo} done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "latency_by_margin_raw.csv", index=False)
    pd.DataFrame(peaks).to_csv(RESULTS / "peak_by_margin.csv", index=False)
    return df


def totals(df, metric, topo, policy, tau):
    """Sum the metric over all layers for one (topo, policy[, tau])."""
    m = (df["topology"] == topo) & (df["policy"] == policy)
    if tau is not None:
        m &= (df["tau"] == tau)
    return float(df.loc[m, metric].sum())


def make_fig(df, metric, tau, fname, ylabel, title):
    topos = list(O.TOPOLOGIES)
    xt = np.arange(len(topos)); w = 0.26
    series = [("raw", None, C_RAW, "raw (baseline)"),
              ("online_local", tau, C_LOCAL, "online local"),
              ("online_global", tau, C_GLOBAL, "online global")]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    raw_tot = {t: totals(df, metric, t, "raw", None) for t in topos}
    for k, (pol, tt, c, lab) in enumerate(series):
        vals = [totals(df, metric, t, pol, tt) / 1e6 for t in topos]
        ax.bar(xt + (k - 1) * w, vals, w, color=c, label=lab)
        if pol != "raw":
            for j, t in enumerate(topos):
                red = 100 * (1 - (vals[j] * 1e6) / raw_tot[t])
                ax.text(xt[j] + (k - 1) * w, vals[j], f"{red:+.1f}%",
                        ha="center", va="bottom", fontsize=8)
    for j, t in enumerate(topos):
        off_v = totals(df, metric, t, "offline", None) / 1e6
        off_red = 100 * (1 - (off_v * 1e6) / raw_tot[t])
        ax.hlines(off_v, xt[j] - 1.5 * w, xt[j] + 1.5 * w, color="k", ls="--", lw=1.6,
                  label="offline lower bound (full balance)" if j == 0 else None)
        ax.text(xt[j] + 1.5 * w, off_v, f" {off_red:+.1f}%", ha="left", va="center",
                fontsize=7.5, color="k")
    ax.set_xticks(xt); ax.set_xticklabels(topos)
    ax.set_ylabel(ylabel); ax.set_xlabel("network topology (100 GB/s, 500 ns links)")
    ax.set_title(title)
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False, columnspacing=1.2, handletextpad=0.5)
    ax.margins(y=0.16)
    plotting.save(fig, RESULTS / fname)


def main():
    plotting.apply_style()
    if "--figonly" in sys.argv and (RESULTS / "latency_by_margin_raw.csv").exists():
        print("[lat-margin] --figonly: reloading cached latency_by_margin_raw.csv (no ASTRA-sim)")
        df = pd.read_csv(RESULTS / "latency_by_margin_raw.csv")
    else:
        df = build_and_run()
    fp = pd.read_csv(RESULTS / "peak_by_margin.csv")
    for tau in TAUS:
        frac = fp[(fp["policy"] == "online_global") & (fp["tau"] == tau)]["frac"].mean()
        make_fig(df, "a2a", tau, f"fig_latency_tau_{tk(tau)}.png",
                 "total All-to-All time over 8 layers (million cycles)",
                 f"Total All-to-All latency vs baseline, fixed M=τ={tau:g} "
                 f"(16 collectives; online moves ~{frac*100:.1f}% of tokens)")
        make_fig(df, "wall", tau, f"fig_fulllatency_tau_{tk(tau)}.png",
                 "total workload latency over 8 layers (million cycles)",
                 f"Full workload latency (compute + both A2As) vs baseline, fixed M=τ={tau:g}")
    print("\n=== whole-model totals (million cycles) — Switch ===")
    for tau in TAUS:
        rr = totals(df, "a2a", "Switch", "raw", None) / 1e6
        og = totals(df, "a2a", "Switch", "online_global", tau) / 1e6
        wr = totals(df, "wall", "Switch", "raw", None) / 1e6
        wg = totals(df, "wall", "Switch", "online_global", tau) / 1e6
        print(f"tau={tau:g}: A2A raw {rr:.2f} / og {og:.2f} ({100*(1-og/rr):+.1f}%) | "
              f"FULL raw {wr:.2f} / og {wg:.2f} ({100*(1-wg/wr):+.1f}%)")
    print("wrote fig_latency_tau_* (total A2A) and fig_fulllatency_tau_* (full workload)")


if __name__ == "__main__":
    main()
