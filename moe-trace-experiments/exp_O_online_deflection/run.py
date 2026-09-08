"""Experiment O — Does *online* (causal, deployable) margin deflection work?"""
from __future__ import annotations

import heapq
import json
import re
import subprocess
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
from common import placement as pl  # noqa: E402
from common.flame_loader import load_layer  # noqa: E402

RESULTS = HERE / "results"
WORK = HERE / "astrasim_work"
GEN_ET = ROOT / "exp_M_astrasim_validation" / "gen_et.py"
RESULTS.mkdir(exist_ok=True)
(WORK / "npy").mkdir(parents=True, exist_ok=True)

MSC = Path.home() / "Desktop/msc"
CHAKRA_PY = MSC / "iso_code/chakra_env/bin/python"
DOCKER_IMAGE = "astra-sim:latest"
BIN = "/workspace/iso_code/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
EX = "/workspace/iso_code/astra-sim/examples"
SYSTEM = f"{EX}/system/native_collectives/Ring_4chunks.json"
REMOTE = f"{EX}/remote_memory/analytical/no_memory_expansion.json"
BYTES_PER_SLOT = 2048

TOPOLOGIES = {"Switch": "network_switch8.yml",
              "Ring": "network_ring8.yml",
              "FullyConnected": "network_fc8.yml"}
PRIMARY_TOPO = "Switch"


def net_container(fname):
    return f"/workspace/{(WORK / 'net' / fname).relative_to(MSC)}"

RELIEF_TARGET = 1.05
TAUS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 1.01]   # margin gates (1.01 = always eligible)
RANDOM_PS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
SHUFFLE_SEEDS = [0, 1, 2]


def layer_arrays(ckpt, layer, n, D, S, place):
    scores, indices = load_layer(ckpt, layer, n, verbose=False)
    top1 = place[indices[:, 0].astype(np.int64)]
    top2 = place[indices[:, 1].astype(np.int64)]
    margin = (scores[:, 0].astype(np.float64) - scores[:, 1].astype(np.float64))
    src = pl.token_source_ranks(len(indices), S, config.NET_SHARDING)
    return top1, top2, margin, src


def peak_of(chosen, D):
    L = np.bincount(chosen, minlength=D).astype(np.float64)
    return L.max() / (len(chosen) / D)


def offline_pareto(top1, top2, margin, D):
    n = len(top1)
    L = np.bincount(top1, minlength=D).astype(np.float64)
    mean = n / D
    pools = [[] for _ in range(D)]
    for t in range(n):
        pools[int(top1[t])].append((float(margin[t]), int(top2[t])))
    for d in range(D):
        heapq.heapify(pools[d])
    peaks = [L.max() / mean]; costs = [0.0]; fracs = [0.0]
    moved = 0; cost = 0.0
    while L.max() / mean > RELIEF_TARGET:
        d = int(np.argmax(L)); did = False
        while pools[d]:
            m, dst = heapq.heappop(pools[d])
            if dst == d:
                continue
            if L[dst] < L[d] - 1:
                L[d] -= 1; L[dst] += 1; moved += 1; cost += m
                peaks.append(L.max() / mean); costs.append(cost); fracs.append(moved / n)
                did = True
                break
        if not did:
            break
    return {"peak": np.array(peaks), "cost": np.array(costs), "frac": np.array(fracs)}


def online_margin(top1, top2, margin, src, D, S, tau, order, local):
    t1 = top1.tolist(); t2 = top2.tolist(); mg = margin.tolist(); sr = src.tolist()
    chosen = [0] * len(order)
    moved = 0; cost = 0.0
    if local:
        L = [[0.0] * D for _ in range(S)]
        for t in order:
            d0 = t1[t]; d1 = t2[t]; Lr = L[sr[t]]
            if d1 != d0 and Lr[d1] < Lr[d0] and mg[t] <= tau:
                d = d1; moved += 1; cost += mg[t]
            else:
                d = d0
            chosen[t] = d; Lr[d] += 1.0
    else:
        L = [0.0] * D
        for t in order:
            d0 = t1[t]; d1 = t2[t]
            if d1 != d0 and L[d1] < L[d0] and mg[t] <= tau:
                d = d1; moved += 1; cost += mg[t]
            else:
                d = d0
            chosen[t] = d; L[d] += 1.0
    chosen = np.asarray(chosen, dtype=np.int64)
    return {"peak": peak_of(chosen, D), "frac": moved / len(order), "cost": cost, "chosen": chosen}


def online_random(top1, top2, margin, src, D, p, order, rng):
    t1 = top1.tolist(); t2 = top2.tolist(); mg = margin.tolist()
    draws = rng.random(len(order))
    chosen = [0] * len(order)
    moved = 0; cost = 0.0
    L = [0.0] * D
    for i, t in enumerate(order):
        d0 = t1[t]; d1 = t2[t]
        if d1 != d0 and L[d1] < L[d0] and draws[i] < p:
            d = d1; moved += 1; cost += mg[t]
        else:
            d = d0
        chosen[t] = d; L[d] += 1.0
    chosen = np.asarray(chosen, dtype=np.int64)
    return {"peak": peak_of(chosen, D), "frac": moved / len(order), "cost": cost, "chosen": chosen}


def online_pareto_margin(top1, top2, margin, src, D, S, order, local):
    pts = [online_margin(top1, top2, margin, src, D, S, tau, order, local) for tau in TAUS]
    return {"peak": np.array([p["peak"] for p in pts]),
            "cost": np.array([p["cost"] for p in pts]),
            "frac": np.array([p["frac"] for p in pts]),
            "tau": np.array(TAUS)}


def online_pareto_random(top1, top2, margin, src, D, order, rng):
    pts = [online_random(top1, top2, margin, src, D, p, order, rng) for p in RANDOM_PS]
    return {"peak": np.array([q["peak"] for q in pts]),
            "cost": np.array([q["cost"] for q in pts]),
            "frac": np.array([q["frac"] for q in pts])}


def offline_cost_at_target(par):
    """Cost & peak at the relief target (last traced point)."""
    return float(par["cost"][-1]), float(par["peak"][-1])


def relief_tau(par):
    """Smallest margin gate at which online reaches the SAME balance target offline is run to"""
    mask = par["peak"] <= RELIEF_TARGET + 1e-9
    if mask.any():
        return float(par["tau"][np.where(mask)[0][0]])
    return float(par["tau"][int(np.argmin(par["peak"]))])


def best_online_at_budget(par, cost_budget, raw_peak, off_peak):
    """Two honest views of the online Pareto vs offline:"""
    denom = (raw_peak - off_peak)
    mask = par["cost"] <= cost_budget + 1e-9
    if not mask.any():
        i = int(np.argmin(par["cost"]))
    else:
        i = int(np.where(mask)[0][np.argmin(par["peak"][mask])])
    peak_mc = float(par["peak"][i])
    captured = (raw_peak - peak_mc) / denom if denom > 1e-9 else np.nan
    j = int(np.argmin(par["peak"]))
    minpeak = float(par["peak"][j]); minpeak_cost = float(par["cost"][j])
    rmask = par["peak"] <= RELIEF_TARGET + 1e-9
    reaches = bool(rmask.any())
    cost_to_relief = float(par["cost"][np.where(rmask)[0][0]]) if reaches else np.nan
    ratio = (cost_to_relief / cost_budget) if (reaches and cost_budget > 1e-9) else np.nan
    return {"peak": peak_mc, "cost": float(par["cost"][i]), "frac": float(par["frac"][i]),
            "tau": float(par["tau"][i]), "captured": captured,
            "minpeak": minpeak, "minpeak_cost": minpeak_cost,
            "reaches_relief": reaches, "cost_to_relief_ratio": ratio}


def matrix_from_chosen(chosen, src, D, S):
    M = np.zeros((S, D), dtype=np.float64)
    np.add.at(M, (src, chosen), 1.0)
    return M


def run_astrasim(name, topo=PRIMARY_TOPO):
    workload = f"/workspace/{(WORK / 'et' / name).relative_to(MSC)}"
    cmd = ["docker", "run", "--rm", "-v", f"{MSC}:/workspace", "-w", "/workspace",
           DOCKER_IMAGE, BIN,
           f"--workload-configuration={workload}",
           f"--system-configuration={SYSTEM}",
           f"--network-configuration={net_container(TOPOLOGIES[topo])}",
           f"--remote-memory-configuration={REMOTE}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    times = [int(m) for m in re.findall(r"Comm time: (\d+)", text)]
    if not times:
        raise RuntimeError(f"no Comm time for {name} on {topo}:\n{text[-1200:]}")
    return max(times)


def main():
    plotting.apply_style()
    import matplotlib.pyplot as plt
    palette = plotting.PALETTE

    D = config.NET_DEVICES; S = config.NET_SRC_RANKS
    place = pl.expert_to_device(config.FLAME_NUM_EXPERTS, D, config.NET_PLACEMENT, config.NET_RANDOM_SEED)
    n = config.SAMPLE_STABILIZATION
    ckpt = config.FLAME_FINAL_CHECKPOINT
    layers = config.FLAME_LAYERS
    (WORK / "net").mkdir(parents=True, exist_ok=True)
    for topo, fname in TOPOLOGIES.items():
        (WORK / "net" / fname).write_text(
            f"topology: [ {topo} ]\nnpus_count: [ {D} ]\nbandwidth: [ 100.0 ]\nlatency: [ 500.0 ]\n")

    per_layer = []
    pareto_store = {}
    astrasim_jobs = []
    astrasim_matrices = {}

    print(f"[O] {len(layers)} layers, ckpt {ckpt}, {n:,} tokens/layer, EP={D}")
    for layer in layers:
        top1, top2, margin, src = layer_arrays(ckpt, layer, n, D, S, place)
        order = np.arange(n)                       # arrival order = trace row order

        raw_peak = peak_of(top1, D)
        off = offline_pareto(top1, top2, margin, D)
        off_cost, off_peak = offline_cost_at_target(off)

        pg = online_pareto_margin(top1, top2, margin, src, D, S, order, local=False)
        plc = online_pareto_margin(top1, top2, margin, src, D, S, order, local=True)
        rng = np.random.default_rng(0)
        prnd = online_pareto_random(top1, top2, margin, src, D, order, rng)

        bg = best_online_at_budget(pg, off_cost, raw_peak, off_peak)
        blc = best_online_at_budget(plc, off_cost, raw_peak, off_peak)

        pareto_store[layer] = {"offline": off, "online_global": pg, "online_local": plc,
                               "online_random": prnd, "raw_peak": raw_peak,
                               "off_cost": off_cost, "off_peak": off_peak}

        raw_chosen = top1
        off_chosen = offline_full_chosen(top1, top2, margin, D)
        g_res = online_margin(top1, top2, margin, src, D, S, relief_tau(pg), order, local=False)
        l_res = online_margin(top1, top2, margin, src, D, S, relief_tau(plc), order, local=True)
        tag = layer.replace("layer_", "L")
        for pol, chosen in [("raw", raw_chosen), ("offline", off_chosen),
                            ("online_global", g_res["chosen"]), ("online_local", l_res["chosen"])]:
            M = matrix_from_chosen(chosen, src, D, S)
            name = f"{tag}_{pol}"
            np.save(WORK / "npy" / f"{name}.npy", M)
            astrasim_jobs.append({"name": name, "matrix": str(WORK / "npy" / f"{name}.npy")})
            astrasim_matrices[name] = M

        per_layer.append({
            "layer": layer, "raw_peak": raw_peak, "offline_peak": off_peak, "offline_cost": off_cost,
            "online_global_peak_matchedcost": bg["peak"], "online_global_captured": bg["captured"],
            "online_global_minpeak": bg["minpeak"], "online_global_minpeak_cost": bg["minpeak_cost"],
            "online_global_reaches_relief": bg["reaches_relief"],
            "online_global_cost_to_relief_ratio": bg["cost_to_relief_ratio"],
            "online_local_peak_matchedcost": blc["peak"], "online_local_captured": blc["captured"],
            "online_local_minpeak": blc["minpeak"], "online_local_reaches_relief": blc["reaches_relief"],
        })
        print(f"  {layer}: raw {raw_peak:.3f} | offline {off_peak:.3f}@{off_cost:.1f} | "
              f"online_g matched {bg['peak']:.3f} (cap {bg['captured']*100:.0f}%), "
              f"minpeak {bg['minpeak']:.3f}@{bg['minpeak_cost']:.1f}, "
              f"relief {'Y' if bg['reaches_relief'] else 'N'} (x{bg['cost_to_relief_ratio']:.1f} cost)")

    df = pd.DataFrame(per_layer)
    df.to_csv(RESULTS / "online_vs_offline_by_layer.csv", index=False)

    # ---- ASTRA-sim confirmation ----
    manifest = {"out_dir": str(WORK), "bytes_per_slot": BYTES_PER_SLOT, "mode": "comm_only", "jobs": astrasim_jobs}
    (WORK / "manifest.json").write_text(json.dumps(manifest))
    print("[O] generating Chakra ET ...")
    g = subprocess.run([str(CHAKRA_PY), str(GEN_ET), "--manifest", str(WORK / "manifest.json")],
                       capture_output=True, text=True)
    print(g.stdout[-400:] + g.stderr[-400:])
    if g.returncode != 0:
        raise RuntimeError("gen_et failed")

    print("[O] running ASTRA-sim across 3 topologies for raw/offline/online per layer ...")
    pols = ("raw", "offline", "online_global", "online_local")
    comm_rows = []
    for topo in TOPOLOGIES:
        for layer in layers:
            tag = layer.replace("layer_", "L")
            c = {pol: run_astrasim(f"{tag}_{pol}", topo) for pol in pols}
            comm_rows.append({"topology": topo, "layer": layer,
                              **{f"{k}_comm": v for k, v in c.items()},
                              "online_global_reduction_pct": 100 * (1 - c["online_global"] / c["raw"]),
                              "offline_reduction_pct": 100 * (1 - c["offline"] / c["raw"]),
                              "online_local_reduction_pct": 100 * (1 - c["online_local"] / c["raw"])})
        sub = [r for r in comm_rows if r["topology"] == topo]
        mg = np.mean([r["online_global_reduction_pct"] for r in sub])
        ml = np.mean([r["online_local_reduction_pct"] for r in sub])
        mo = np.mean([r["offline_reduction_pct"] for r in sub])
        print(f"  [{topo}] mean latency reduction vs raw:  offline {mo:.1f}%  online_g {mg:.1f}%  online_l {ml:.1f}%")
    comm = pd.DataFrame(comm_rows)
    comm.to_csv(RESULTS / "online_vs_offline_commtime.csv", index=False)
    comm_primary = comm[comm["topology"] == PRIMARY_TOPO].reset_index(drop=True)

    showcase = df.loc[df["raw_peak"].idxmax(), "layer"]
    top1, top2, margin, src = layer_arrays(ckpt, showcase, n, D, S, place)
    tau_star = 0.05
    sens = []
    for seed in SHUFFLE_SEEDS:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        rg = online_margin(top1, top2, margin, src, D, S, tau_star, order, local=False)
        rl = online_margin(top1, top2, margin, src, D, S, tau_star, order, local=True)
        sens.append({"seed": seed, "order": "shuffle", "global_peak": rg["peak"], "local_peak": rl["peak"]})
    natord = np.arange(n)
    rg = online_margin(top1, top2, margin, src, D, S, tau_star, natord, local=False)
    rl = online_margin(top1, top2, margin, src, D, S, tau_star, natord, local=True)
    sens.append({"seed": -1, "order": "trace", "global_peak": rg["peak"], "local_peak": rl["peak"]})
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(RESULTS / "order_sensitivity.csv", index=False)

    # ================= figures =================
    ps = pareto_store[showcase]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(ps["offline"]["cost"], ps["offline"]["peak"], "-", color=palette[0], lw=2.4, label="offline (hindsight upper bound)")
    ax.plot(ps["online_global"]["cost"], ps["online_global"]["peak"], "o-", color=palette[1], lw=2, label="online global (deployable)")
    ax.plot(ps["online_local"]["cost"], ps["online_local"]["peak"], "s--", color=palette[2], lw=2, label="online local (no coordination)")
    ax.plot(ps["online_random"]["cost"], ps["online_random"]["peak"], "d:", color=palette[3], lw=2, label="online random (margin-blind)")
    ax.axhline(ps["raw_peak"], color="gray", ls="-", lw=1.2, label=f"raw peak {ps['raw_peak']:.2f}")
    ax.axhline(RELIEF_TARGET, color="k", ls="-.", lw=1, label=f"relief target {RELIEF_TARGET}")
    ax.set_xlabel("Router-score cost (sum of deflected margins)")
    ax.set_ylabel("Peak device load (max/mean)")
    ax.set_title(f"Online vs offline deflection ({showcase}, FLAME-290M, real traces)")
    ax.legend(fontsize=8)
    plotting.save(fig, RESULTS / "fig1_pareto_showcase.png")

    x = np.arange(len(df)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.bar(x - w/2, df["online_global_captured"] * 100, w, color=palette[1], label="online global")
    ax.bar(x + w/2, df["online_local_captured"] * 100, w, color=palette[2], label="online local")
    ax.axhline(100, color=palette[0], ls="-", lw=1.5, label="offline (=100%)")
    ax.set_xticks(x); ax.set_xticklabels([r.replace("layer_", "L") for r in df["layer"]])
    ax.set_ylabel("% of offline peak-reduction benefit captured")
    ax.set_xlabel("MoE layer")
    ax.set_title("How much of the hindsight benefit does causal online deflection keep? (cost ≤ offline)")
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / "fig2_benefit_by_layer.png")

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.bar(x - w/2, comm_primary["online_global_reduction_pct"], w, color=palette[1], label="online global (relief)")
    ax.bar(x + w/2, comm_primary["offline_reduction_pct"], w, color=palette[0], label="offline (full balance)")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([r.replace("layer_", "L") for r in comm_primary["layer"]])
    ax.set_ylabel("ASTRA-sim comm-time reduction vs raw (%)")
    ax.set_xlabel("MoE layer")
    ax.set_title(f"External-simulator comm-time reduction ({PRIMARY_TOPO}) — small/±noisy: fluid\n"
                 "all-to-all time tracks aggregate per-link volume, not peak device load (non-monotonic)",
                 fontsize=10)
    ax.legend(fontsize=9)
    plotting.save(fig, RESULTS / "fig3_astrasim_reduction.png")

    # ================= verdict =================
    gcap = df["online_global_captured"].mean()
    lcap = df["online_local_captured"].mean()
    g_comm = comm_primary["online_global_reduction_pct"].mean()
    off_comm = comm_primary["offline_reduction_pct"].mean()
    per_topo = {t: {"offline": comm.loc[comm["topology"] == t, "offline_reduction_pct"].mean(),
                    "online_global": comm.loc[comm["topology"] == t, "online_global_reduction_pct"].mean(),
                    "online_local": comm.loc[comm["topology"] == t, "online_local_reduction_pct"].mean()}
                for t in TOPOLOGIES}
    online_helps = (df["online_global_minpeak"] < df["raw_peak"] - 1e-6).all()
    reaches = df["online_global_reaches_relief"].mean()
    sens_spread = sens_df[["global_peak", "local_peak"]].std().max()

    summary = {
        "layers": len(layers), "checkpoint": ckpt, "tokens_per_layer": n,
        "mean_raw_peak": df["raw_peak"].mean(),
        "mean_offline_peak": df["offline_peak"].mean(),
        "mean_online_global_minpeak": df["online_global_minpeak"].mean(),
        "mean_online_global_peak_matchedcost": df["online_global_peak_matchedcost"].mean(),
        "mean_benefit_captured_global_pct": 100 * gcap,
        "mean_benefit_captured_local_pct": 100 * lcap,
        "frac_layers_online_reaches_relief": reaches,
        "mean_commtime_reduction_online_global_pct_primary": g_comm,
        "mean_commtime_reduction_offline_pct_primary": off_comm,
        "online_reduces_peak_all_layers": bool(online_helps),
        "order_sensitivity_peak_std": float(sens_spread),
        **{f"latred_{t}_online_global_pct": per_topo[t]["online_global"] for t in TOPOLOGIES},
        **{f"latred_{t}_online_local_pct": per_topo[t]["online_local"] for t in TOPOLOGIES},
        **{f"latred_{t}_offline_pct": per_topo[t]["offline"] for t in TOPOLOGIES},
    }
    pd.DataFrame([summary]).to_csv(RESULTS / "verdict.csv", index=False)

    print("\n=== Experiment O verdict ===")
    print(df.to_string(index=False))
    print("\n" + comm.to_string(index=False))
    print(f"\nOnline GLOBAL captures {100*gcap:.0f}% of the offline peak-reduction benefit at <= offline cost;")
    print(f"online LOCAL (zero coordination) captures {100*lcap:.0f}%.")
    print(f"ASTRA-sim comm-time reduction ({PRIMARY_TOPO}): online_global {g_comm:.1f}% vs offline {off_comm:.1f}% (vs raw).")
    print("Per-topology mean latency reduction vs raw baseline:")
    for t in TOPOLOGIES:
        print(f"  {t:>15}: offline {per_topo[t]['offline']:5.1f}% | "
              f"online_global {per_topo[t]['online_global']:5.1f}% | online_local {per_topo[t]['online_local']:5.1f}%")
    print(f"Online reduces peak on ALL layers: {online_helps}. Order-sensitivity peak std: {sens_spread:.4f}.")
    verdict = ("ONLINE HELPS" if online_helps and gcap > 0.5 else
               "ONLINE HELPS PARTIALLY" if online_helps else "ONLINE DOES NOT HELP")
    print(f"VERDICT: {verdict}.")


def offline_chosen(top1, top2, margin, D):
    """Replay the offline greedy but return a per-token chosen-device array (for ASTRA-sim)."""
    n = len(top1)
    L = np.bincount(top1, minlength=D).astype(np.float64)
    mean = n / D
    chosen = top1.copy()
    pools = [[] for _ in range(D)]
    for t in range(n):
        pools[int(top1[t])].append((float(margin[t]), t))
    for d in range(D):
        heapq.heapify(pools[d])
    while L.max() / mean > RELIEF_TARGET:
        d = int(np.argmax(L)); did = False
        while pools[d]:
            m, t = heapq.heappop(pools[d])
            dst = int(top2[t])
            if dst == d:
                continue
            if L[dst] < L[d] - 1:
                L[d] -= 1; L[dst] += 1; chosen[t] = dst; did = True
                break
        if not did:
            break
    return chosen


def offline_full_chosen(top1, top2, margin, D):
    """Offline greedy traced to FULL balance (peak->1.0, until no move reduces the peak)."""
    n = len(top1)
    L = np.bincount(top1, minlength=D).astype(np.float64)
    mean = n / D
    chosen = top1.copy()
    pools = [[] for _ in range(D)]
    for t in range(n):
        pools[int(top1[t])].append((float(margin[t]), t))
    for d in range(D):
        heapq.heapify(pools[d])
    while L.max() / mean > 1.0 + 1e-9:
        d = int(np.argmax(L)); did = False
        while pools[d]:
            m, t = heapq.heappop(pools[d])
            dst = int(top2[t])
            if dst == d:
                continue
            if L[dst] < L[d] - 1:
                L[d] -= 1; L[dst] += 1; chosen[t] = dst; did = True
                break
        if not did:
            break
    return chosen


if __name__ == "__main__":
    main()
