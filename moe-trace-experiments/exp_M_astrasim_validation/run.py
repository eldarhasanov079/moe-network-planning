"""Experiment M - ASTRA-sim validation of the MoE All-to-All findings."""

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
from common import traffic_matrix as tmx  # noqa: E402
from common.flame_loader import load_layer  # noqa: E402

RESULTS = HERE / "results"
WORK = HERE / "astrasim_work"
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

NET_DIR = "/workspace/checkpointing/moe-trace-experiments/exp_M_astrasim_validation/astrasim_work/net"
TOPOLOGIES = {
    "Switch": f"{NET_DIR}/network_switch8.yml",
    "Ring": f"{EX}/network/analytical/Ring_8npus.yml",
    "FullyConnected": f"{NET_DIR}/network_fc8.yml",
}
PRIMARY_TOPO = "Switch"
HIDDEN_SIZE = 1024
PEAK_GFLOPS = 900.0
BW_GB_S = 100.0
ATTN_GATE_FLOPS_PER_TOKEN = 10 * HIDDEN_SIZE * HIDDEN_SIZE
EXPERT_FLOPS_PER_SLOT = 16 * HIDDEN_SIZE * HIDDEN_SIZE


def build_membership_matrix(indices, place):
    src = pl.token_source_ranks(len(indices), config.NET_SRC_RANKS, config.NET_SHARDING)
    return tmx.build_dispatch_matrix(indices, place, src, config.NET_SRC_RANKS,
                                     config.NET_DEVICES, dedup_device=config.NET_DEDUP_DEVICE)


def top1_matrix(top1_dev, src, D, S):
    M = np.zeros((S, D), dtype=np.int64)
    np.add.at(M, (src, top1_dev), 1)
    return M


def margin_deflect_matrix(top1_dev, top2_dev, margin, src, D, S, relief_target=1.05):
    """Move lowest-margin tokens off the busiest device to their top-2 device until the peak"""
    M = top1_matrix(top1_dev, src, D, S).astype(np.float64)
    Ld = M.sum(axis=0); mean = Ld.sum() / D
    pools = [[] for _ in range(D)]
    for t in range(len(top1_dev)):
        pools[int(top1_dev[t])].append((float(margin[t]), t))
    for d in range(D):
        heapq.heapify(pools[d])
    while Ld.max() / mean > relief_target:
        d = int(np.argmax(Ld)); pool = pools[d]; did = False
        while pool:
            m, t = heapq.heappop(pool)
            dst = int(top2_dev[t])
            if dst == d:
                continue
            if Ld[dst] < Ld[d] - 1:
                s = int(src[t]); M[s, d] -= 1; M[s, dst] += 1
                Ld[d] -= 1; Ld[dst] += 1; did = True
                break
        if not did:
            break
    return M


def run_astrasim(name, topo=PRIMARY_TOPO, et_subdir="et", metric="comm"):
    """Run ASTRA-sim; return max per-rank ``metric`` time ('comm' or 'wall')."""
    workload = f"/workspace/{(WORK / et_subdir / name).relative_to(MSC)}"
    cmd = ["docker", "run", "--rm", "-v", f"{MSC}:/workspace", "-w", "/workspace",
           DOCKER_IMAGE, BIN,
           f"--workload-configuration={workload}",
           f"--system-configuration={SYSTEM}",
           f"--network-configuration={TOPOLOGIES[topo]}",
           f"--remote-memory-configuration={REMOTE}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    key = "Wall time" if metric == "wall" else "Comm time"
    times = [int(m) for m in re.findall(rf"{key}: (\d+)", text)]
    if not times:
        raise RuntimeError(f"no {key} parsed for {name} ({et_subdir}):\n{text[-1500:]}")
    return max(times)


def _flops_to_comm_cycles(flops: float, comm_ref_cycles: int, comm_ref_bytes: float) -> int:
    """Map compute FLOPs to ASTRA-sim 'cycles' by calibrating against measured comm time."""
    comm_s = comm_ref_bytes / (BW_GB_S * 1e9)
    compute_s = flops / (PEAK_GFLOPS * 1e9)
    if comm_s <= 0:
        return 0
    return max(1, int(comm_ref_cycles * (compute_s / comm_s)))


def e2e_wall_cycles(M: np.ndarray, comm_dispatch: int, comm_combine: int, bytes_per_slot: int) -> int:
    """One MoE EP layer forward: attn+gating + dispatch + expert + combine."""
    D = M.shape[0]
    total_bytes = float(M.sum() * bytes_per_slot)
    compute_per_rank = []
    for r in range(D):
        local_tok = float(M[r, :].sum())
        recv_slots = float(M[:, r].sum())
        attn = _flops_to_comm_cycles(local_tok * ATTN_GATE_FLOPS_PER_TOKEN, comm_dispatch, total_bytes)
        expert = _flops_to_comm_cycles(recv_slots * EXPERT_FLOPS_PER_SLOT, comm_dispatch, total_bytes)
        compute_per_rank.append(attn + expert)
    return max(compute_per_rank) + comm_dispatch + comm_combine


def main():
    plotting.apply_style()
    import matplotlib.pyplot as plt

    D = config.NET_DEVICES; S = config.NET_SRC_RANKS
    netdir = WORK / "net"; netdir.mkdir(parents=True, exist_ok=True)
    (netdir / "network_switch8.yml").write_text(
        f"topology: [ Switch ]\nnpus_count: [ {D} ]\nbandwidth: [ 100.0 ]\nlatency: [ 500.0 ]\n")
    (netdir / "network_fc8.yml").write_text(
        f"topology: [ FullyConnected ]\nnpus_count: [ {D} ]\nbandwidth: [ 100.0 ]\nlatency: [ 500.0 ]\n")
    place = pl.expert_to_device(config.FLAME_NUM_EXPERTS, D, config.NET_PLACEMENT, config.NET_RANDOM_SEED)
    n = config.SAMPLE_STABILIZATION
    layer = config.FLAME_LAYERS[0]                 # representative layer
    final = config.FLAME_FINAL_CHECKPOINT

    jobs = []   # (name, matrix)
    stab_ckpts = [config.FLAME_CHECKPOINTS[1], config.FLAME_CHECKPOINTS[3], final]
    for c in stab_ckpts:
        M = build_membership_matrix(load_layer(c, layer, n)[1], place)
        jobs.append((f"stab_ckpt{c}", M.astype(np.float64)))

    scores, indices = load_layer(final, layer, n)
    top1 = indices[:, 0].astype(np.int64); top2 = indices[:, 1].astype(np.int64)
    margin = (scores[:, 0] - scores[:, 1]).astype(np.float64)
    src = pl.token_source_ranks(len(indices), S, config.NET_SHARDING)
    M_raw = top1_matrix(place[top1], src, D, S).astype(np.float64)
    M_def = margin_deflect_matrix(place[top1], place[top2], margin, src, D, S)
    jobs.append(("defl_raw", M_raw))
    jobs.append(("defl_deflected", M_def))

    manifest = {"out_dir": str(WORK), "bytes_per_slot": BYTES_PER_SLOT, "jobs": []}
    for name, M in jobs:
        p = WORK / "npy" / f"{name}.npy"
        np.save(p, M)
        manifest["jobs"].append({"name": name, "matrix": str(p)})
    man_path = WORK / "manifest.json"
    man_path.write_text(json.dumps(manifest))
    print("[M] generating Chakra ET via chakra_env ...")
    g = subprocess.run([str(CHAKRA_PY), str(HERE / "gen_et.py"), "--manifest", str(man_path)],
                       capture_output=True, text=True)
    print(g.stdout + g.stderr)
    if g.returncode != 0:
        raise RuntimeError("gen_et.py failed")

    print(f"[M] running ASTRA-sim analytical (Docker), primary topology = {PRIMARY_TOPO} ...")
    comm = {name: run_astrasim(name) for name, _ in jobs}    # primary topology
    for k, v in comm.items():
        print(f"  {k}: comm_time={v} cycles")

    defl_topo = {t: {v: run_astrasim(v, t) for v in ("defl_raw", "defl_deflected")}
                 for t in TOPOLOGIES}

    # ---- tables ----
    stab = pd.DataFrame([{"checkpoint": str(c), "comm_time_cycles": comm[f"stab_ckpt{c}"]}
                         for c in stab_ckpts])
    stab["pct_vs_final"] = 100 * (stab["comm_time_cycles"] / comm[f"stab_ckpt{final}"] - 1)
    stab.to_csv(RESULTS / "stability_commtime.csv", index=False)
    defl = pd.DataFrame([
        {"topology": t, "raw_cycles": defl_topo[t]["defl_raw"],
         "deflected_cycles": defl_topo[t]["defl_deflected"],
         "reduction_pct": 100 * (1 - defl_topo[t]["defl_deflected"] / defl_topo[t]["defl_raw"])}
        for t in TOPOLOGIES
    ])
    defl.to_csv(RESULTS / "deflection_commtime.csv", index=False)

    # ---- figures ----
    palette = plotting.PALETTE
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(range(len(stab_ckpts)), stab["comm_time_cycles"], "o-", color=palette[0], lw=2)
    ax.set_xticks(range(len(stab_ckpts))); ax.set_xticklabels([str(c) for c in stab_ckpts])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Checkpoint"); ax.set_ylabel(f"ASTRA-sim comm time, {PRIMARY_TOPO} (cycles)")
    ax.set_title("V1: comm time is stable across training (ASTRA-sim)")
    plotting.save(fig, RESULTS / "fig1_stability_commtime.png")

    x = np.arange(len(TOPOLOGIES)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w / 2, defl["raw_cycles"], w, label="raw top-1", color=palette[1])
    ax.bar(x + w / 2, defl["deflected_cycles"], w, label="margin-deflected", color=palette[0])
    for i, r in enumerate(defl["reduction_pct"]):
        ax.annotate(f"-{r:.1f}%", (x[i] + w / 2, defl["deflected_cycles"].iloc[i]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(defl["topology"])
    ax.set_ylabel("ASTRA-sim All-to-All comm time (cycles)"); ax.legend()
    ax.set_title("V2: margin-aware deflection lowers comm time (largest when egress-bound)")
    plotting.save(fig, RESULTS / "fig2_deflection_commtime.png")

    print("[M] measuring combine All-to-All comm (M.T) for deflection pair ...")
    comb_jobs = []
    for name in ("defl_raw", "defl_deflected"):
        M = np.load(WORK / "npy" / f"{name}.npy")
        cname = f"{name}_comb"
        np.save(WORK / "npy" / f"{cname}.npy", M.T)
        comb_jobs.append({"name": cname, "matrix": str(WORK / "npy" / f"{cname}.npy")})
    comb_manifest = {"out_dir": str(WORK), "bytes_per_slot": BYTES_PER_SLOT, "mode": "comm_only",
                     "jobs": comb_jobs}
    comb_man_path = WORK / "manifest_comb.json"
    comb_man_path.write_text(json.dumps(comb_manifest))
    g2 = subprocess.run([str(CHAKRA_PY), str(HERE / "gen_et.py"), "--manifest", str(comb_man_path)],
                        capture_output=True, text=True)
    print(g2.stdout + g2.stderr)
    if g2.returncode != 0:
        raise RuntimeError("gen_et.py (combine) failed")

    M_raw = np.load(WORK / "npy" / "defl_raw.npy")
    M_def = np.load(WORK / "npy" / "defl_deflected.npy")
    e2e = []
    for t in TOPOLOGIES:
        disp_raw = defl_topo[t]["defl_raw"]
        disp_def = defl_topo[t]["defl_deflected"]
        comb_raw = run_astrasim("defl_raw_comb", t)
        comb_def = run_astrasim("defl_deflected_comb", t)
        wall_raw = e2e_wall_cycles(M_raw, disp_raw, comb_raw, BYTES_PER_SLOT)
        wall_def = e2e_wall_cycles(M_def, disp_def, comb_def, BYTES_PER_SLOT)
        e2e.append({
            "topology": t,
            "raw_cycles": wall_raw,
            "deflected_cycles": wall_def,
            "reduction_pct": 100 * (1 - wall_def / wall_raw),
            "comm_only_reduction_pct": float(defl.loc[defl["topology"] == t, "reduction_pct"].iloc[0]),
            "dispatch_comm_raw": disp_raw, "combine_comm_raw": comb_raw,
        })
    e2e = pd.DataFrame(e2e)
    e2e.to_csv(RESULTS / "deflection_walltime.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w / 2, e2e["raw_cycles"], w, label="raw top-1", color=palette[1])
    ax.bar(x + w / 2, e2e["deflected_cycles"], w, label="margin-deflected", color=palette[0])
    for i, r in enumerate(e2e["reduction_pct"]):
        ax.annotate(f"-{r:.1f}%", (x[i] + w / 2, e2e["deflected_cycles"].iloc[i]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(e2e["topology"])
    ax.set_ylabel("MoE-layer wall time (cycles)")
    ax.legend()
    ax.set_title("V2 (end-to-end): one MoE layer — attn + dispatch + expert + combine")
    plotting.save(fig, RESULTS / "fig2_deflection_walltime.png")

    prim = defl[defl["topology"] == PRIMARY_TOPO].iloc[0]
    prim_e2e = e2e[e2e["topology"] == PRIMARY_TOPO].iloc[0]
    print("\n=== Experiment M (ASTRA-sim) headline ===")
    print(stab.to_string(index=False))
    print(defl.to_string(index=False))
    print("\n--- end-to-end MoE layer (one layer forward) ---")
    print(e2e.to_string(index=False))
    print(f"\nHEADLINE: on the external ASTRA-sim analytical engine, All-to-All comm time "
          f"varies only {stab['pct_vs_final'].abs().max():.1f}% across checkpoints (corroborates "
          f"Exp E/H stability); margin-aware deflection cuts comm time by {prim['reduction_pct']:.1f}% "
          f"on the egress-bound {PRIMARY_TOPO} fabric (corroborates Exp K). "
          f"End-to-end one MoE layer wall time drops by {prim_e2e['reduction_pct']:.1f}% on {PRIMARY_TOPO} "
          f"(vs {prim_e2e['comm_only_reduction_pct']:.1f}% comm-only).")


if __name__ == "__main__":
    main()
