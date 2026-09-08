"""Experiment N — full MoE forward communication (8 layers, real top-6, ASTRA-sim)."""
from __future__ import annotations

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
NET_YML = "/workspace/checkpointing/moe-trace-experiments/exp_N_fullmodel_astrasim/astrasim_work/net/network_switch8.yml"

FLAME_HIDDEN = 1024
FLAME_EP = 8
BYTES_PER_SLOT = FLAME_HIDDEN * 2


def build_matrix(indices, place):
    src = pl.token_source_ranks(len(indices), config.NET_SRC_RANKS, config.NET_SHARDING)
    return tmx.build_dispatch_matrix(indices, place, src, config.NET_SRC_RANKS,
                                     config.NET_DEVICES, dedup_device=config.NET_DEDUP_DEVICE)


def run_astrasim(name: str) -> int:
    workload = f"/workspace/{(WORK / 'et' / name).relative_to(MSC)}"
    cmd = ["docker", "run", "--rm", "-v", f"{MSC}:/workspace", "-w", "/workspace",
           DOCKER_IMAGE, BIN,
           f"--workload-configuration={workload}",
           f"--system-configuration={SYSTEM}",
           f"--network-configuration={NET_YML}",
           f"--remote-memory-configuration={REMOTE}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    times = [int(m) for m in re.findall(r"Comm time: (\d+)", text)]
    if not times:
        raise RuntimeError(f"no Comm time for {name}:\n{text[-1200:]}")
    return max(times)


def main():
    plotting.apply_style()
    import matplotlib.pyplot as plt

    D = config.NET_DEVICES
    assert D == FLAME_EP, f"config NET_DEVICES={D} must match FLAME EP={FLAME_EP}"
    netdir = WORK / "net"
    netdir.mkdir(parents=True, exist_ok=True)
    (netdir / "network_switch8.yml").write_text(
        f"topology: [ Switch ]\nnpus_count: [ {D} ]\nbandwidth: [ 100.0 ]\nlatency: [ 500.0 ]\n")

    place = pl.expert_to_device(config.FLAME_NUM_EXPERTS, D, config.NET_PLACEMENT, config.NET_RANDOM_SEED)
    n = config.SAMPLE_STABILIZATION
    ckpt = config.FLAME_FINAL_CHECKPOINT
    layers = config.FLAME_LAYERS

    jobs = []
    layer_mats = {}
    for layer in layers:
        _, indices = load_layer(ckpt, layer, n, verbose=False)
        M = build_matrix(indices, place)
        layer_mats[layer] = M
        tag = layer.replace("layer_", "L")
        disp = f"{tag}_disp"
        comb = f"{tag}_comb"
        np.save(WORK / "npy" / f"{disp}.npy", M)
        np.save(WORK / "npy" / f"{comb}.npy", M.T)
        jobs.append({"name": disp, "matrix": str(WORK / "npy" / f"{disp}.npy")})
        jobs.append({"name": comb, "matrix": str(WORK / "npy" / f"{comb}.npy")})

    manifest = {"out_dir": str(WORK), "bytes_per_slot": BYTES_PER_SLOT, "mode": "comm_only", "jobs": jobs}
    man_path = WORK / "manifest.json"
    man_path.write_text(json.dumps(manifest))
    print(f"[N] generating Chakra ET for {len(layers)} MoE layers x 2 phases ...")
    g = subprocess.run([str(CHAKRA_PY), str(GEN_ET), "--manifest", str(man_path)],
                       capture_output=True, text=True)
    print(g.stdout + g.stderr)
    if g.returncode != 0:
        raise RuntimeError("gen_et failed")

    rows = []
    print("[N] running ASTRA-sim (Switch) per layer x dispatch/combine ...")
    for layer in layers:
        tag = layer.replace("layer_", "L")
        disp_t = run_astrasim(f"{tag}_disp")
        comb_t = run_astrasim(f"{tag}_comb")
        M = layer_mats[layer]
        rows.append({
            "layer": layer,
            "tokens": n,
            "slots": int(M.sum()),
            "dispatch_comm_cycles": disp_t,
            "combine_comm_cycles": comb_t,
            "layer_comm_cycles": disp_t + comb_t,
        })
        print(f"  {layer}: dispatch={disp_t:,} combine={comb_t:,} total={disp_t + comb_t:,}")

    df = pd.DataFrame(rows)
    df["tokens"] = n
    total = int(df["layer_comm_cycles"].sum())
    df.to_csv(RESULTS / "fullmodel_comm_by_layer.csv", index=False)
    pd.DataFrame([{
        "checkpoint": ckpt,
        "moe_layers": len(layers),
        "collectives": len(layers) * 2,
        "topology": "Switch",
        "bytes_per_slot": BYTES_PER_SLOT,
        "total_moe_forward_comm_cycles": total,
        "mean_layer_comm_cycles": df["layer_comm_cycles"].mean(),
    }]).to_csv(RESULTS / "fullmodel_comm_total.csv", index=False)

    palette = plotting.PALETTE
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["dispatch_comm_cycles"], w, label="dispatch A2A", color=palette[1])
    ax.bar(x + w / 2, df["combine_comm_cycles"], w, label="combine A2A", color=palette[0])
    ax.set_xticks(x)
    ax.set_xticklabels([r["layer"].replace("layer_", "L") for r in rows], rotation=0)
    ax.set_ylabel("ASTRA-sim comm time (cycles)")
    ax.set_xlabel("MoE layer")
    ax.legend()
    ax.set_title(f"Full MoE forward comm (top-6, EP=8, ckpt {ckpt}) — {total/1e6:.2f}M cycles total")
    plotting.save(fig, RESULTS / "fig1_comm_by_layer.png")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.barh([0], [total], color=palette[0], height=0.4)
    ax.set_yticks([0])
    ax.set_yticklabels(["8 MoE layers\n(16 All-to-Alls)"])
    ax.set_xlabel("Total MoE forward comm time (cycles)")
    ax.set_title("Full MoE forward — real top-6 traffic, ASTRA-sim Switch")
    ax.annotate(f"{total:,}", (total, 0), textcoords="offset points", xytext=(6, 0), va="center")
    plotting.save(fig, RESULTS / "fig2_comm_total.png")

    print(f"\n=== Experiment N headline ===")
    print(f"Full MoE forward (8 layers, top-6, dispatch+combine): {total:,} ASTRA-sim comm cycles")
    print(f"  ({len(layers)*2} All-to-All collectives on Switch / EP={FLAME_EP})")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
