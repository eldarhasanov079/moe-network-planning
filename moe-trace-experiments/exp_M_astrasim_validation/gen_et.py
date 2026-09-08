"""Convert a saved source->destination traffic matrix into a Chakra ET workload for ASTRA-sim."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MLSYNTH = Path.home() / "Desktop/msc/iso_code/MLSynth"
sys.path.insert(0, str(MLSYNTH))

from utils import send, receive, compute, reset_id_counter  # noqa: E402
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata  # noqa: E402
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message  # noqa: E402


def _flops_to_duration_us(flops: int, peak_gflops: float) -> int:
    """Map FLOPs to Chakra duration_micros using ASTRA-sim peak-perf (GFLOPS)."""
    return max(1, int(flops / (peak_gflops * 1e9) * 1e6))


def _write_p2p(nodes, rank, M, bps, parents, tag_base, prefix):
    """Post RECVs then SENDs for matrix M on ``rank``; return all comm nodes."""
    D = M.shape[0]
    comm = []
    for s in range(D):
        if s != rank and M[s, rank] > 0:
            n = receive(s, rank, int(M[s, rank] * bps), name=f"{prefix}_RECV_{s}_{rank}",
                        parents=parents, comm_tag=tag_base + s * D + rank)
            nodes[rank].append(n)
            comm.append(n)
    for d in range(D):
        if d != rank and M[rank, d] > 0:
            n = send(rank, d, int(M[rank, d] * bps), name=f"{prefix}_SEND_{rank}_{d}",
                     parents=parents, comm_tag=tag_base + rank * D + d)
            nodes[rank].append(n)
            comm.append(n)
    return comm


def write_workload_comm_only(M, name, out_dir, bytes_per_slot, et_subdir="et"):
    D = M.shape[0]
    out = Path(out_dir)
    (out / et_subdir).mkdir(parents=True, exist_ok=True)
    reset_id_counter(0)
    nodes = defaultdict(list)
    for r in range(D):
        nodes[r].append(GlobalMetadata(version="0.0.4"))
        root = compute(1, 1, name="ROOT")
        nodes[r].append(root)
        _write_p2p(nodes, r, M, bytes_per_slot, [root], 0, "DISP")
    for npu_id in range(D):
        with open(out / et_subdir / f"{name}.{npu_id}.et", "wb") as f:
            for node in nodes[npu_id]:
                encode_message(f, node)
    return D


def write_workload_moe_layer(M, name, out_dir, bytes_per_slot, compute_cfg, et_subdir="et_e2e"):
    """One MoE layer: local attn+gating -> dispatch -> expert compute -> combine (M.T)."""
    D = M.shape[0]
    out = Path(out_dir)
    (out / et_subdir).mkdir(parents=True, exist_ok=True)
    H = int(compute_cfg["hidden_size"])
    peak = float(compute_cfg["peak_gflops"])
    attn_gate_per_token = int(10 * H * H)
    expert_per_slot = int(16 * H * H)
    Mt = M.T

    reset_id_counter(0)
    nodes = defaultdict(list)
    for r in range(D):
        nodes[r].append(GlobalMetadata(version="0.0.4"))
        local_tokens = int(M[r, :].sum())
        recv_slots = int(M[:, r].sum())
        pre = compute(1, 1, name="ATTN_GATE",
                      duration_micros=_flops_to_duration_us(attn_gate_per_token * max(local_tokens, 1), peak))
        nodes[r].append(pre)
        disp_recvs = []
        for s in range(D):
            if s != r and M[s, r] > 0:
                n = receive(s, r, int(M[s, r] * bytes_per_slot), name=f"DISP_RECV_{s}_{r}",
                            parents=[pre], comm_tag=s * D + r)
                nodes[r].append(n)
                disp_recvs.append(n)
        disp_sends = []
        for d in range(D):
            if d != r and M[r, d] > 0:
                n = send(r, d, int(M[r, d] * bytes_per_slot), name=f"DISP_SEND_{r}_{d}",
                         parents=[pre], comm_tag=10_000 + r * D + d)
                nodes[r].append(n)
                disp_sends.append(n)
        expert_parent = disp_recvs if disp_recvs else [pre]
        expert = compute(1, 1, name="EXPERT", parents=expert_parent,
                         duration_micros=_flops_to_duration_us(expert_per_slot * max(recv_slots, 1), peak))
        nodes[r].append(expert)
        comb_recvs = []
        for s in range(D):
            if s != r and Mt[s, r] > 0:
                n = receive(s, r, int(Mt[s, r] * bytes_per_slot), name=f"COMB_RECV_{s}_{r}",
                            parents=[expert], comm_tag=20_000 + s * D + r)
                nodes[r].append(n)
                comb_recvs.append(n)
        for d in range(D):
            if d != r and Mt[r, d] > 0:
                nodes[r].append(send(r, d, int(Mt[r, d] * bytes_per_slot), name=f"COMB_SEND_{r}_{d}",
                                     parents=comb_recvs if comb_recvs else [expert],
                                     comm_tag=30_000 + r * D + d))
    for npu_id in range(D):
        with open(out / et_subdir / f"{name}.{npu_id}.et", "wb") as f:
            for node in nodes[npu_id]:
                encode_message(f, node)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text())
    out_dir = man["out_dir"]
    bps = man["bytes_per_slot"]
    mode = man.get("mode", "comm_only")
    et_subdir = man.get("et_subdir", "et_e2e" if mode == "moe_layer" else "et")
    compute_cfg = man.get("compute", {"hidden_size": 1024, "peak_gflops": 900})
    D = None
    for job in man["jobs"]:
        M = np.load(job["matrix"])
        if mode == "moe_layer":
            D = write_workload_moe_layer(M, job["name"], out_dir, bps, compute_cfg, et_subdir)
        else:
            D = write_workload_comm_only(M, job["name"], out_dir, bps, et_subdir)
        print(f"wrote ET ({mode}) for {job['name']} ({D} npus) -> {et_subdir}/")
    if D is not None:
        Path(out_dir, "comm_groups.json").write_text(json.dumps({"0": list(range(D))}))


if __name__ == "__main__":
    main()
