"""Export one Stage 2 step as inputs for a packet-level (ns-3) run — prepared, not run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .config import DEFAULT_TOPOLOGY
from .planner import freeze_paths
from .policy import slots_to_bytes

CLASSES = [
    {
        "name": "reserved",
        "priority": 0,
        "dscp": 46,
        "queue": "strict-high",
        "routing": "frozen",
        "congestion_control": "none (planned rates; enforce with per-flow rate schedule or leave uncapped)",
    },
    {
        "name": "tail",
        "priority": 1,
        "dscp": 0,
        "queue": "strict-low",
        "routing": "ecmp | spray",
        "congestion_control": "DCQCN on this queue only",
    },
]

README = """# ns-3 packet-level check for the Stage 2 two-class All-to-All (prepared, not run)

This directory holds one step of one layer in one direction pair (dispatch + combine),
split into the two Stage 2 traffic classes. Everything is in bytes; no tokens.

## Files
- `topology.json` — 2 pods x 2 leaves x 2 GPUs, 2 spines per pod, 2 super-spines,
  100 Gbps links, 500 ns per hop. Same fabric as Puppeteer's `topology_8gpu_clos.yaml`.
- `classes.json` — reserved = strict high-priority queue, frozen single path per
  (src,dst); tail = strict low-priority queue, ECMP (per-flow hash) or spray
  (per-packet round-robin over all equal-cost paths), DCQCN enabled on this queue only.
- `reserved_*.csv`, `tail_*.csv` — per (src,dst) bytes. Combine is the transpose phase.
  Two kernel structures, as in the fluid Stage 2b runs: **monolithic (today)** — rank d
  starts its expert kernel when its all-to-all completes for BOTH classes (NCCL
  semantics: all of d's sends and receives done), then sends its combine; **chunked
  (proposal)** — rank d runs the reserved-chunk kernel as soon as the reserved
  all-to-all completes, then the tail-chunk kernel, with two combine collectives.
  Kernel time per slot and a launch floor are in `manifest.json`.
- `frozen_paths.json` — the exact link sequence the reserved class must take.

## Protocol (when a binary is available)
1. Build the Clos in ns-3 with two strict-priority egress queues per port; map DSCP 46
   to the high queue and DSCP 0 to the low queue. Enable PFC on both, DCQCN on the low
   queue only.
2. Install static routes for the reserved class from `frozen_paths.json`; use the
   switch ECMP hash (or per-packet spray) for the tail class.
3. At t=0 every rank starts its dispatch flows of both classes; each flow is one
   RoCE QP, MTU 4096.
4. Measure per class: completion time of the last byte, per-link busy time, queue
   occupancy on leaf->GPU downlinks, PFC pause time, and (spray only) out-of-order
   packets at the receiver.
5. Compare with `stage2/clos.csv` (Puppeteer two-class, fluid) and `stage2/astrasim.csv`
   (ASTRA-sim bracket) for the same (source, layer, checkpoint, B, block, policy).

## What would confirm / refute the fluid result
- Reserved class completion within a few percent of the Puppeteer reserved makespan
  and unchanged when the tail is added: strict priority works on real queues.
- Tail completion between the ASTRA-sim one-class and serialized numbers.
- If PFC pauses from the low queue delay the high queue, the tail must be rate-limited
  (DCQCN) harder or dropped at a deadline; that is the packet-level result Stage 2's
  fluid runs cannot see.
"""


def _write_matrix(path: Path, M_bytes: np.ndarray, paths: Optional[Dict[Tuple[int, int], object]] = None,
                  routing: str = "frozen") -> None:
    n = int(M_bytes.shape[0])
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src", "dst", "bytes", "routing", "path"])
        for s in range(n):
            for d in range(n):
                if s == d or int(M_bytes[s, d]) <= 0:
                    continue
                route = ""
                if paths is not None:
                    r = paths.get((s, d))
                    route = "|".join(r.links) if r is not None else ""
                writer.writerow([s, d, int(M_bytes[s, d]), routing, route])


def export_step(
    out_dir: str,
    *,
    manifest: Dict,
    M_reserved_slots: np.ndarray,
    M_tail_slots: np.ndarray,
    route_slots: np.ndarray,
    tail_routing: str = "ecmp",
    bytes_per_slot: int = 2048,
    topology: str = str(DEFAULT_TOPOLOGY),
) -> Path:
    """Write the ns-3 input bundle for one step. Returns the directory."""
    from puppeteer.config import RunConfig

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = RunConfig.load(topology)
    topo = config.topology
    paths_d = freeze_paths(route_slots, config, bytes_per_slot=bytes_per_slot)
    paths_c = freeze_paths(np.asarray(route_slots).T.copy(), config, bytes_per_slot=bytes_per_slot)

    R = slots_to_bytes(M_reserved_slots, bytes_per_slot)
    T = slots_to_bytes(M_tail_slots, bytes_per_slot)
    _write_matrix(out / "reserved_dispatch.csv", R, paths_d, "frozen")
    _write_matrix(out / "tail_dispatch.csv", T, None, tail_routing)
    _write_matrix(out / "reserved_combine.csv", R.T.copy(), paths_c, "frozen")
    _write_matrix(out / "tail_combine.csv", T.T.copy(), None, tail_routing)

    links = []
    for link_id, link in sorted(topo.links.items()):
        links.append({
            "id": link_id, "src": link.src, "dst": link.dst,
            "capacity_bps": float(link.capacity_bps),
            "latency_ns": 500,
        })
    (out / "topology.json").write_text(json.dumps({
        "type": "clos", "num_ranks": int(R.shape[0]),
        "host_of_rank": {str(r): topo.location(r).host for r in range(int(R.shape[0]))},
        "links": links, "mtu_bytes": 4096,
    }, indent=1) + "\n")
    (out / "frozen_paths.json").write_text(json.dumps({
        "dispatch": {"{}-{}".format(s, d): list(r.links) for (s, d), r in paths_d.items()},
        "combine": {"{}-{}".format(s, d): list(r.links) for (s, d), r in paths_c.items()},
    }, indent=1) + "\n")
    classes = json.loads(json.dumps(CLASSES))
    classes[1]["routing"] = tail_routing
    (out / "classes.json").write_text(json.dumps(classes, indent=1) + "\n")
    man = dict(manifest)
    man.update({
        "bytes_per_slot": bytes_per_slot,
        "reserved_bytes": int(R.sum()), "tail_bytes": int(T.sum()),
        "tail_share": float(T.sum() / max(R.sum() + T.sum(), 1)),
        "packets_at_mtu_4096_dispatch": int(np.ceil((R.sum() + T.sum()) / 4096)),
    })
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    (out / "README.md").write_text(README)
    return out
