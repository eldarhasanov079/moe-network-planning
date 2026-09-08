# Netmodel toolkit (`common/`)

Shared layer that turns placement-free router traces into source-to-destination traffic matrices, maps them onto a topology for per-link loads, and scores them with similarity and envelope metrics. Used by Exp H and the later experiments. Analysis only: no time, queueing or bandwidth. Defaults live in `config.py` under the `NET_*` block (8 devices, 8 source ranks, contiguous placement+sharding, 2 nodes of 4, per-slot counting).

## Modules
| Module | Role | Key functions |
|---|---|---|
| `placement.py` | Where experts live + which rank made each token | `expert_to_device`, `token_source_ranks` |
| `traffic_matrix.py` | Build the dispatch All-to-All matrix `M[src,dst]` | `build_dispatch_matrix`, `as_distribution` |
| `topology.py` | Map `M` → per-link loads | `FullMeshFabric`, `TwoTierNodes` |
| `netmetrics.py` | Similarity, link stats, envelope split | `cosine`, `norm_l1`, `topk_flow_overlap`, `link_stats`, `hot_link_overlap`, `build_envelope`, `split_matrix`, `envelope_metrics` |
| `netsim.py` | Flow-level All-to-All simulator → collective time | `ideal_maxload`, `maxload_under`, `ecmp_assign`, `greedy_assign`, `normalized_time`, `ecmp_normalized_time` |

## Pipeline
```
trace indices (n_tokens, top_k)                  # real router decisions (FLAME/OLMoE)
    + expert→device map, token→source-rank map   # placement.py  (ASSUMED)
M[src, dst]  (S × D)                             # traffic_matrix.py  forward dispatch, token-slots
    + topology                                   # topology.py
{link: load}                                     # per-link loads (bytes∝slots), NOT times
cosine / L1 / overlap / envelope split           # netmetrics.py
```

## Assumptions
| ID | Assumption |
|---|---|
| A1 | Static, even expert placement: `num_experts % num_devices == 0`, equal experts per device (FLAME: 64/8 = 8). |
| A2 | Placement is constant across training checkpoints. |
| A3 | Tokens are partitioned into source ranks by a synthetic map (real rank ids are not in the traces): default `contiguous` (consecutive rows = one rank's microbatch, heterogeneous sources); `strided` control gives near-i.i.d. sources. Placement scheme (`contiguous` / `roundrobin` / `random`) is swept in Exp H. |
| B1 | Only the forward dispatch All-to-All is modelled; combine is its transpose and backward collectives mirror it. |
| B2 | Traffic is counted in token-slots, not bytes; bytes = slots × constant activation size, so ratios, cosines and overflows are unchanged. |
| B3 | top-k membership is the traffic: a token reaches every device holding one of its top-k experts. |
| B4 | Co-located experts are counted per slot by default (2 experts on one device = 2 slots, as in Exp E); `dedup_device=True` counts once per device. |
| B5 | The diagonal `M[s,s]` is on-device traffic with zero network load (excluded by the topology layer). |
| C0 | Outputs are link loads (slots), not times; no queueing, bandwidth, latency or contention. |
| C1 | `FullMeshFabric` (default): non-blocking full-bisection fabric; directed link `(s,d)` load = off-diagonal `M[s,d]`; no routing or ECMP assumption. |
| C2/C3 | `TwoTierNodes`: devices grouped into nodes of `node_size`; dedicated intra-node links; one shared uplink and one downlink per node to a single spine; no multi-spine ECMP. |
| D1 | Envelope = per-entry upper-bound reservation: `base=min(M,E)`, `refund=max(E−M,0)`, `overflow=max(M−E,0)`; no time or queueing. |
| E1 | Rail-optimised All-to-All: each device has R parallel rails; each device-pair flow is assigned to one rail; bottleneck = busiest (device, rail, direction) link. |
| E2 | A policy controls only the rail assignment (ECMP random hash / greedy-LPT planned / oracle); static plans are computed once on a profiling window and frozen. |
| E3 | Fluid time: `collective_time ∝ max_link_load`, reported as `max_link_load / ideal` with `ideal = busiest device I/O / R`; baseline is per-flow ECMP; no packet dynamics. |
| F1 | Matrices are exported to ASTRA-sim (analytical congestion-aware backend); `gen_et.py` writes per-NPU Chakra execution traces with explicit P2P `SEND`/`RECV` nodes (one per non-zero pair, size `M[src,dst]×bytes_per_slot`), which preserves the heterogeneous matrix. |
| F2 | Deadlock-safe ET: each rank posts a root `COMP`, then all RECVs, then all SENDs, each depending only on the root; collective time = max over ranks of parsed `Comm time`. |
| F3 | `bytes_per_slot=2048` sets absolute scale only; all reported claims are ratios. |
| F4 | Topology decides the bottleneck: Switch = shared per-device leaf uplink (egress-bound); Ring = hop-distance congestion; FullyConnected = dedicated per-pair link. |
| F5 | Run via the `astra-sim:latest` Docker image, the prebuilt Linux analytical binary and a Python environment with the `chakra` package (plus MoE-MLSynth `utils.send/receive`). |
| F6 | The analytical backend is fluid, like `netsim.py`; it validates load-to-time, stability and deflection, but does not expose per-flow rail routing, so Exp I's ECMP-vs-plan comparison is not reproduced there. |

## Out of scope
- Per-packet queueing delay and congestion-control dynamics.
- Packet-spraying ECMP and the packet-level ASTRA-sim backends (ns-3 / Garnet) needed for Exp I's per-flow rail-routing speedup.
