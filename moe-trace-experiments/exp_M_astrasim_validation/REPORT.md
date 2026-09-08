# Exp M: ASTRA-sim validation of the All-to-All findings

| | |
|---|---|
| Question | Do the stability (Exp E/H) and deflection (Exp K) All-to-All results hold on ASTRA-sim (analytical congestion-aware backend, ASTRA-sim 2.0)? |
| Data | FLAME-MoE router `indices`/`scores`; checkpoints 1080 / 2160 / 5473; top-1 matrices (~250 k slots) and top-6 membership matrices (~1.5 M slots) |
| Model | dispatch matrix `M[src, dst]` encoded as point-to-point `SEND`/`RECV` transfers in a Chakra execution trace; `bytes_per_slot = 2048`; collective time = max over ranks of `Comm time` from ASTRA-sim stdout |
| Run | `python exp_M_astrasim_validation/run.py` (builds matrices, `.npy` → `gen_et.py` → Docker → parse); needs the `astra-sim:latest` Docker image and a Python environment with the `chakra` package |
| Outputs | `results/stability_commtime.csv`, `results/deflection_commtime.csv`, `results/deflection_walltime.csv`, `results/fig2_deflection_commtime.png`, `results/fig2_deflection_walltime.png` |

## Setup

| Item | Value |
|---|---|
| Binary | `AstraSim_Analytical_Congestion_Aware` in the `astra-sim:latest` image (built from the `astra-sim/` submodule), repo mounted at `/workspace` |
| Configs | system `Ring_4chunks.json`, remote memory `no_memory_expansion.json` (stock ASTRA-sim examples) |
| Execution trace | one `.et` per NPU, written with MoE-MLSynth helpers (`utils.send/receive`, `encodeMessage`); one transfer per non-zero `(src, dst)` pair sized `M[src,dst] × bytes_per_slot` |
| Node ordering | per rank: root `COMP` node, then all RECVs, then all SENDs, each depending only on the root; a blocking send before the matching receive deadlocks |
| Topologies | Switch (one shared leaf-switch uplink per device, egress-bound), Ring (hop-distance congestion), FullyConnected (dedicated link per pair) |
| Layer wall time | `wall = max_rank(attn + expert compute) + dispatch comm + combine comm`; comm from ASTRA-sim (`M` and `M.T`), compute from a FLOP model (MoE-MLSynth-style, h=1024, peak 900 GFLOPS from `Ring_4chunks.json`) calibrated to measured dispatch comm time |

## Results

| ID | Claim | ASTRA-sim result |
|----|---|---|
| V1 | Stability: dispatch traffic stable across training (Exp E/H) | comm time varies ≤ 2.6% across checkpoints 1080 / 2160 / 5473 |
| V2 | Deflection: margin-aware deflection of the top-1 matrix lowers collective time (Exp K) | −6.5% comm time on Switch (−3.4% Ring, −1.1% FullyConnected) |

| Topology | Bottleneck | Deflection comm-time change |
|---|---|---|
| Switch | busiest device egress | −6.5% |
| Ring | hop-distance congestion | −3.4% |
| FullyConnected | single largest flow | −1.1% |

Figures: `results/fig2_deflection_commtime.png` (comm only, raw vs deflected, three fabrics); `results/fig2_deflection_walltime.png` (one MoE layer forward pass: attn+gating + dispatch + expert + combine).

Scale check: membership matrices (top-6, ~1.5 M slots) give ~9.0–9.3 M cycles; top-1 matrices (~250 k slots, 6× less volume) give ~1.6 M cycles on the same fabric.

## Finding
- Comm time varies ≤ 2.6% across checkpoints 1080 / 2160 / 5473.
- Margin-aware deflection cuts comm time by 6.5% on Switch, 3.4% on Ring, 1.1% on FullyConnected.
- Deflection gain is largest on the egress-bound fabric; wall-time gain can exceed comm-only gain because hot devices also receive fewer tokens.

## Limits
- The per-flow ECMP-vs-plan rail comparison of Exp I is not reproduced; the analytical backend does not expose per-flow rail assignment (ns-3 / Garnet backends needed).
- Fluid model, no per-packet queueing; the backend cannot run COMP nodes with `duration_micros`, so compute is added analytically.
- Sampled-trace matrices (first N tokens); absolute cycle counts are illustrative, ratios are the result.
