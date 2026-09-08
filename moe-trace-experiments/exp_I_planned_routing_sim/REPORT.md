# Exp I: Planned routing simulation

| | |
|---|---|
| Question | Does a rail-assignment plan computed from the predicted dispatch matrix, frozen from early checkpoints, lower All-to-All collective time vs ECMP, and how close is it to a per-step oracle? |
| Data | FLAME-MoE-290M (profile iters 1080–3240, test 3780–5473); OLMoE-1B-7B (profile 5k–245k, test 490k+final); traces and deployment model as Exp H |
| Model | rail-optimised All-to-All, 8 devices, R=4 rails; fluid model, collective_time ∝ max_link_load; normalized time = max_link_load / ideal, ideal = busiest device's I/O / R; diagonal excluded (`common/netsim.py`) |
| Run | `python exp_I_planned_routing_sim/run.py` |
| Outputs | `results/sim_policy_summary.csv`, `sim_detail.csv`, `sim_rail_sweep.csv`, `headline.csv` (OLMoE: `olmoe_*`); `fig1_time_by_policy.png`, `fig2_frozen_plan_over_time.png`, `fig3_rail_sweep.png`, `fig4_cross_model_summary.png` (OLMoE: `olmoe_fig1`–`olmoe_fig3`) |

## Setup

| Policy | Rail assignment `A[s,d]` |
|---|---|
| ECMP | oblivious per-flow hash, averaged over 256 hashes |
| static-mean | balanced on the profiling-window mean matrix, computed once, frozen |
| static-P95 | balanced on the per-entry P95 envelope, frozen |
| static-worst | balanced on the worst-case envelope, frozen |
| oracle | replanned per checkpoint (upper bound) |

## Results

FLAME-MoE-290M (`sim_policy_summary.csv`, averaged over layers and held-out checkpoints):

| Policy | normalized time | speedup vs ECMP | gap to oracle |
|---|---|---|---|
| ECMP (oblivious) | 2.58× | 1.00× | +96% |
| static-mean (frozen) | **1.33×** | **1.95×** | +0.9% |
| static-P95 (frozen) | 1.50× | 1.72× | +14% |
| static-worst (frozen) | 1.52× | 1.70× | +15% |
| oracle (per-step) | 1.31× | 1.96× | 0% |

OLMoE-1B-7B (`olmoe_sim_policy_summary.csv`):

| Policy | normalized time | speedup vs ECMP | gap to oracle |
|---|---|---|---|
| ECMP | 2.36× | 1.00× | +90% |
| static-mean (frozen) | **1.26×** | **1.87×** | +1.2% |
| static-P95 (frozen) | 1.26× | 1.87× | +1.4% |
| oracle | 1.24× | 1.90× | 0% |

- Rail sweep (`fig3_rail_sweep.png`, `sim_rail_sweep.csv`): at R=8 ECMP ~3.9× vs planned ~1.6×; at R=2 the gain is 1.68× → 1.14×.
- Frozen plan over held-out checkpoints (`fig2_frozen_plan_over_time.png`): planned normalized time flat, ECMP ~2.6×.

## Finding

- Frozen static-mean plan: 1.33× (FLAME) and 1.26× (OLMoE) vs ECMP 2.58× / 2.36×; speedup 1.95× / 1.87×.
- Gap to per-step oracle: static-mean +0.9% / +1.2%; static-P95 +14% / +1.4%.
- static-mean beats static-P95 for routing on FLAME (1.33× vs 1.50×); P95 is for reservation, not balancing.

## Limits

- Fluid bottleneck model, per-flow ECMP baseline; no packet-spraying, queueing or congestion control.
- 8 devices give 7 inter-device flows per device, so absolute times are inflated (oracle 1.3–1.5×), more so at large R.
- Deployment assumptions as Exp H (`common/TOOLKIT.md`).
