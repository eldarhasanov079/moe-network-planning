# Exp H: Matrix- and link-level stabilisation, envelope quality

| | |
|---|---|
| Question | Are the source→destination dispatch matrix and the induced link loads stable across training, and how well does a percentile envelope cover held-out traffic? |
| Data | `flame-moe-290m` (8 MoE layers, 64 experts, top-6), 250,000 tokens/layer × 11 checkpoints; `OLMoE-1B-7B` (16 MoE layers, 64 experts, top-8, 5 checkpoints), ~205,000 tokens; cached `indices` from Exp B/D/E |
| Model | modelled expert-parallel deployment (`common/TOOLKIT.md`; `common/{placement,traffic_matrix,topology,netmetrics}.py`): 8 devices (8 experts each), 8 source ranks, contiguous placement and sharding, 2 nodes of 4, per-slot forward-dispatch top-k membership traffic; link loads, not times |
| Run | `python exp_H_link_stabilization/run.py` |
| Outputs | `results/matrix_stability.csv`, `matrix_stability_avg.csv`, `link_stability.csv`, `link_stability_avg.csv`, `envelope_quality.csv`, `placement_robustness.csv`, `headline.csv` (OLMoE: `olmoe_*.csv`); `fig1`–`fig6` (OLMoE: `olmoe_fig*`) |

## Setup

| Item | Assumption |
|---|---|
| Placement (A1–A2) | 64 experts spread evenly over 8 devices, static across training |
| Sharding (A3) | 8 source ranks from a synthetic map; default `contiguous`, swept `strided` |
| Traffic (B1–B5) | forward-dispatch All-to-All only (combine = transpose); top-k membership; token-slots; diagonal carries no link load |
| Topology (C0–C3) | `FullMeshFabric` (directed link load = off-diagonal entry); `TwoTierNodes` (one uplink/downlink per node, single spine) |
| Envelope (D1) | per-entry upper bound; cost = waste (refund) + uncovered bytes (overflow) |
| Window | profile on checkpoints 1080–3240; apply to held-out 3780–5473 |

Method, per (layer, checkpoint): `indices` → `M[src,dst]` → link loads on both topologies. Metrics: cosine / normalised L1 of the normalised matrix vs final and vs consecutive checkpoints; rank-1 residual (distance of `M` from the source-marginal × device-popularity outer product); full-mesh link-load max/mean and P95, 2-tier uplink skew, hot-link overlap vs final; mean/P90/P95/P99/worst envelopes scored by overflow ratio, fraction of flows overflowing and waste.

## Results

Matrix level, FLAME (`matrix_stability_avg.csv`):

| Checkpoint | matrix cosine vs final | norm-L1 vs final | rank-1 residual |
|---|---|---|---|
| 540 (earliest) | **0.9991** | 0.033 | 0.031 |
| 2160 | 0.9997 | 0.019 | 0.029 |
| 5400 | 0.9999 | 0.007 | 0.029 |
| 5473 | 1.000 (ref) | 0 | 0.029 |

Link level, FLAME (`link_stability_avg.csv`):

| Checkpoint | full-mesh link max/mean | 2-tier uplink max/mean | hot-link overlap vs final |
|---|---|---|---|
| 540 | 1.11× | 1.02× | 0.58 |
| 2700 | 1.10× | 1.01× | 0.67 |
| 5473 | 1.09× | 1.01× | 1.00 |

Envelope, FLAME, link level (`envelope_quality.csv`):

| Policy | link overflow | flows overflowing | link waste (refund) |
|---|---|---|---|
| mean | 0.81% | 50.7% | 0.80% |
| P90 | 0.22% | 20.6% | 2.01% |
| **P95** | **0.17%** | 17.4% | **2.22%** |
| P99 | 0.15% | 14.9% | 2.40% |
| worst-case | 0.14% | 14.5% | 2.45% |

Robustness (`placement_robustness.csv`), contiguous / roundrobin / random placement and strided sharding: matrix cosine 0.999, rank-1 residual ~0.03, full-mesh max/mean ~1.08–1.09×.

Cross-architecture (`olmoe_*.csv`, `olmoe_fig*`; OLMoE Gini ≈ 0.19 vs FLAME 0.02 in Exp D):

| Metric | FLAME-290M | OLMoE-1B-7B |
|---|---|---|
| matrix cosine vs final @ first ckpt | 0.9991 | **0.9924** |
| full-mesh link max/mean (final) | 1.09× | **1.25×** |
| rank-1 residual (contiguous) | ~0.03 | ~0.04→**0.05** (grows over training) |
| P95 link overflow / waste | 0.2% / 2.2% | **0.8% / 5.0%** |
| worst-case waste | 2.5% | 5.4% |

OLMoE: hot-link overlap rises 0.53→1.0; rank-1 residual falls to ~0.005 under strided sharding (`olmoe_placement_robustness.csv`).

Figures: `fig1_matrix_cosine_vs_final.png`, `fig2_predictability_ladder.png`, `fig3_link_load_skew.png`, `fig4_hot_link_overlap.png`, `fig5_envelope_tradeoff.png`, `fig6_placement_robustness.png`; OLMoE: same six with the `olmoe_` prefix.

## Finding

- Matrix cosine vs final ≥ 0.999 from iter 540 (FLAME); 0.9924 at the first OLMoE checkpoint.
- Full-mesh link max/mean 1.09× (FLAME) and 1.25× (OLMoE), flat across training.
- P95 envelope: 0.17% overflow at 2.22% waste (FLAME); 0.8% / 5.0% (OLMoE).

## Limits

- Rank-1 residual ~0.03 on FLAME: the matrix is ≈97% explained by source-marginal × device-popularity, so matrix predictability restates volume predictability under this source model.
- Loads, not times: no queueing, bandwidth or contention; no iteration-time claim.
- Checkpoints are training snapshots, not consecutive iterations (drift proxy); FLAME 721M/1.7B not run.
