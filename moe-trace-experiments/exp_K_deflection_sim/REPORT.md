# Exp K: Margin-aware deflection in the All-to-All simulator

| | |
|---|---|
| Question | Does margin-aware deflection relieve per-device hotspots at lower router-score cost than random deflection or token-dropping? |
| Data | FLAME-MoE-290M, final checkpoint, all 8 layers, 8 devices, 250k tokens/layer |
| Model | top-1 dispatch matrix `M[s,d]` (tokens from source rank s whose top-1 expert is on device d); collective_time ∝ peak device load = max/mean; contiguous placement |
| Run | `python exp_K_deflection_sim/run.py` |
| Outputs | `results/deflection_summary.csv`, `fig1_pareto_scoreloss.png`, `fig3_scoreloss_per_layer.png` |

## Setup
| Policy | Move | Token | Cost |
|---|---|---|---|
| margin-aware | lowest `score(top1)-score(top2)` margin token off the busiest device to its top-2 device | kept | margin |
| random deflection | same move, margin-blind | kept | margin |
| token-dropping | drop the cheapest token | lost | full top-1 score |

## Results
`deflection_summary.csv`, relief target peak ≤ 1.05, most-skewed layer (`layer_02`, initial peak 1.46×):

| Policy | tokens affected | router-score lost | vs margin-aware |
|---|---|---|---|
| margin-aware | 7.5% | 17.3 | 1× |
| random deflection | 7.7% | 209.5 | 12× more |
| token-dropping | 6.5% (tokens lost) | 365.0 | 21× more |

All 8 layers (`fig3_scoreloss_per_layer.png`): margin-aware reaches the same relief at 9–23× lower score cost than either baseline. Pareto (`fig1_pareto_scoreloss.png`): margin-aware flattens the peak at score-loss ~30, random ~300, dropping ~450.

## Finding
- Margin-aware deflection reaches the relief target at 12× (random) and 21× (dropping) lower score cost on `layer_02`; 9–23× across all 8 layers.
- Tokens moved (~7%) is set by the imbalance; cost depends on which tokens are moved.

## Limits
- Trained traffic is already near-balanced (mean initial peak 1.26×), so absolute headroom is modest.
- Top-1 primary-assignment model; static single-step shaping, no queueing dynamics.
- FLAME only: OLMoE traces carry no router scores, so no margins.
