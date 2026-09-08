# Exp D: OLMoE router saturation and load skew

| | |
|---|---|
| Question | Do routing stabilisation and expert load skew replicate on a second model? |
| Data | OLMoE-1B-7B (`allenai/analysis_olmoe`, C4), 16 MoE layers, 64 experts, top-8; checkpoints 5k / 120k / 245k / 490k steps + final (5 files); first 205,000 tokens (100 sequences × 2048) per checkpoint, token-aligned across files |
| Model | none (expert ids only; this dataset has no router scores) |
| Run | `python exp_D_olmoe_saturation/run.py` |
| Outputs | `results/*.csv` (`olmoe_stabilization_avg.csv`, `olmoe_load_skew.csv`); `fig1_olmoe_stabilization.png`, `fig2_olmoe_stabilization_heatmap.png`, `fig3_olmoe_load_skew.png` |

## Setup

- Stabilisation vs final: top-1 match and top-8 set overlap, per layer per checkpoint (method of Exp B).
- Load skew at final: Gini, max/mean, top-8 traffic share, per layer (method of Exp C).

## Results

Layer-averaged agreement with the final model (`olmoe_stabilization_avg.csv`, `fig1_olmoe_stabilization.png`, `fig2_olmoe_stabilization_heatmap.png`).

| Steps | top-1 match | top-8 overlap |
|---|---|---|
| 5,000   | 29.5% | 51.4% |
| 120,000 | 50.4% | 69.3% |
| 245,000 | 58.8% | 75.9% |
| 490,000 | 68.1% | 82.5% |

Load skew at the final checkpoint (`olmoe_load_skew.csv`, `fig3_olmoe_load_skew.png`), with Exp C values for comparison; even routing gives a top-8 traffic share of 12.5%.

| Metric | OLMoE (this exp) | FLAME-290M (Exp C) |
|---|---|---|
| mean Gini | 0.19 | 0.02 |
| max Gini (worst layer) | 0.28 (layer 15) | 0.038 |
| max/mean load | up to 3.2× (layer 6) | 1.18× |
| mean top-8 traffic share | 20.3% | 13.4% |

## Finding

- Routing converges monotonically on OLMoE; top-8 overlap runs ~15–20 points above top-1 match.
- Every layer follows the trend (`fig2_olmoe_stabilization_heatmap.png`); deeper layers slightly slower.
- OLMoE is more skewed than FLAME: mean Gini 0.19 vs 0.02, max/mean up to 3.2× vs 1.18×.

## Limits

- No router scores, so no margin analysis on OLMoE.
- Per-expert skew, not per-link congestion; no expert-to-device placement applied.
- No checkpoints before 5k steps are published.
