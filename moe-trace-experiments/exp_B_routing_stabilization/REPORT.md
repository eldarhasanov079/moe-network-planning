# Exp B: Routing stabilisation over training

| | |
|---|---|
| Question | How early does each token's routing settle to its final value? |
| Data | FLAME-MoE-290M (`flame-moe-290m`), 64 experts, top-6, 8 layers, 11 checkpoints (iter 540 to 5473); 250,000 tokens per layer per checkpoint, 88 (checkpoint, layer) files, 22M routing decisions |
| Model | none (per-token comparison against the final checkpoint; rows are token-aligned across checkpoints) |
| Run | `python exp_B_routing_stabilization/run.py` |
| Outputs | `results/*.csv` (`stabilization.csv`, `stabilization_layer_avg.csv`, `headline.csv`); `fig1_top1_match_by_layer.png`, `fig2_top6_overlap_by_layer.png`, `fig3_stabilization_avg.png` |

## Setup

- Reference: the same token's routing at the final checkpoint (iter 5473).
- top-1 match rate: % of tokens whose top-1 expert equals the final top-1.
- top-6 overlap: mean fraction of the 6 chosen experts shared with the final 6.
- The 11 checkpoints are FLAME's capture window; iter 540 is the earliest captured point, not the start of pre-training.

## Results

Layer-averaged agreement with the final checkpoint (`stabilization_layer_avg.csv`, `fig3_stabilization_avg.png`).

| Checkpoint | top-1 match | top-6 overlap |
|---|---|---|
| 540  | 47.2% | 58.1% |
| 1080 | 57.4% | 67.4% |
| 2160 | 67.3% | 75.1% |
| 3240 | 73.5% | 79.8% |
| 4320 | 78.3% | 83.4% |
| 4860 | 80.9% | 85.6% |
| 5400 | 93.6% | 95.0% |
| 5473 | 100%  | 100% (by definition) |

Per-layer curves: `fig1_top1_match_by_layer.png`, `fig2_top6_overlap_by_layer.png` (data in `stabilization.csv`).

## Finding

- Routing converges monotonically: top-6 overlap 58% to 95%, top-1 match 47% to 94% over the window.
- top-6 overlap stays ~10 points above top-1 match; the expert set settles before the exact top-1.
- All layers converge, later layers slightly slower; largest step between iter 4860 and 5400.

## Limits

- Routing level only; traffic volume, matrix and link stabilisation are not measured here.
- Checkpoints before iter 540 are not published, so the earliest stabilisation point is not observable.
