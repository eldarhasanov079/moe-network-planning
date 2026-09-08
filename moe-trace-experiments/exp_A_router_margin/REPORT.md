# Exp A: Router margin

| | |
|---|---|
| Question | How often is the router nearly indifferent between its top-1 and top-2 expert? |
| Data | FLAME-MoE-290M (`CMU-FLAME/FLAME-MoE-Traces`, `flame-moe-290m`), 64 experts, top-6, 8 MoE layers (`layer_02` to `layer_09`), 11 checkpoints; 500,000 tokens per layer at the final checkpoint, 250,000 tokens per layer for the training trend |
| Model | none (counting on published router scores) |
| Run | `python exp_A_router_margin/run.py` |
| Outputs | `results/*.csv` (`headline.csv`, `margin_thresholds.csv`, `margin_by_layer.csv`, `deflection_cost.csv`, `margin_trend.csv`); `fig1_margin_hist_overall.png`, `fig2_margin_hist_by_layer.png`, `fig3_threshold_share.png`, `fig4_margin_by_layer.png`, `fig5_margin_trend.png` |

## Setup

- Metric per token: `margin = (top-1 probability) - (top-2 probability)`.
- Final checkpoint (iter 5473): 500,000 tokens per layer, 8 layers, 4,000,000 routing decisions.
- Training trend: 250,000 tokens per layer for 3 layers (early/mid/late) across all 11 checkpoints.
- Deflection cost: router probability lost when a token is moved to its top-2 expert (equal to its margin).

## Results

Share of tokens below each margin threshold, all 8 layers, final checkpoint (`margin_thresholds.csv`, `fig3_threshold_share.png`).

| Margin threshold | % of tokens below it |
|---|---|
| < 0.01 | 24.9% |
| < 0.03 | 47.8% |
| < 0.05 | 62.0% |
| < 0.10 | 81.7% |

Summary (`headline.csv`, `deflection_cost.csv`, `fig1_margin_hist_overall.png`): mean margin 0.059, median 0.033; deflection cost P95 0.20, median 0.033.

Per layer (`margin_by_layer.csv`, `fig2_margin_hist_by_layer.png`, `fig4_margin_by_layer.png`): `layer_02` 93.3% of tokens below 0.03 (mean margin 0.0096); `layer_07` to `layer_09` 31–35% below 0.03.

Training trend (`margin_trend.csv`, `fig5_margin_trend.png`): `layer_02` mean margin 0.024 to 0.0098 and share below 0.03 from 79% to 93% over the 11 checkpoints; mid and late layers settle at a stable, moderate margin.

## Finding

- 47.8% of tokens have margin < 0.03; 62.0% have margin < 0.05.
- `layer_02` is least committed (93.3% below 0.03); `layer_07` to `layer_09` sit at 31–35%.
- Low-margin share does not shrink with training; `layer_02` rises from 79% to 93%.

## Limits

- Shows tokens are deflectable, not that deflectable tokens sit on congested paths (needs placement and link model).
- No deflection simulation here.
- No check that deflection preserves validation loss (needs a forward pass or training run).
