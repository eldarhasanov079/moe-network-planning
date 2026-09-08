# Exp C: Expert load skew

| | |
|---|---|
| Question | How uneven is expert usage across the 64 experts? |
| Data | FLAME-MoE-290M (`flame-moe-290m`), 64 experts, top-6, 8 layers; 500,000 tokens per layer at the final checkpoint (3M expert selections per layer); 250,000 tokens per checkpoint for `layer_06` across all 11 checkpoints. Reuses the cached `indices` arrays from Exp A |
| Model | none (top-6 membership counts per expert) |
| Run | `python exp_C_expert_load_skew/run.py` |
| Outputs | `results/*.csv` (`skew_metrics_by_layer.csv`, `expert_load_shares.csv`, `skew_evolution.csv`, `headline.csv`); `fig1_load_distribution_by_layer.png`, `fig2_sorted_load_curves.png`, `fig3_skew_metrics_by_layer.png`, `fig4_skew_evolution.png` |

## Setup

| Metric | Definition | Even-routing value |
|---|---|---|
| max/mean load | busiest expert's count over the mean count | 1.0 |
| coefficient of variation | std/mean of per-expert counts | |
| Gini | 0 = even, 1 = one expert takes everything | 0.0 |
| top-8 traffic share | share of all selections captured by the 8 busiest experts | 12.5% |

## Results

Final checkpoint, per layer (`skew_metrics_by_layer.csv`, `expert_load_shares.csv`, `fig3_skew_metrics_by_layer.png`).

| Metric | Range across layers | Even-routing value |
|---|---|---|
| max / mean load | 1.06 – 1.18 | 1.0 |
| Gini coefficient | 0.016 – 0.038 | 0.0 |
| top-8 traffic share | 13.1% – 14.0% | 12.5% |
| experts used | 64 / 64 (all) | 64 |

- Busiest expert (`layer_09`): 1.18× the mean. min/mean ≈ 0.8–0.94.
- `layer_02` to `layer_05`: Gini ≈ 0.016–0.029; `layer_09`: Gini 0.038.

Training evolution, `layer_06` (`skew_evolution.csv`, `fig4_skew_evolution.png`): first checkpoint max/mean 1.18, Gini 0.045; skew decreases with training and settles near Gini ≈ 0.02.

Figures: `fig1_load_distribution_by_layer.png` (sorted load bars per layer vs even line), `fig2_sorted_load_curves.png` (all layers overlaid), `fig3_skew_metrics_by_layer.png` (max/mean and Gini by layer), `fig4_skew_evolution.png` (skew vs checkpoint).

## Finding

- top-6 membership load is close to even: Gini 0.016–0.038, max/mean 1.06–1.18, all 64 experts used.
- Skew rises slightly with depth; `layer_09` is the most skewed (Gini 0.038).
- Skew is highest at the first checkpoint (Gini 0.045) and falls to ≈ 0.02 with training.

## Limits

- Counts experts, not devices or links; per-link skew under a placement is not measured here.
- Long-run averages only; per-batch or sliding-window bursts are not measured.
- Whether low-margin tokens (Exp A) concentrate on the busier experts is not measured here.
