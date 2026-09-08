# Exp F: Margin-gated load rebalancing

| | |
|---|---|
| Question | How much primary-expert hotspot can top1→top2 deflection of low-margin tokens remove, and at what router-score cost? |
| Data | `flame-moe-290m`, final checkpoint, 8 layers, 64 experts, top-6, 500,000 tokens per layer (cached `scores` + `indices` from Experiment A) |
| Model | `load(e)` = tokens whose top-1 expert is `e`; a deflection moves one token to its top-2 expert; cost = `margin = scores[0] − scores[1]`; peak is per expert, no placement |
| Run | `python exp_F_margin_rebalancing/run.py` |
| Outputs | `results/rebalancing_summary.csv`, `results/rebalancing_budgets.csv`, `results/headline.csv`; `fig1_pareto_frac_moved.png`, `fig2_pareto_score_loss.png`, `fig3_initial_vs_floor.png` |

## Setup

- Greedy and deterministic: deflect the cheapest helpful token off the current hottest expert; each token moves at most once.
- Record (cumulative cost, peak) after each move; this traces the Pareto frontier.
- Peak = max/mean of primary (top-1) load per layer.
- Primary load is more skewed than the top-6 membership load of Experiment C (peak 1.06–1.18×).

## Results

Per layer, initial vs. best-achievable primary-load peak (`rebalancing_summary.csv`, `rebalancing_budgets.csv`):

| Layer | initial peak | floor peak | % excess removed | % tokens moved | avg score loss / token |
|---|---|---|---|---|---|
| layer_02 | 2.36 | **1.07** | 95% | 21.8% | 0.0005 |
| layer_03 | 1.98 | **1.02** | 98% | 15.7% | 0.0010 |
| layer_04 | 2.70 | **1.07** | 96% | 13.9% | 0.0023 |
| layer_05 | 2.27 | **1.04** | 97% | 15.0% | 0.0026 |
| layer_06 | 2.27 | **1.08** | 94% | 11.7% | 0.0023 |
| layer_07 | 2.14 | **1.04** | 97% | 15.9% | 0.0047 |
| layer_08 | 2.25 | **1.11** | 91% | 13.0% | 0.0030 |
| layer_09 | 1.84 | **1.28** | 67% | 3.5% | 0.0010 |

Figures:
- `fig1_pareto_frac_moved.png`: peak vs % tokens deflected.
- `fig2_pareto_score_loss.png`: peak vs average score-loss budget.
- `fig3_initial_vs_floor.png`: initial vs best-achievable peak per layer.

## Finding

- For 7 of 8 layers, peak falls from ~2.0–2.7× to 1.02–1.11× mean, removing 91–98% of the excess.
- Cost: 12–22% of tokens moved, average score loss ~0.0005–0.005 per token.
- `layer_09` floors at 1.28×; moving 3.5% of tokens removes 67% of excess at 0.001/token.

## Limits

- Peak is per expert, not per device or link; no placement assumed.
- OLMoE traces store expert ids but no router scores, so the cost axis cannot be computed there.
- Single-step deflection to the token's own top-2 expert only; no forward pass to check validation loss.
