# Exp G: Cross-model generalisation (FLAME 290M / 721M / 1.7B)

| | |
|---|---|
| Question | Do the Exp A/B/C/E/F trace findings on FLAME-290M hold at 721M and 1.7B? |
| Data | `CMU-FLAME/FLAME-MoE-Traces`: 290M (8 MoE layers), 721M (11 layers), 1.7B (17 layers); all top-6 over 64 experts; 250,000 token-aligned rows per (checkpoint, layer); all 11 checkpoints per model (≈ 300 files for 721M and 1.7B) |
| Model | none (trace counting; deflection as in Exp F; training progress normalised to iteration / final-iteration) |
| Run | `python exp_G_cross_model_generalization/run.py` |
| Outputs | `results/cross_model_final_metrics.csv`, `results/cross_model_stabilization.csv`; `fig1_volume_stab_cross_model.png`, `fig2_volume_l1_cross_model.png`, `fig3_routing_vs_volume_cross_model.png`, `fig4_final_metrics_cross_model.png` |

## Setup

| Metric | Definition |
|---|---|
| MARGIN | % tokens with top1−top2 < 0.03 / < 0.05 (final checkpoint) |
| SKEW | mean Gini of top-6 membership load; mean top-1 primary peak (max/mean) |
| VOLUME stabilisation | layer-avg cosine and normalised L1 of the per-expert volume vector vs final, across training |
| ROUTING stabilisation | layer-avg top-1 match and top-6 overlap vs final |
| REBALANCING | mean % of primary-load excess removable by margin-gated deflection |

## Results

Final-checkpoint metrics (`cross_model_final_metrics.csv`, `fig4`):

| Metric | 290M | 721M | 1.7B |
|---|---|---|---|
| % tokens margin < 0.03 | 48.0% | 49.9% | 46.2% |
| % tokens margin < 0.05 | 62.1% | 64.6% | 60.3% |
| mean Gini (top-6 membership) | 0.026 | 0.027 | 0.021 |
| mean primary top-1 peak (max/mean) | 2.25× | 2.32× | 2.34× |
| mean % primary-excess removed (deflection) | 91.8% | 92.1% | 96.0% |

Stabilisation (`cross_model_stabilization.csv`, `fig1`, `fig2`, `fig3`): layer-avg volume cosine vs final ≥ 0.995 from the earliest checkpoint for all three models; normalised L1 shrinks the same way; routing top-6 overlap vs final climbs from ~0.48–0.58 to ~0.95 while the volume curve sits at ~1.0 throughout.

Transient at 1.7B, iter 3300 (flagged on `fig1` and `fig3`): 49.7% of primary routing on one expert (normal ~2–4%), 60/64 experts used, volume cosine vs final 0.40, routing overlap ~0.12; recovered by iter 4400; the file has the full 250k valid rows.

## Finding

- Margin < 0.03 share is 46.2–49.9% at all three sizes; primary peak 2.25–2.34×; Gini 0.021–0.027.
- Deflection removes 91.8–96.0% of primary-load excess at every size.
- Volume cosine vs final ≥ 0.995 from the earliest checkpoint; the iter 3300 collapse at 1.7B drops it to 0.40.

## Limits

- One model family and one training recipe; OLMoE (Exp D) shows different skew.
- The iter 3300 transient has not been correlated with the published training-loss curve.
- Per-expert quantities, no placement, so no per-link statement.
