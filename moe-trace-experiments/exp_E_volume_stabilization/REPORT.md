# Exp E: Volume-level traffic stabilisation

| | |
|---|---|
| Question | Does the per-expert traffic volume distribution stabilise, and earlier than per-token routing? |
| Data | FLAME-MoE-290M (`flame-moe-290m`), 8 layers, 64 experts, top-6, 11 checkpoints, 250,000 tokens per layer per checkpoint; OLMoE-1B-7B (`allenai/analysis_olmoe`), 16 layers, 64 experts, top-8, 5 checkpoints, ~205,000 tokens per checkpoint. Reuses cached arrays from earlier experiments |
| Model | none (no placement, topology, simulator or router scores) |
| Run | `python exp_E_volume_stabilization/run.py` |
| Outputs | `results/*.csv` (`volume_stability.csv`, `volume_stability_avg.csv`, `olmoe_volume_stability.csv`, `olmoe_volume_stability_by_layer.csv`, `headline.csv`); figures `fig1` to `fig7` listed under Results |

## Setup

- Per (checkpoint, layer): `counts = bincount(indices.ravel(), minlength=num_experts)`, a length-64 vector `L`; `L[e]` = token-slots routed to expert `e`.
- Normalise: `p = L / ΣL`.
- Distance to the final checkpoint: cosine similarity `cos(p_c, p_final)` and normalised L1 `Σ_e |p_c − p_final|` (0 = identical, 2 = disjoint); computed per layer, then averaged.
- Consecutive-checkpoint distance `‖p_c − p_{c+1}‖` reported alongside.
- Per-token routing agreement (top-1 match, top-k overlap) recomputed on the same samples. Volume counts are order-independent; token alignment matters only for the routing comparison.

## Results

FLAME, layer-averaged distance from the final model (`volume_stability_avg.csv`).

| Checkpoint | cosine vs final | normalised L1 vs final |
|---|---|---|
| 540 (earliest)  | 0.9956 | 0.069 |
| 1620 | 0.9979 | 0.048 |
| 4860 | 0.9974 | 0.046 |
| 5400 | 0.9994 | 0.022 |
| 5473 | 1.000 (ref) | 0 |

- At iter 540: volume cosine 0.996, normalised L1 ≈ 0.069 (≈ 3.5% total variation); routing top-1 match 47%, top-6 overlap 58% (`fig4_volume_vs_routing.png`).
- Consecutive-checkpoint normalised L1 falls from ~0.078 to ~0.022 (`fig3_consecutive_change.png`).
- All 8 layers behave alike (`fig1_cosine_vs_final.png`, `fig2_norm_l1_vs_final.png`, `volume_stability.csv`).

OLMoE-1B-7B, same method (`olmoe_volume_stability.csv`, `fig5_olmoe_volume_stab.png`).

| OLMoE checkpoint (steps) | volume cosine vs final | volume norm L1 | routing top-8 overlap |
|---|---|---|---|
| 5,000   | 0.949 | 0.248 | 0.514 |
| 120,000 | 0.985 | 0.127 | 0.693 |
| 245,000 | 0.992 | 0.093 | 0.759 |
| 490,000 | 0.997 | 0.064 | 0.825 |
| final   | 1.000 | 0.000 | 1.000 |

- At 5k steps: volume cosine 0.95 vs routing top-8 overlap 0.51.
- Per layer (`olmoe_volume_stability_by_layer.csv`, `fig6_olmoe_l1_vs_final_by_layer.png`, `fig7_olmoe_consecutive_change_by_layer.png`): L1 vs final decreases monotonically in all 16 layers; most movement is in the 5k to 120k window; layer 0 settles slowest.

Figures: FLAME `fig1_cosine_vs_final.png`, `fig2_norm_l1_vs_final.png`, `fig3_consecutive_change.png`, `fig4_volume_vs_routing.png`; OLMoE `fig5_olmoe_volume_stab.png`, `fig6_olmoe_l1_vs_final_by_layer.png`, `fig7_olmoe_consecutive_change_by_layer.png`.

## Finding

- FLAME volume distribution matches final from the first checkpoint: cosine ≥ 0.996 throughout, normalised L1 0.069 to 0.
- Volume stabilises ahead of routing on both models: FLAME 0.996 vs 47% top-1; OLMoE 0.95 vs 0.51 overlap.
- OLMoE starts lower (0.95 vs 0.996): earlier checkpoint and more skewed load (Exp D, Gini ≈ 0.19).

## Limits

- Cosine is optimistic on near-uniform positive vectors; normalised L1 is the stricter number.
- Expert-level volume only; the source-to-destination matrix and per-link loads need placement and topology assumptions.
- Fixed probe token set isolates the routing function; input token-mix stationarity across real training batches is not measured.
