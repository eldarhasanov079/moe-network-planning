# Exp J: First-layer cross-layer prediction

| | |
|---|---|
| Question | Does the first MoE layer's dispatch matrix in an iteration predict later layers' matrices and hot links better than that layer's own training history? |
| Data | FLAME and OLMoE traces, windows and checkpoints as Exp H/I; one forward routing snapshot per layer (no phase-resolved traces) |
| Model | flow-level simulator `common/netsim.py`; deployment as Exp H (`common/TOOLKIT.md`); hot-link overlap on the hottest 25% of links |
| Run | `python exp_J_first_layer_prediction/run.py` |
| Outputs | `results/prediction_quality.csv`, `prediction_quality_avg.csv`, `sim_summary.csv`, `sim_detail.csv`, `headline.csv` (OLMoE: `olmoe_*`); `fig1_prediction_quality.png`, `fig2_sim_plan_source.png`, `fig3_cosine_vs_distance.png` (OLMoE: `olmoe_fig*`) |

## Setup

- Target: dispatch matrix of layer L at held-out checkpoint c. Predictors: first-layer identity (layer 0 matrix at checkpoint c, within-iteration); first-layer transpose; history mean (profiling-window mean of layer L, across-iteration, the Exp H/I frozen plan); uniform (lower bound). Scores: cosine, normalised L1, hot-link overlap. Simulator: route each later layer with a plan from the trigger, from history, ECMP, or its own oracle.

## Results

`prediction_quality_avg.csv` (FLAME / OLMoE):

| Predictor | cosine | hot-link overlap |
|---|---|---|
| first-layer identity | 0.998 / 0.988 | **0.23 / 0.26** |
| first-layer transpose | 0.998 / 0.987 | 0.26 / 0.26 |
| **history mean** | 0.9998 / 0.998 | **0.76 / 0.76** |
| uniform (lower bound) | 0.999 / 0.992 | 0.23 / 0.23 |

Simulator normalized All-to-All time (`sim_summary.csv`):

| Plan source | FLAME | OLMoE |
|---|---|---|
| ECMP | 2.59× | 2.36× |
| first-layer plan | 1.56× | 1.49× |
| **history plan** | **1.28×** | **1.25×** |
| oracle | 1.29× | 1.24× |

## Finding

- First-layer hot-link overlap 0.23 / 0.26, at chance (0.25 for the hottest 25%); history mean 0.76 / 0.76.
- Uniform prediction scores cosine 0.999 / 0.992, so cosine does not discriminate on these matrices.
- History plan 1.28× / 1.25× matches oracle 1.29× / 1.24×; first-layer plan 1.56× / 1.49×.

## Limits

- Cross-layer form only; dispatch→combine phase prediction needs phase-resolved traces, and combine is the transpose by construction.
- Same flow-level model and deployment assumptions as Exp H/I.
