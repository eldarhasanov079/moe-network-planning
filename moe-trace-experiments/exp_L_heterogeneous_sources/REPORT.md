# Exp L: Heterogeneous sources, matrix versus volume planning

| | |
|---|---|
| Question | If data-parallel sources were domain-specialised, would matrix-aware planning beat volume-only planning, and would the matrix stay stable across checkpoints? |
| Data | FLAME-MoE-290M and OLMoE-1B-7B traces, 8 devices, profiling-window mean tested on held-out checkpoints |
| Model | heterogeneity knob `alpha ∈ [0,1]`; flow-level model and assumptions of Exp H/I (`common/TOOLKIT.md`) |
| Run | `python exp_L_heterogeneous_sources/run.py` |
| Outputs | `results/hetero_sweep.csv`, `fig2_matrix_vs_volume_plan.png` |

## Setup
- Public traces carry no DP-rank or domain labels, so heterogeneity is modelled.
- Each token's source rank is set by routing affinity with probability `alpha` (the rank specialised to the token's reference top-1 device, targeting off-diagonal device `(s+1) mod S`) or uniformly at random with probability `1-alpha`. Token→rank map fixed across checkpoints.
- `alpha=0` reproduces the Exp H homogeneous baseline (residual ~0.005); `alpha=1` is fully domain-specialised shards.
- Volume plan: greedy routing on the rank-1 approximation `M_vol[s,d]=rowsum[s]·colsum[d]/total`. Matrix plan: greedy routing on the joint `M[s,d]`.
- Both plans frozen from the profiling-window mean; compared with ECMP and the per-step oracle on held-out checkpoints.

## Results

`hetero_sweep.csv`, FLAME-MoE-290M:

| alpha | rank-1 residual | cosine vs final | ECMP | volume plan | matrix plan | oracle | matrix gain |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.005 | 0.9998 | 2.57× | 1.38× | 1.44× | 1.35× | −4% |
| 0.25 | 0.097 | 0.9998 | 2.57× | 1.30× | 1.38× | 1.34× | −6% |
| 0.50 | 0.191 | 0.9998 | 2.52× | 1.47× | 1.34× | 1.32× | +9% |
| 0.75 | 0.278 | 0.9998 | 2.49× | 1.52× | 1.30× | 1.27× | +15% |
| 1.00 | 0.359 | 0.9997 | 2.47× | 1.61× | 1.28× | 1.27× | +21% |

OLMoE-1B-7B: residual 0.005→0.236, cosine vs final ≥0.9996, matrix gain −7% → +10.5% at alpha=1 (matrix 1.20×, volume 1.34×, oracle 1.13×). Plan comparison: `fig2_matrix_vs_volume_plan.png`.

## Finding
- Rank-1 residual rises from ~0.005 to 0.36 (FLAME) / 0.24 (OLMoE) at full specialisation.
- Cosine vs final stays ≥0.9996 at every alpha; heterogeneity does not reduce cross-checkpoint stability.
- Matrix plan beats volume plan past alpha≈0.35 (residual ≈0.1), by +10–21% at alpha=1.

## Limits
- `alpha` is modelled, not measured; the public traces sit near alpha≈0 (residual ~0.03 under contiguous sharding, Exp H).
- At alpha=0 the matrix plan is −4 to −7% worse than the volume plan; the matrix helps only once residual exceeds ~0.1.
- Flow-level, per-flow ECMP, 8-device assumptions as Exp H/I.
