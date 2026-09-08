# Exp O: Online margin deflection

| | |
|---|---|
| Question | Does a causal, single-pass margin deflection policy using only running load counters match offline (hindsight) deflection? |
| Data | FLAME-MoE-290M actives, final checkpoint (5473), all 8 MoE layers, 250k tokens/layer, per-token top-6 `scores`/`indices` |
| Model | top-1 primary-traffic lens (as Exp F/K/M); EP=8, contiguous placement (`flame-moe-290m.sh`, `EXPERT_MODEL_PARALLEL_SIZE=8`); peak device load = max/mean; quality proxied by router-score loss (sum of deflected margins) |
| Run | `python exp_O_online_deflection/run.py`; `python exp_O_online_deflection/latency_by_margin.py`; `python exp_O_online_deflection/figures_global_local.py` |
| Outputs | `online_vs_offline_by_layer.csv`, `online_vs_offline_commtime.csv` (long-form, `topology` column), `latency_by_margin_raw.csv`, `peak_by_margin.csv`, `order_sensitivity.csv`, `verdict.csv`; figures listed below |

## Setup

| Policy | Deployable | Rule |
|--------|-------------|------|
| raw | n/a | no deflection |
| offline | no (hindsight upper bound) | greedy: repeatedly move lowest-margin token off the global busiest device to its top-2, until peak ≤ 1.05 |
| online_global | yes | one causal pass: deflect token to top-2 if `margin ≤ τ` and top-2 device's running load < top-1's (shared counter) |
| online_local | yes (zero coordination) | same, each source rank uses only its own per-device counters |
| online_random | ablation | causal, margin-blind (deflect random eligible token w.p. p) |

- Online rule: power-of-two-choices between the token's top-1 and top-2 device, gated by router margin.
- Arrival order = trace row order; also 3 real-token reshufflings.
- ASTRA-sim (analytical, congestion-aware) on Switch / Ring / FullyConnected, identical 100 GB/s, 500 ns links.

## Results

| Quantity | Value |
|---|---|
| Raw peak device load, per layer | 1.18–1.46× |
| Peak after online deflection | 1.000× on every layer |
| Router-score cost to reach peak 1.0, showcase layer | online ~100, online_random ~500 (~5× lower for margin-aware) |
| Online cost vs offline at the 1.05 target | ~4–6× more |
| Offline benefit captured at matched (offline) cost | ~33% (global) / 29% (local) mean; 36–54% on L04–L09; ~0% on L02–L03 |
| Online peak spread over 3 reshufflings + trace order | σ ≈ 0.0001 |

Figures: `fig1_pareto_showcase.png` (peak vs router-score cost, showcase layer), `fig2_benefit_by_layer.png` (% of offline benefit at matched cost, per layer).

ASTRA-sim mean All-to-All latency reduction vs raw (online at its relief operating point, offline traced to full balance; `fig3_astrasim_reduction.png`, Switch per layer):

| topology | offline (full balance) | online_global | online_local |
|----------|------------------------|---------------|--------------|
| Switch (primary) | 3.1% | 3.4% | 2.3% |
| Ring | 2.1% | 0.8% | 0.6% |
| FullyConnected | 1.9% | 2.3% | 2.5% |

Fixed-margin whole-model totals (`latency_by_margin.py`, τ = 0.001 / 0.002 / 0.004, summed over 8 layers, raw / online_local / online_global bars, offline at full balance (`offline_full_chosen`) as a dashed lower-bound tick). Savings vs raw on Switch at τ=0.004:

| Figure | Quantity | online | offline lower bound |
|---|---|---|---|
| `fig_latency_tau_t00{10,20,40}.png` | total All-to-All time, 16 collectives (dispatch + combine) | ≈ +3% | ≈ +4% |
| `fig_fulllatency_tau_t00{10,20,40}.png` | full workload latency, Σ e2e_wall (max-rank compute + both a2as, compute model as Exp M) | +8% | +16% |

Global-vs-local figures (`figures_global_local.py`; margin grid `TAUS_FINE`, 10 values from 0 to 0.01; offline traced to full balance 1.0):
- `fig4_margin_sweep_global_local.png`: peak falls from 1.46 to 1.0 across τ≈0.002–0.005; cost keeps climbing after balance; global≈local.
- `fig5_heatmap_layers_margin.png`: peak over layers × margin (0–0.01), global and local; local lags global by ~one τ step.
- `fig6a_cost_frontier.png`: cost vs peak reached, showcase layer; offline lowest at every balance level. `fig6b_cost_across_layers.png`: cost to reach peak ≤ 1.15 per layer, offline ~3–10× cheaper than online.

## Finding
- Causal online deflection drives peak device load from 1.18–1.46× to 1.000× on every layer; order spread σ ≈ 0.0001.
- Margin gating cuts online cost ~5× versus margin-blind online; offline remains ~3–10× cheaper than online at equal balance.
- ASTRA-sim comm-time reduction is ~1–3%; full-layer latency saving on Switch at τ=0.004 is +8% online, +16% offline bound.

## Limits
- Top-1 lens; a top-6 set-replacement online policy is not evaluated.
- Quality proxied by router-score loss, not validation loss; `online_global` assumes periodic load broadcast, `online_local` needs none.
- Fluid analytical model: All-to-All time follows aggregate per-link volume, so per-layer comm-time change can be negative (e.g. L06 online); compute dominates comm ~100:1 in the full-latency figures.
