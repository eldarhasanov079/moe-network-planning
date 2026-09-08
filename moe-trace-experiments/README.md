# MoE Trace Experiments

Analysis of the published FLAME-MoE and OLMoE router traces. No model training, no GPU, no synthetic routing data. Pure Python on the traces, plus optional ASTRA-sim runs via Docker (Exp M, N, O). Both datasets are public (no auth). Loaders stream the first N rows of each file and cache them as `.npy` in `data_cache/`; first-N rows stay token-aligned across checkpoints (needed by Exp B/D).

## Data

| Source | Model | Contents | Checkpoints | Used by |
|---|---|---|---|---|
| `CMU-FLAME/FLAME-MoE-Traces` (`flame-moe-290m`) | FLAME-MoE-290M (64 experts, top-6, 8 layers) | per-token `scores` (top-6 softmax probs) + `indices` (expert ids) | 11 | Exp A, B, C, E, F, G, H |
| `allenai/analysis_olmoe` | OLMoE-1B-7B (64 experts, top-8, 16 layers) | per-token expert `ids` only (no scores) | 5 | Exp D, E, H |

## Run

```bash
cd moe-trace-experiments
python3.14 -m venv .venv               # any Python with numpy/pyarrow wheels
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python exp_A_router_margin/run.py
.venv/bin/python exp_C_expert_load_skew/run.py        # reuses Exp A cache
.venv/bin/python exp_B_routing_stabilization/run.py
.venv/bin/python exp_D_olmoe_saturation/run.py
.venv/bin/python exp_E_volume_stabilization/run.py    # reuses Exp B cache
.venv/bin/python exp_F_margin_rebalancing/run.py      # reuses Exp A cache
.venv/bin/python exp_G_cross_model_generalization/run.py   # 290M/721M/1.7B (streams the larger two)
.venv/bin/python exp_H_link_stabilization/run.py      # reuses Exp B/E cache; uses common/ netmodel toolkit
.venv/bin/python exp_I_planned_routing_sim/run.py     # flow-level All-to-All sim; both models
.venv/bin/python exp_J_first_layer_prediction/run.py  # cross-layer prediction; both models
.venv/bin/python exp_K_deflection_sim/run.py          # margin-aware deflection in-sim; FLAME only
.venv/bin/python exp_L_heterogeneous_sources/run.py   # matrix vs volume planning; both models
.venv/bin/python exp_M_astrasim_validation/run.py     # ASTRA-sim re-check of stability + deflection (needs Docker)
.venv/bin/python exp_N_fullmodel_astrasim/run.py       # full 8-layer MoE forward comm, real top-6, ASTRA-sim (needs Docker)
.venv/bin/python exp_O_online_deflection/run.py        # online (causal) vs offline deflection (needs Docker)
```

Exp M, N and O need the `astra-sim:latest` Docker image, the prebuilt analytical binary and a Python environment with the `chakra` package. Exp A to F default to the 290M model; set `FLAME_MODEL` to switch, e.g. `FLAME_MODEL=flame-moe-1.7b .venv/bin/python exp_A_router_margin/run.py`. Each experiment writes `results/*.csv` and `results/*.png` and has its own `REPORT.md`. All knobs (model, layers, checkpoints, sample sizes) live in `config.py`.

## Experiments

| Experiment | Question | Headline number(s) | Report |
|---|---|---|---|
| A Router margin | How often is the router nearly indifferent between its 1st and 2nd choice expert? | 47.8% of tokens have margin < 0.03 (62% < 0.05); persists after convergence | `exp_A_router_margin/REPORT.md` |
| B Routing stabilisation | How early does each token's routing settle to its final value? | FLAME top-6 overlap 58%→95%; expert set saturates before the exact top-1 | `exp_B_routing_stabilization/REPORT.md` |
| C Expert load skew | How uneven is expert usage? | top-6 membership load even (Gini 0.02); top-1 primary load skewed (peak 1.8–2.7×) | `exp_C_expert_load_skew/REPORT.md` |
| D OLMoE cross-check | Do stabilisation and skew replicate on a second model? | Stabilisation replicates; OLMoE more skewed (Gini 0.19) | `exp_D_olmoe_saturation/REPORT.md` |
| E Volume stabilisation | Does the per-expert traffic volume stabilise? | FLAME volume cosine ≥ 0.996 vs final throughout; OLMoE volume cosine 0.95→1.0 while routing overlap 0.51→1.0 | `exp_E_volume_stabilization/REPORT.md` |
| F Margin-gated rebalancing | How much hotspot can deflecting low-margin tokens remove, at what cost? | Deflecting 12–22% of lowest-margin tokens removes 91–98% of primary-expert hotspot excess at ~0.0005–0.005 score-loss/token | `exp_F_margin_rebalancing/REPORT.md` |
| G Cross-model generalisation | Do Exp A/B/C/E/F findings replicate on FLAME 721M and 1.7B? | Margin ~46–50% <0.03; 92–96% deflection effectiveness; router-collapse spike at 1.7B iter 3300 | `exp_G_cross_model_generalization/REPORT.md` |
| H Matrix/link stabilisation + envelope | Do the src→dst matrix and link loads stabilise under an 8-device EP model? | Matrix cosine ≥ 0.999 FLAME / 0.992 OLMoE from the first checkpoint; link max/mean ~1.09× FLAME, ~1.25× OLMoE; P95 envelope covers ~99.8% / 99.2% of link traffic at 2.2% / 5.0% waste; rank-1 residual ~3–5% | `exp_H_link_stabilization/REPORT.md` |
| I Planned-routing simulation | Does a stabilisation-gated frozen plan cut All-to-All time? | ~1.7–1.95× vs per-flow ECMP, within ~1–14% of a per-step oracle; both models | `exp_I_planned_routing_sim/REPORT.md` |
| J First-layer cross-layer prediction | Does the first MoE layer's All-to-All predict later layers' hot links? | First-layer hot-link overlap ~0.25 (chance); per-layer history 0.76; in-sim 1.28× vs oracle 1.29× | `exp_J_first_layer_prediction/REPORT.md` |
| K Deflection in-sim | Does margin-aware deflection beat random deflection and token-dropping per link? | Same relief at 9–23× lower router-score cost; trained traffic already balanced (mean peak 1.26×); FLAME only | `exp_K_deflection_sim/REPORT.md` |
| L Heterogeneous sources | When does matrix planning beat volume planning? | Residual 0.005→0.36 FLAME / 0.24 OLMoE with cosine-vs-final ≥0.9996; matrix-aware plan +10–21% (crossover at residual ≈0.1) | `exp_L_heterogeneous_sources/REPORT.md` |
| M ASTRA-sim validation | Do stability and deflection results hold on an external simulator? | Comm time flat across training (≤2.6%); deflection cuts comm time up to 6.5% Switch, 3.4% Ring, 1.1% full mesh | `exp_M_astrasim_validation/REPORT.md` |
| N Full-MoE-forward comm | What is the whole-forward All-to-All comm baseline? | 8 MoE layers, real top-6, 16 All-to-Alls, EP=8: 149.7M cycles, per-layer flat (~±3%) | `exp_N_fullmodel_astrasim/REPORT.md` |
| O Online deflection | Does a causal single-pass deflection policy match offline hindsight? | Fully balances load, order-robust (σ≈0.0001); margin 5× cheaper than margin-blind; hindsight ~4–6× more cost-efficient; online recovers ~33% at matched cost (36–54% on the 6 moderately-skewed layers); total A2A ≈ +3%, full workload ≈ +8% vs baseline on Switch at τ=0.004 | `exp_O_online_deflection/REPORT.md` |

## Layout

```
moe-trace-experiments/
  README.md
  requirements.txt
  config.py                      <- all settings (model, layers, checkpoints, sample sizes)
  common/
    flame_loader.py              <- stream+cache FLAME parquet samples
    olmoe_loader.py              <- stream+cache OLMoE jsonl samples
    plotting.py                  <- shared figure style
    TOOLKIT.md                   <- netmodel toolkit docs + assumptions
    placement.py                 <- expert->device + token->source-rank maps
    traffic_matrix.py            <- src->dst dispatch matrix from traces
    topology.py                  <- FullMeshFabric + TwoTierNodes -> per-link loads
    netmetrics.py                <- cosine/L1/overlap, link stats, envelope base/refund/overflow
    netsim.py                    <- flow-level rail-optimised All-to-All simulator (ECMP/planned/oracle)
  data_cache/                    <- cached sampled arrays (gitignored)
  figures/                       <- figures assembled from results/
  exp_A_router_margin/ .. exp_F_margin_rebalancing/   <- run.py + results/ + REPORT.md (same for every exp_*/)
  exp_G_cross_model_generalization/   <- 290M/721M/1.7B
  exp_H_link_stabilization/      <- matrix+link stabilisation + envelope (common/ toolkit)
  exp_I_planned_routing_sim/     <- flow-level All-to-All sim: planned vs ECMP/oracle
  exp_J_first_layer_prediction/  <- first-layer cross-layer prediction vs history/oracle
  exp_K_deflection_sim/          <- margin-aware deflection in-sim vs random/dropping (FLAME)
  exp_L_heterogeneous_sources/   <- matrix vs volume planning under heterogeneity
  exp_M_astrasim_validation/     <- matrix -> Chakra ET -> ASTRA-sim analytical engine in Docker
    gen_et.py                    <- matrix .npy -> per-NPU Chakra ET (needs a Python environment with the `chakra` package)
    run.py                       <- builds matrices, drives ASTRA-sim via Docker, parses comm time
  exp_N_fullmodel_astrasim/      <- full 8-layer MoE-forward comm (real top-6, dispatch+combine) via ASTRA-sim
  exp_O_online_deflection/       <- online (causal) vs offline deflection (reuses Exp M gen_et)
```
