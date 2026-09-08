# Experiment R: reconfigure the frozen plan at runtime
| | |
|---|---|
| Question | Install the Stage 1 plan early, keep profiling, let a guard fire on token drops and install a new reservation at a step boundary. How often does it fire, what does it buy, what does a reconfiguration cost? |
| Inputs | Checkpoints as router samples: 11 per FLAME model (290M, 721M, 1.7B), 540 to 1100 iterations apart; 5 for OLMoE, 115k to 245k apart. Token-aligned rows (250k FLAME, 205k OLMoE) cut into steps of 32k tokens (4k per rank), 7 steps per checkpoint FLAME, 6 OLMoE, seeded random order. Profile window K = 4 (290M, 721M), K = 2 (OLMoE, 1.7B, so the 3300 collapse is held out; K = 4 containing it as sensitivity). Held-out checkpoints 7 / 7 / 9 / 3. |
| Model | numpy timeline: guard observes per-step overflow and idle reserve causally and reacts one step later (rebuild folds in the install step's matrix, about +0.1% on E at the collapse fire); the planner's full-step model (fair share, anchor compute) prices the collapse layers. Paths never change (least-loaded paths depend only on sparsity). |
| Reproduce | `plane experiment --suite reconfig -o plane/output` (~3 min, 8 workers); `.venv/bin/python plane/scripts/flagship_stage3.py` (figures R1-R3 and D1, D2) |
| Outputs | `plane/output/stage3/figures/figR1_trajectories.png`, `figR2_pareto.png`, `figR3_collapse.png`; `plane/output/stage3/reconfig.csv`, `reconfig_summary.json` |
## Setup
- Policies (P95 over K checkpoints, rate-scaled to the step): freeze; freeze_robust (samples with L1 > 3× the median distance excluded; needs three or more samples); margin 1.05 / 1.10; uniform cf 1.25 (off-diagonal cells at 1.25× the window's mean off-diagonal load); periodic sliding / expanding re-plan at checkpoint boundaries; EMA every step (α = 0.2, × 1.05); oracle (P95 over the current checkpoint's own steps).
- Guards: LOO × f (leave-one-checkpoint-out threshold, f ∈ {1, 1.5, 2}), cap 2% / 5%, cosine 0.99; fire UP when mean overflow of the last 3 steps exceeds the threshold; DOWN when mean idle of the last 3 steps exceeds 10%; rebuild from the last K−1 checkpoints plus the current checkpoint's step-level P95; flagged checkpoints evicted on DOWN; 3-step refractory period.
## Results
### Stable traces: post-window means over layers, step 32k, overflow % of slots / idle % of E / model-level installs
| policy | FLAME 290M overflow / idle / events | FLAME 721M | OLMoE |
|---|---:|---:|---:|
| freeze (Stage 1) | 1.36 (0.96–1.92) / 3.5 / 0 | 1.31 (0.74–1.98) / 3.5 / 0 | 3.85 (2.4–5.3) / 7.3 / 0 |
| guard LOO ×1 | 1.36 / 3.5 / 0 | 1.25 / 3.8 / 1 | 3.79 / 7.2 / 2 (both DOWN) |
| cap 2% | 1.14 / 4.0 / 1 | 1.10 / 4.1 / 1 | 2.66 / 8.9 / 5 |
| periodic, sliding K | 1.44 / 3.2 / 7 | 1.38 / 3.4 / 7 | 4.03 / 5.9 / 3 |
| periodic, expanding | 1.21 / 3.7 / 7 | 1.15 / 3.8 / 7 | 3.37 / 7.4 / 3 |
| EMA every step | 0.34 / 5.2 / 48 | 0.32 / 5.2 / 48 | 2.01 / 7.3 / 17 |
| margin 1.05 | 0.31 / 7.1 / 0 | 0.31 / 7.2 / 0 | 2.34 / 10.3 / 0 |
| margin 1.10 | 0.05 / 11.1 / 0 | 0.06 / 11.2 / 0 | 1.44 / 13.6 / 0 |
| uniform cf 1.25 | 0.28 / 18.2 / 0 | 0.28 / 18.2 / 0 | 1.40 / 19.1 / 0 |
| oracle | 0.13 / 6.0 | 0.13 / 5.7 | 0.40 / 12.8 |
### Collapse (FLAME 1.7B, iteration 3300): window K = 2 (1100, 2200), 9 held-out checkpoints
| policy | overflow, post-window mean (min–max over layers) | idle | events | fires in checkpoints |
|---|---:|---:|---:|---:|
| freeze | 4.89 (2.7–6.7); worst step 32.6% | 6.4 | 0 | 0 / 9 |
| guard LOO ×1 | 2.58 (1.6–3.9) | 7.9 | 4 | 2–3 / 9 |
| cap 2% | 2.38 | 8.5 | 4 | 2–3 / 9 |
| periodic, sliding K | 5.12 | 10.8 | 9 | 9 / 9 |
| periodic, expanding | 4.11 | 21.8 | 9 | 9 / 9 |
| EMA every step | 2.98 | 7.6 | 62 | 9 / 9 |
| margin 1.05 / 1.10 | 3.60 / 3.10 | 9.6 / 13.3 | 0 | |
| uniform cf 1.25 | 2.85 | 20.3 | 0 | |
| oracle | 0.13 | 5.5 | | |
### Six of seventeen layers at 3300, full-step model; frozen plan 14.8% to 49.3% over across all seventeen, mean 31.8%; bound counts inter-host bytes only (two GPUs per host on a 900 Gbps scale-up link)
| layer | slots over the clean E | hot device's share of inter-host bytes (13–14% normally) | hot-NIC bound | clean E + tail carried (no reconfig) | E accommodates (reconfig) |
|---|---:|---:|---:|---:|---:|
| 02 | 29.5% | 31% | 86 ms (clean 50) | 116 ms | 125 ms |
| 03 | 14.7% | 21% | 62 ms (48) | 75 ms | 82 ms |
| 04 | 30.8% | 31% | 84 ms (47) | 110 ms | 123 ms |
| 05 | 41.5% | 32% | 88 ms (47) | 113 ms | 128 ms |
| 06 | 36.8% | 24% | 75 ms (47) | 87 ms | 98 ms |
| 07 | 28.3% | 27% | 74 ms (47) | 99 ms | 108 ms |
### Installed-reservation timetable, layer 02, dispatch + combine, guard LOO ×1; idle reserve per checkpoint 2.9% before, 28% during, 10.5% at release, 4.5–4.8% after (sliding periodic: 24% for two checkpoints, expanding: 22% for the run)
| when | what happens | installed timetable | that step's overflow / idle |
|---|---|---:|---:|
| through 2200 | frozen plan | 63 ms | 1.0% / 4.9% |
| 3300, steps 1–3 | collapse, undetected | 63 ms | 30% / 33% |
| 3300, step 4 | guard fires UP, installs a reservation that fits | 114 ms | 0.6% / 24% |
| 3300, step 7 | idle rule fires; rebuild still folds in this checkpoint's collapsed steps, nothing released | 115 ms | 0.5% / 25% |
| 4400, steps 1–2 | still carrying the collapsed reservation | 115 ms | 2.1% / 26% |
| 4400, step 3 | first clean sample; idle rule releases | 55 ms | 0.1% / 2.5% |
| 8800 | one small re-plan | 65 ms | 0.1% / 3.6% |
### Reconfiguration cost (re-plan 14–16 ms per layer and direction, sizes and rates only)
| | FLAME 290M | 721M | 1.7B | OLMoE |
|---|---:|---:|---:|---:|
| re-plan, whole model (measured wall time) | 0.26 s | 0.32 s | 0.52 s | 0.45 s |
| installs the guard makes over the run | 0 | 1 | 4 | 2 |
| stop-the-world bound (whole plan charged serially) | 0% (never fires) | 0.004% | 0.014% | < 0.001% |
| per iteration, install cost, NIC rate reprogramming (10 ms) | 1.6% | 1.2% | 0.8% | 0.6% |
| per iteration, planner occupancy if it runs every iteration | 41% | 38% | 40% | 26% |
## Finding
- Stable traces: out-of-sample guard fires 0 to 2 times; margin 1.05 or EMA cut drops to ~0.3% on FLAME, floor 0.13% / 0.40%.
- Collapse: guard detects after 3 steps, cuts that checkpoint's drops from 31.8% to 14.3% at a 1.5–1.8× timetable; the tail class is 8–13% faster than accommodating.
- A guard firing four times costs < 0.001% pipelined; every-iteration re-planning costs about 1% if NIC rates are reprogrammed.
## Limits
- The collapse is one event, duration bounded only to (0, 2200] iterations; steps re-route the same tokens; loss during the collapse and receive-buffer re-registration for a column grown 1.5–3× are outside the model.
