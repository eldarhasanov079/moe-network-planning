# Claim v1: frozen early plan, no reconfiguration
| | |
|---|---|
| Question | After a short early window, can a per-layer frozen reservation `E` be kept for the rest of training with later steps sending `min(M, E)` (overflow dropped, no deflection, no reconfiguration), and does Clos least-loaded beat Clos ECMP on the same admitted graph? |
| Inputs | FLAME-MoE-290M: 8 MoE layers, 250k tokens, 11 ckpts, freeze after 4 (profile 540–2160, held-out 2700–5473). OLMoE-1B-7B: 16 layers, 205k tokens, 5 ckpts, freeze after 2 (held-out last 3). Policies uniform, mean, P95, worst. |
| Model | ASTRA-sim analytical congestion-aware Switch 8×100 GB/s, 500 ns, whole model = sum of per-layer dispatch+combine, sizes only, no Clos paths; floor = last held-out live `M` on a 1e6 GB/s Switch. Always-replan = causal `E_t = collapse(M_0 … M_{t-1})`. Clos vs ECMP: the planner's 2-pod × 2-leaf × 2-spine Clos, least-loaded vs 64-seed oblivious per-flow hash, same admitted graphs, whole-model time = sum over layers of `iteration_time_s`. |
| Reproduce | `plane experiment --suite claim-v1 -o plane/output` |
| Outputs | `plane/output/claim-v1/summary.json`, `astrasim_jobs.csv`, `admitted-live/{overflow,token_drops}.csv`, `freeze-sweep/overflow.csv`, `clos-ecmp/plans.csv`; figures `admitted-live/figures/{flame,olmoe}_*.png`, `freeze-sweep/figures/{flame,olmoe}_freeze_vs_replan_*.png`, `clos-ecmp/figures/{flame,olmoe}_planned_vs_ecmp.png` |
## Setup
- Admitted live: time the later checkpoint's `min(M, E)` on the Switch and compare with sending live `M` unclipped; token-order drops reported separately.
- Admitted can be slightly faster than live because fewer bytes are sent, not because the reservation added bandwidth.
## Results
### Admitted-live ASTRA-sim, FLAME (live avg 150.33M cycles, congestion-free floor 0.13M)
| Policy | Admitted cycles | vs live | overflow | waste | token drop (≥1 slot) | slot drop |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 150.33M | +0.00% | 12.48% | 12.48% | 56.91% | 12.48% |
| mean | 148.45M | −1.25% | 0.97% | 0.97% | 2.81% | 0.97% |
| p95 | 149.56M | −0.51% | 0.26% | 2.46% | 1.02% | 0.26% |
| worst | 149.79M | −0.36% | 0.23% | 2.68% | 0.92% | 0.23% |
### Admitted-live ASTRA-sim, OLMoE (live avg 340.44M cycles, floor 0.26M)
| Policy | Admitted cycles | vs live | overflow | waste | token drop | slot drop |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 332.73M | −2.27% | 13.20% | 13.20% | 69.77% | 13.20% |
| mean | 331.52M | −2.62% | 2.98% | 2.98% | 10.30% | 2.98% |
| p95 | 336.59M | −1.13% | 1.40% | 4.89% | 5.49% | 1.40% |
| worst | 336.96M | −1.02% | 1.26% | 5.13% | 5.04% | 1.26% |
### Default freeze vs always-replan, held-out average
| Source | Policy | Freeze cycles | Replan cycles | gap |
|---|---|---:|---:|---:|
| FLAME | mean | 148.45M | 148.40M | +0.03% |
| FLAME | p95 | 149.56M | 149.85M | −0.20% |
| FLAME | worst | 149.79M | 149.87M | −0.06% |
| OLMoE | mean | 331.52M | 332.66M | −0.35% |
| OLMoE | p95 | 336.59M | 336.45M | +0.04% |
| OLMoE | worst | 336.96M | 336.74M | +0.06% |
### Freeze after K, FLAME admitted, million cycles (default K=4; OLMoE grid K=1…4, K=2 within 0.4% of always-replan)
| K | uniform | mean | p95 | worst |
|---|---:|---:|---:|---:|
| 1 | 150.35 | 147.97 | 147.97 | 147.97 |
| 2 | 150.42 | 148.50 | 149.12 | 149.07 |
| 3 | 150.36 | 148.79 | 149.52 | 149.61 |
| 4 | 150.33 | 148.45 | 149.56 | 149.79 |
| 6 | 150.23 | 148.05 | 149.99 | 149.83 |
| 10 | 149.68 | 147.86 | 149.70 | 149.72 |
FLAME mean overflow 1.66% at K=1 and 0.97% at K=4; P95 at K=4 is 0.26%.
### Clos planned vs ECMP, admitted last held-out checkpoint, 64 ECMP seeds
| Source | Policy | Planned (ms) | ECMP mean ± std (ms) | speedup |
|---|---|---:|---:|---:|
| FLAME | uniform | 499.9 | 660.3 ± 20.9 | 1.32× |
| FLAME | mean | 491.8 | 654.2 ± 20.9 | 1.33× |
| FLAME | p95 | 487.1 | 658.0 ± 20.6 | 1.35× |
| FLAME | worst | 496.8 | 657.1 ± 21.9 | 1.32× |
| OLMoE | uniform | 1053.5 | 1362.5 ± 58.4 | 1.29× |
| OLMoE | mean | 1056.4 | 1396.1 ± 48.5 | 1.32× |
| OLMoE | p95 | 1056.1 | 1418.6 ± 49.8 | 1.34× |
| OLMoE | worst | 1061.7 | 1420.3 ± 49.8 | 1.34× |
### Clos planned vs ECMP, frozen reservation `E` (the installed plan)
| Source | Policy | Planned (ms) | ECMP mean (ms) | speedup |
|---|---|---:|---:|---:|
| FLAME | uniform | 421.3 | 630.9 | 1.50× |
| FLAME | mean | 484.2 | 660.4 | 1.36× |
| FLAME | p95 | 511.2 | 675.7 | 1.32× |
| OLMoE | uniform | 921.2 | 1379.5 | 1.50× |
| OLMoE | mean | 1116.4 | 1443.5 | 1.29× |
| OLMoE | p95 | 1150.5 | 1497.8 | 1.30× |
## Finding
- P95 frozen after 2–4 checkpoints: within 1.2% of live Switch time, 0.3–1.4% slot overflow, 1–5.5% of tokens lose a dest.
- Always-replan is within 0.35% of the default freeze; K buys overflow (1.66% to 0.97% for FLAME mean), not latency.
- Uniform keeps Switch time but drops a dest for 56.91% / 69.77% of tokens; the matrix shape is needed.
- Clos least-loaded beats per-flow ECMP by 1.29–1.35×; below Exp I's 1.7–1.95× on a 4-rail fabric (2-way choice here).
## Limits
- No packet-level queues, congestion control or packet-spraying ECMP; ASTRA-sim is fluid Switch and does not consume the planner's Clos paths.
- No training loss for dropped slots; no runtime deflection, refunds or mid-training reconfiguration.
- Not larger than 8-GPU EP; FLAME 721M / 1.7B not run on this chain.
