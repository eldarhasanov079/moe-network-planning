# Clos routing baselines
| | |
|---|---|
| Question | On the admitted last-checkpoint graphs of claim v1 (`sent = min(M, E)`), how do planned least-loaded Clos paths compare with ECMP and spray? |
| Inputs | flame and olmoe admitted last-checkpoint graphs, policies uniform, mean, p95, worst |
| Model | The planner only, no ASTRA-sim. least-loaded (plan): one Clos path per pair, occupancy-aware. ECMP: same dest, hashed path, 32 seeds. spray: split each pair across every equal-cost path (fluid RPS / REPS-family). DCQCN is a rate loop on top of ECMP/spray, not modelled. |
| Reproduce | `plane experiment --suite clos-baselines -o plane/output` |
| Outputs | `plane/output/clos-baselines/` |
## Results
### Whole-model time, admitted last checkpoint
| Trace | Policy | planned (ms) | ECMP (ms) | spray (ms) | vs ECMP | vs spray |
|---|---|---:|---:|---:|---:|---:|
| flame | uniform | 499.9 | 661.5±22.4 | 430.7 | 1.32× | 0.86× |
| flame | mean | 491.8 | 655.6±22.5 | 409.5 | 1.33× | 0.83× |
| flame | p95 | 487.1 | 661.0±21.7 | 429.9 | 1.36× | 0.88× |
| flame | worst | 496.8 | 659.3±22.5 | 430.1 | 1.33× | 0.87× |
| olmoe | uniform | 1053.5 | 1369.5±61.8 | 930.2 | 1.30× | 0.88× |
| olmoe | mean | 1056.4 | 1404.2±51.7 | 926.9 | 1.33× | 0.88× |
| olmoe | p95 | 1056.1 | 1426.9±54.2 | 952.9 | 1.35× | 0.90× |
| olmoe | worst | 1061.7 | 1428.6±52.6 | 954.4 | 1.35× | 0.90× |
## Finding
- Plan beats ECMP by ~1.35× and trails spray by ~0.88–0.90×; the 10–12% gap is the single-path, no-reorder cost.
## Limits
- Spray is fluid multipath; DCQCN at packet level is not modelled.
