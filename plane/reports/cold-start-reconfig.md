# Cold start: uniform paths, then one profiled reconfiguration
| | |
|---|---|
| Question | Starting without router traces, install least-loaded Clos paths from a row-uniform matrix (known token count, top-k, eight-rank job) with a 1.25x uniform quota, profile live matrices for the established window, then replace paths and quota once (per-layer paths from the observed mean, P95 quota). What changes? |
| Inputs | flame and olmoe traces, held-out means of whole-model dispatch plus combine; ECMP averaged over 32 seeds |
| Model | The planner carries every live byte and compares paths on the 2-pod Clos; ASTRA-sim evaluates admitted sizes on an analytical 8x100 GB/s Switch and cannot distinguish path policies. Current-matrix least-loaded is a non-causal lower reference; spray is a fluid multipath reference. |
| Reproduce | `plane experiment --suite cold-start-reconfig -o plane/output` |
| Outputs | `plane/output/cold-start-reconfig/` (figures and machine-readable results) |
## Results
### Whole-model time, held-out mean
| Trace | Reconfigured | Uniform frozen | ECMP | Spray | Current-matrix LL | vs uniform | vs ECMP | planner CPU | payback vs uniform |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flame | 491.3 ms | 491.3 ms | 661.1 ms | 424.0 ms | 491.3 ms | 1.000x | 1.346x | 0.257 s | no saving |
| olmoe | 1101.8 ms | 1101.8 ms | 1449.0 ms | 979.1 ms | 1101.8 ms | 1.000x | 1.315x | 0.489 s | no saving |
### Route-table identity audit against the topology-only uniform table (56 off-diagonal routes per direction)
| Trace | Profiled entries changed | Current-matrix entries changed | Distinct path tables |
|---|---:|---:|---:|
| flame | 0 / 896 | 0 / 9856 | 1 |
| olmoe | 0 / 1792 | 0 / 8960 | 1 |
### Size-side check (a shorter admitted time means quota overflow was omitted from the ASTRA-sim arm, not a speedup)
| Trace | Quota | Overflow | Idle reserve | ASTRA-sim admitted/live cycles |
|---|---|---:|---:|---:|
| flame | uniform 1.00x | 1.54% | 1.54% | 0.974x |
| flame | uniform 1.25x | 0.00% | 20.00% | 1.000x |
| flame | profiled P95 | 0.26% | 2.46% | 0.995x |
| olmoe | uniform 1.00x | 4.89% | 4.89% | 0.917x |
| olmoe | uniform 1.25x | 0.11% | 20.09% | 0.997x |
| olmoe | profiled P95 | 1.40% | 4.90% | 0.989x |
## Finding
- Reconfiguration changes the quota table, not the path table: 0 route entries change; the advantage over ECMP (1.346x / 1.315x) comes from the deterministic balanced table.
- P95 quota releases most of the 1.25x uniform idle reserve (20.00% / 20.09% to 2.46% / 4.90%) for 0.26% / 1.40% overflow.
## Limits
- The greedy Clos router balances flow counts per choice, not bytes. No packet queues, congestion control, path-install disruption, or training quality under quota overflow.
