# Prefix vs full-file sanity
| | |
|---|---|
| Question | Do the 205k (OLMoE) / 250k (FLAME) prefixes give the same overflow and token-drop as the whole files? Gate: moved if \|Δ overflow\| > 0.3 pp or \|Δ token-drop\| > 1.0 pp. |
| Inputs | OLMoE: 205k prefix vs every sequence in `allenai/analysis_olmoe` (205,000 vs 761,856 tokens), freeze after 2, all 16 layers, held-out mean. FLAME: `layer_02`, ckpt 5473, E = Stage 1 freeze (first 4 ckpts × 250k), later 250k blocks of the same file scored against that E (208 blocks, 52,428,800 tokens). |
| Model | Overflow and token-drop counts only; no ASTRA-sim, no Clos. |
| Reproduce | `.venv/bin/python plane/scripts/full_trace_sanity.py` |
| Outputs | tables below |
## Results
### OLMoE, 205k prefix vs whole JSONL
| Policy | overflow 205k | overflow full | Δ pp | tokens≥1 dest 205k | full | Δ pp |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 13.20% | 13.16% | -0.04 | 69.77% | 69.84% | +0.08 |
| mean | 2.98% | 2.80% | -0.18 | 10.30% | 9.70% | -0.60 |
| p95 | 1.40% | 1.31% | -0.09 | 5.49% | 5.14% | -0.35 |
| worst | 1.26% | 1.18% | -0.08 | 5.04% | 4.71% | -0.33 |
### FLAME, layer_02 ckpt 5473, later 250k blocks vs the frozen E; dest-share cosine vs prefix 0.9999
| Policy | overflow prefix | overflow later mean [min, max] | Δ pp | token-drop prefix | last block | Δ pp |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 12.46% | 12.51% [12.24, 12.89] | +0.05 | 56.84% | 56.66% | -0.18 |
| mean | 0.93% | 1.37% [1.05, 1.73] | +0.43 | 3.18% | 4.39% | +1.21 |
| p95 | 0.14% | 0.33% [0.08, 0.65] | +0.19 | 0.64% | 1.46% | +0.81 |
| worst | 0.12% | 0.30% [0.05, 0.61] | +0.18 | 0.56% | 1.30% | +0.75 |
## Finding
- OLMoE: no gate tripped; the 205k prefix matches the whole file (761,856 tokens, 3.7×).
- FLAME: gate tripped on mean only (+0.43 / +1.21 pp); P95 (0.14% → 0.33%, 0.64% → 1.46%) stays inside both gates.
- Stage 1 stays on the 205k / 250k prefix; the other 87 FLAME parquets are not used.
## Limits
- FLAME checked on one layer and one checkpoint; same file, later blocks only.
