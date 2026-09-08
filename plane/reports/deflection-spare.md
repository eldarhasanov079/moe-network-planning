# Experiment D: deflect overflow to a spare expert
| | |
|---|---|
| Question | Under the frozen Stage 1 plan a send overflows its reservation cell about 0.7% (FLAME) / 2.2% (OLMoE) of the time per microbatch and is dropped. Can the dispatcher send that slot to a different expert with room, and at what cost? |
| Inputs | FLAME 290M, 721M, 1.7B and OLMoE traces; emulated top-(k−1) router with the recorded k-th choice as spare (FLAME top-5 + 6th, OLMoE top-7 + 8th); `E` re-profiled as P95 over the Stage 1 profile window; ~1 microbatch; 1.7B excludes checkpoint 3300 (whole-trace values in Limits). |
| Model | numpy slot-level admission (batch dispatch, causal walk as check); the planner's Stage 2b full-step model (dispatch, expert kernel, combine; fair share within class) for the regression check. |
| Reproduce | `plane experiment --suite deflect -o plane/output` (~13 min, 8 workers); `.venv/bin/python plane/scripts/deflect_neighbourhood.py`; `.venv/bin/python plane/scripts/flagship_stage3.py` (figures D1, D2 and R1-R3; `_reaggregate_without_collapse` excludes the 1.7B collapse and marks the model with an asterisk). |
| Outputs | `plane/output/stage3/figures/figD1_resolved.png`, `figD2_cost.png`; `plane/output/stage3/deflect.csv`, `deflect_summary.json` (aggregates include ckpt 3300), `deflect_puppeteer.csv`, `deflect_neighbourhood.json` |
## Setup
- Primary admission: the k−1 chosen slots of each token admitted into cells up to `floor(E)`, in arrival order or in descending renormalised router weight ("score order"; OLMoE has no scores and sorts by router rank).
- Overflow policy: dropped (Stage 1); spare expert's device if that cell has room and the token has not used its spare (two spares tried in rank order); any device of the source with room (ceiling); or spare-then-any. Combine returns on the transposed cell; the source masks a dropped expert out.
## Results
### Resolution, ~1 microbatch, contiguous placement, one spare, arrival order unless stated
| | FLAME 290M | FLAME 721M | FLAME 1.7B | OLMoE |
|---|---:|---:|---:|---:|
| overflow, top-(k−1) system / real top-k control | 0.73% / 0.67% | 0.68% / 0.64% | 1.00% / 0.89% | 2.33% / 2.20% |
| tokens with ≥1 overflow slot | 2.4% | 2.2% | 3.0% | 8.4% |
| resolved by the spare, median (min–max over layer × ckpt) | 43% (34–67) | 43% (29–69) | 38% (28–57) | 31% (24–46) |
| the spare's own cell was full | 21% | 20% | 23% | 20% |
| resolved, dispatcher sorts by score / rank | 58% (48–78) | 59% (40–82) | 51% (36–68) | 58% (48–69) |
| resolved with two spares (top-(k−2) system) | 71% (61–88) | 69% (57–92) | 65% (51–84) | 56% (44–72) |
| resolved by any device with room (ceiling) | 100% | 100% | 100% | 100% |
| causal walk instead of batch | 48% (+7.1% induced overflow) | 48% (+7.2%) | 44% (+8.4%) | 37% (+7.6%) |
| round-robin / random / random placement | 40 / 41 / 41% | 44 / 44 / 45% | 37 / 38 / 36% | 30 / 31 / 31% |
| slots still dropped: drop → spare → two spares | 0.73 → 0.42 → 0.25% | 0.68 → 0.39 → 0.23% | 1.00 → 0.63 → 0.44% | 2.33 → 1.59 → 1.09% |
| reservation used: drop → spare → any | 97.0 → 97.3 → 97.7% | 97.0 → 97.3 → 97.6% | 97.4 → 97.8 → 98.4% | 94.0 → 94.7 → 96.2% |
Step size (FLAME-290M, spare, arrival): 55% resolved at 250k, 43% at 125k, 37% at 32k tokens; spare cell full 16% → 21% → 24%. Spare on a device the token already sends to: 46% on FLAME vs chance 50% (58% vs 64% on OLMoE). Largest single-step utilisation 99.3%.
### Cost, FLAME, router weight lost per token, ×1e-3, ~1 microbatch; gap = `w_j − w_spare`, pessimistic = `w_j`; renormalised top-5 weights average 0.35 / 0.22 / 0.17 / 0.14 / 0.12
| | arrival order | score order |
|---|---:|---:|
| drop everything (Stage 1) | 7.4 | 2.2 |
| spare: preference gap / pessimistic | 5.7 / 7.4 | 1.1 / 2.2 |
| any device: lower bound / pessimistic | 3.7 / 7.4 | 0.4 / 2.2 |
| Stage 2 tail + deadline cut at zero lateness (for scale) | 0.28 | 0.28 |
721M and 1.7B: 5.2 / 6.8 and 8.1 / 10.1 under arrival order. Planner regression (FLAME-290M, Stage 2b full-step, fair share): single spare 173.78 ms vs 173.35 ms drop matrix, +0.25% time for +0.34% slots; spare-then-any +0.69% for +0.76%; slack-tiered two-problem planning charges +1.2% and +5.0% instead.
## Finding
- One spare resolves 31–43% of overflows, 51–59% with score sorting, 56–71% with two spares; any device 100%.
- Placement moves median resolution by at most 5 points (290M 40–43%, 721M 43–45%, 1.7B 36–38%, OLMoE 30–31%).
- Remaining drops (0.4% / 1.6% of slots) are about 5× / 3× the Stage 2 deadline cut (0.085% / 0.48%).
- Planned step time tracks the byte share: +0.25% for +0.34% slots.
## Limits
- No training loss; the emulated top-(k−1) system overflows within about 10% of the real top-k one; with ckpt 3300 included, 1.7B overflow is 4.8% of slots, spare resolves 36%, drop cost 51.5e-3, spare gap 46.6e-3.
