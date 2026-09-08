# Chapter 6 figures
Each stem is saved as `.pdf`, `.svg` and `.png`. Redraw with `python plane/figures/chapter6/redraw_ch6.py` from the repository root inside the `.venv`.
Stage 2 curves are measured (`plane/output/stage2/summary.json`); Stage 3 bars are transcribed from `plane/reports/deflection-spare.md` and `reconfig-runtime.md` (the stage3 output directory is not in-tree). Schematics (decision tree, two-class step, spare, guard) are Mermaid in `chapter6.md`, not in this folder.

| File | Thesis figure | Shows |
|---|---|---|
| `fig6_1_leftover_{flame,olmoe}` | Fig. 6.2 | Per-step leftover vs idle of a frozen P95, one model per figure |
| `fig6_1_tightness` | Fig. 6.3 | Idle reserve vs tail lateness |
| `fig6_2_drop_cost_flame` | Fig. 6.4 | Arrival leftover is average weight |
| `fig6_2_leftover_rank_olmoe` | Fig. 6.5 | Rank-order leftover sits on expert 8 |
| `fig6_3_deadline_{flame,olmoe}` | Fig. 6.7 | Sender-side quota vs drop-all |
| `fig6_4_kernel_{flame,olmoe}` | Fig. 6.8 | Monolithic vs chunked vs $r$ |
| `fig6_5_step_{flame,olmoe}` | Fig. 6.9 | Full step at the compute anchor |
| `fig6_6_spare_resolved` | Fig. 6.11 | Spare resolution; 1.7B excludes ckpt 3300 |
| `fig6_7_remaining_drops` | Fig. 6.12 | Order of magnitude only; spare uses emulated $k$ |
| `fig6_8_reconfig_stable` | Fig. 6.14 | Stable traces: headroom, not installs |
| `fig6_9_collapse_timetable` | Fig. 6.16 | Guard inflate then release, layer 02 |
| `fig6_9_collapse_layer_times` | Fig. 6.15 | Reconfig does not beat the hot NIC |
