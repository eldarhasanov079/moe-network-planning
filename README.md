# MoE network planning

Code and measurements behind *Towards Predictable Scale-Out Communication for
Mixture-of-Experts Training* (MSc Computing, Imperial College London, 2026).

Three parts:

1. **Trace analysis** of published MoE router traces (FLAME-MoE, OLMoE): how
   predictable expert-parallel All-to-All traffic is, at four aggregation levels.
2. **MoE-MLSynth**: a fork of MLSynth that synthesises Chakra execution traces for
   MoE models with non-uniform expert-parallel traffic.
3. **PLANE** (Profile-guided Link Allocation for Networked Experts): turns a short
   traffic profile into a per-layer reservation and a byte matrix for a network
   planner, then handles the leftover traffic at runtime.

## Layout

| Path | Contents |
|---|---|
| `moe-trace-experiments/` | Experiments A to O on the public traces, shared toolkit in `common/`, thesis figures in `figures/`. See its README. |
| `MoE-MLSynth/` | Modified MLSynth (submodule, branch `add-moe-synthesis`). Adds `transformer_moe`, expert-demand distributions, placement and capacity rules. |
| `plane/` | PLANE package (`moe_feeder`), CLI `plane`, experiment scripts, tests, measured reports. See its README. |
| `chakra_inputs/` | MLSynth configs used for synthesis sweeps. |
| `astra-sim/`, `astra-sim-expanded/` | ASTRA-sim 2.0 and the group's fork with concurrent-flow support (submodules). Run through Docker. |
| `FLAME-MoE/`, `OLMoE/`, `OpenMoE/` | Model repositories the traces come from (submodules). |
| `HolisticTraceAnalysis/`, `param/` | Execution-trace tooling (submodules). |

Not included: the network planner that PLANE feeds (Puppeteer, unpublished group
work). PLANE imports it as the `puppeteer` package and expects a checkout at
`Puppeteer/`. Trace analysis and MoE-MLSynth do not need it.

## Environment

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
git clone --recurse-submodules <this-repo>
cd moe-network-planning
uv sync --extra dev
source .venv/bin/activate
```

Traces are streamed from Hugging Face on first use and cached under
`moe-trace-experiments/data_cache/`.

## Reproduce

```bash
# trace analysis (no GPU, no training)
python moe-trace-experiments/exp_B_routing_stabilization/run.py
python moe-trace-experiments/figures/redraw_ch3.py

# MoE trace synthesis
cd MoE-MLSynth && python synthesise_workload.py   # reads input-moe.yaml

# PLANE (needs the planner package)
pytest plane/tests
plane experiment --suite claim-v1 -o plane/output
python plane/scripts/flagship_stage1.py
```

Generated data (`plane/output/`, `data_cache/`, Chakra `.et` files) is not
committed. Measured numbers live in `plane/reports/` and in each experiment's
`REPORT.md`.
