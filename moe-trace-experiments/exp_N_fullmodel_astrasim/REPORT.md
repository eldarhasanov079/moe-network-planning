# Exp N: Full MoE forward communication on ASTRA-sim

| | |
|---|---|
| Question | Total ASTRA-sim communication time for the full FLAME-MoE-290M MoE forward pass with real top-6 routing. |
| Data | FLAME-MoE-290M actives, 8 MoE layers (`layer_02`–`layer_09`), top-6 (`moe-router-topk 6`), final checkpoint, 250k tokens/layer |
| Model | EP = 8, contiguous placement; 2048 B/slot (`hidden_size=1024` × bf16); dispatch + combine All-to-All per layer (combine = transpose); Switch fabric, 100 GB/s |
| Run | `python exp_N_fullmodel_astrasim/run.py` (needs Docker, the `astra-sim:latest` image and a Python environment with the `chakra` package) |
| Outputs | `results/fullmodel_comm_by_layer.csv`, `results/fullmodel_comm_total.csv`, `fig1_comm_by_layer.png`, `fig2_comm_total.png` |

## Setup
- Included: all 8 MoE layers from FLAME actives traces; top-6 membership dispatch; EP = 8 from `flame-moe-290m.sh` (`EXPERT_MODEL_PARALLEL_SIZE=8`); 2048 B/slot; ASTRA-sim Switch comm time.
- Excluded: backward pass (traces are forward routing snapshots), dense `layer_01`, shared expert, embedding / LM head, GPU compute time (analytical ASTRA-sim cannot run COMP ET reliably), deflection, full training iteration.
- Assumed: contiguous token → DP rank sharding (traces have no rank ids); 100 GB/s Switch is a simulator parameter, not a FLAME cluster measurement.

## Results

| Quantity | Value |
|---|---|
| Total comm, 8 layers, 16 All-to-Alls | 149,677,088 cycles |
| Per-layer comm | ~18.2–19.1M cycles/layer, ~±3% spread |
| Slots per layer | 250k tokens × top-6 = 1.5M |

## Finding
- Full MoE forward comm: 149,677,088 ASTRA-sim cycles on Switch / EP=8.
- Per-layer comm is flat (~±3%), consistent with Exp E/H stability at membership-traffic level.

## Limits
- Forward routing only; no backward, optimiser, dense layer, shared expert or compute time.
- Absolute cycles depend on the 100 GB/s Switch parameter and 2048 B/slot; Exp H showed robustness to placement/sharding sweeps.
