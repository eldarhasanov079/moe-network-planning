# Packet and flowlet spraying with finite reorder buffers
| | |
|---|---|
| Question | On the Stage 1 P95 admitted last-checkpoint matrices, how much does spraying gain over the planned single path, and how large a reorder buffer does it need? |
| Inputs | Stage 1 P95 admitted last-checkpoint matrices, FLAME and OLMoE; each matrix cell is one ordered message; whole model = sum over 16 FLAME or 32 OLMoE phases. |
| Model | Analytical: directed FIFO links, store-and-forward serialisation, 500 ns per Clos hop, 100 ns intra-host, phase barriers between dispatch and combine. `backpressure` = per-message credit window, 6 us credit RTT. `retransmit` = drop excess out-of-order flowlets, wait for the phase burst to drain, replay after 50 us on one ordered recovery path (conservative). ECMP = mean of eight hash seeds; planned and spray deterministic. |
| Reproduce | `plane experiment --suite packet-spray -o plane/output` |
| Outputs | `plane/output/packet-spray/` |
## Results
### Main result, 64 KiB flowlets
| Model | Planned | ECMP (8 seeds) | Spray, unbounded | Spray speedup vs plan | Peak reorder / message | Peak reorder / GPU |
|---|---:|---:|---:|---:|---:|---:|
| FLAME | 379.39 ms | 539.37 ± 39.41 ms | 379.10 ms | 1.001x | 0.75 MiB | 0.84 MiB |
| OLMoE | 925.01 ms | 1180.94 ± 72.56 ms | 926.24 ms | 0.999x | 0.81 MiB | 1.12 MiB |
### Buffer sweep, 64 KiB flowlets
| Model | Policy | Buffer/message | Time (ms) | Speedup vs plan | Retransmitted | Peak/GPU (MiB) |
|---|---|---:|---:|---:|---:|---:|
| FLAME | backpressure | 0 | 598.14 | 0.634x | 0.00% | 0.00 |
| FLAME | backpressure | 64 KiB | 404.37 | 0.938x | 0.00% | 0.25 |
| FLAME | backpressure | 256 KiB | 380.46 | 0.997x | 0.00% | 0.62 |
| FLAME | backpressure | 1 MiB | 379.34 | 1.000x | 0.00% | 0.62 |
| FLAME | backpressure | 4 MiB | 379.31 | 1.000x | 0.00% | 0.75 |
| FLAME | retransmit | 0 | 1606.91 | 0.236x | 85.33% | 0.00 |
| FLAME | retransmit | 64 KiB | 530.13 | 0.716x | 5.71% | 0.38 |
| FLAME | retransmit | 256 KiB | 384.37 | 0.987x | 0.15% | 1.00 |
| FLAME | retransmit | 1 MiB | 379.10 | 1.001x | 0.00% | 0.84 |
| FLAME | retransmit | 4 MiB | 379.10 | 1.001x | 0.00% | 0.84 |
| FLAME | unbounded | unbounded | 379.10 | 1.001x | 0.00% | 0.84 |
| OLMoE | backpressure | 0 | 1439.79 | 0.642x | 0.00% | 0.00 |
| OLMoE | backpressure | 64 KiB | 971.29 | 0.952x | 0.00% | 0.25 |
| OLMoE | backpressure | 256 KiB | 929.02 | 0.996x | 0.00% | 0.62 |
| OLMoE | backpressure | 1 MiB | 926.57 | 0.998x | 0.00% | 0.62 |
| OLMoE | backpressure | 4 MiB | 926.54 | 0.998x | 0.00% | 0.75 |
| OLMoE | retransmit | 0 | 3605.57 | 0.257x | 85.19% | 0.00 |
| OLMoE | retransmit | 64 KiB | 1449.13 | 0.638x | 11.52% | 0.38 |
| OLMoE | retransmit | 256 KiB | 954.05 | 0.970x | 0.44% | 1.38 |
| OLMoE | retransmit | 1 MiB | 926.24 | 0.999x | 0.00% | 1.12 |
| OLMoE | retransmit | 4 MiB | 926.24 | 0.999x | 0.00% | 1.12 |
| OLMoE | unbounded | unbounded | 926.24 | 0.999x | 0.00% | 1.12 |
### Granularity sweep, unbounded reorder buffer
| Model | Unit | Time (ms) | Out-of-order packets | Peak/message (MiB) | Peak/GPU (MiB) |
|---|---:|---:|---:|---:|---:|
| FLAME | 4 KiB | 377.83 | 26.1% | 0.05 | 0.07 |
| FLAME | 16 KiB | 378.09 | 28.3% | 0.20 | 0.22 |
| FLAME | 64 KiB | 379.10 | 28.4% | 0.75 | 0.84 |
| FLAME | 256 KiB | 384.67 | 28.5% | 2.51 | 3.58 |
| OLMoE | 4 KiB | 923.87 | 25.0% | 0.07 | 0.08 |
| OLMoE | 16 KiB | 924.34 | 26.9% | 0.22 | 0.23 |
| OLMoE | 64 KiB | 926.24 | 26.9% | 0.81 | 1.12 |
| OLMoE | 256 KiB | 933.84 | 27.2% | 3.75 | 4.49 |
## Finding
- Unbounded spray matches the planned path: 1.001x (FLAME) / 0.999x (OLMoE); ECMP is 539.37 / 1180.94 ms.
- Backpressure reaches 0.997x / 0.996x at 256 KiB per message; retransmit needs 1 MiB to reach 1.001x / 0.999x.
- Peak reorder buffer per GPU 0.84 / 1.12 MiB at 64 KiB flowlets; 3.58 / 4.49 MiB at 256 KiB.
## Limits
- No headers, switch buffer limits, PFC/DCQCN, ACK traffic, loss detection or compute/communication overlap; a transport-level analytical estimate, not a packet-accurate RoCE result.
