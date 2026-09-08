"""Run the finite-reorder-buffer Clos spraying experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from puppeteer.config import RunConfig

from .chakra import matrix_from_chakra
from .config import DEFAULT_TOPOLOGY
from .packet_spray import (
    PacketModelConfig,
    PacketSimulationResult,
    RouteTable,
    route_table,
    simulate_phase,
)
from .planner import freeze_paths


UNIT_BYTES = 64 * 1024
BUFFER_BYTES = (0, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024)
GRANULARITIES = (4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024)
LINK_LATENCY_NS = 500.0
INTRA_HOST_LATENCY_NS = 100.0
CREDIT_RTT_US = 6.0
RETRANSMIT_TIMEOUT_US = 50.0
ECMP_SEEDS = 8

DATASETS = {
    "flame": {"stem": "flame_adm_p95_5473", "phases": 16},
    "olmoe": {"stem": "olmoe_adm_p95_final", "phases": 32},
}
SOURCE_LABELS = {"flame": "FLAME", "olmoe": "OLMoE"}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_phases(source: str, out: Path) -> List[np.ndarray]:
    spec = DATASETS[source]
    repo = Path(__file__).resolve().parents[1]
    work = repo / "output" / "claim-v1" / "astrasim_work"
    cache = out / "input_matrices" / source
    cache.mkdir(parents=True, exist_ok=True)
    phases: List[np.ndarray] = []
    for phase in range(int(spec["phases"])):
        cached = cache / "p{}.npy".format(phase)
        if cached.exists():
            matrix = np.load(cached)
        else:
            trace = work / "{}_p{}".format(spec["stem"], phase)
            if not trace.exists():
                raise FileNotFoundError(
                    "missing closed Stage-1 Chakra phase {}".format(trace)
                )
            matrix, _ = matrix_from_chakra(trace)
            np.save(cached, matrix)
        phases.append(np.asarray(matrix, dtype=np.int64))

    for phase in range(0, len(phases), 2):
        if not np.array_equal(phases[phase + 1], phases[phase].T):
            raise ValueError(
                "{} phases {} and {} are not dispatch/combine transposes".format(
                    source, phase, phase + 1
                )
            )
    return phases


def _aggregate(results: Sequence[PacketSimulationResult]) -> Dict[str, object]:
    messages = sum(result.messages for result in results)
    packets = sum(result.original_packets for result in results)
    transmissions = sum(result.transmissions for result in results)
    original_bytes = sum(result.original_bytes for result in results)
    transmitted_bytes = sum(result.transmitted_bytes for result in results)
    ooo = sum(result.out_of_order_packets for result in results)
    weighted_peak = sum(
        result.mean_message_peak_reorder_bytes * result.messages
        for result in results
    )
    return {
        "completion_time_s": sum(result.completion_time_s for result in results),
        "messages": messages,
        "original_packets": packets,
        "transmissions": transmissions,
        "retransmissions": sum(result.retransmissions for result in results),
        "dropped_arrivals": sum(result.dropped_arrivals for result in results),
        "original_bytes": original_bytes,
        "transmitted_bytes": transmitted_bytes,
        "extra_transmitted_fraction": (
            transmitted_bytes / original_bytes - 1.0 if original_bytes else 0.0
        ),
        "out_of_order_packets": ooo,
        "out_of_order_fraction": ooo / packets if packets else 0.0,
        "max_reorder_distance_packets": max(
            (result.max_reorder_distance_packets for result in results), default=0
        ),
        "max_message_reorder_bytes": max(
            (result.max_message_reorder_bytes for result in results), default=0
        ),
        "mean_message_peak_reorder_bytes": weighted_peak / messages if messages else 0.0,
        "max_receiver_reorder_bytes": max(
            (result.max_receiver_reorder_bytes for result in results), default=0
        ),
    }


def _run_whole(
    phases: Sequence[np.ndarray],
    topology,
    routes: RouteTable,
    model: PacketModelConfig,
) -> Tuple[Dict[str, object], List[PacketSimulationResult]]:
    results = [simulate_phase(matrix, topology, routes, model) for matrix in phases]
    return _aggregate(results), results


def _model(
    *,
    unit_bytes: int = UNIT_BYTES,
    policy: str = "unbounded",
    buffer_bytes: Optional[int] = None,
    credit_rtt_us: float = CREDIT_RTT_US,
    retransmit_timeout_us: float = RETRANSMIT_TIMEOUT_US,
) -> PacketModelConfig:
    return PacketModelConfig(
        unit_bytes=unit_bytes,
        link_latency_ns=LINK_LATENCY_NS,
        intra_host_latency_ns=INTRA_HOST_LATENCY_NS,
        overflow_policy=policy,
        buffer_bytes=buffer_bytes,
        credit_rtt_us=credit_rtt_us,
        retransmit_timeout_us=retransmit_timeout_us,
    )


def _fluid_baselines() -> Dict[str, Dict[str, float]]:
    repo = Path(__file__).resolve().parents[1]
    path = repo / "output" / "clos-baselines" / "summary.json"
    raw = json.loads(path.read_text())
    answer: Dict[str, Dict[str, float]] = {}
    for source in DATASETS:
        routers = raw["sources"][source]["p95"]["routers"]
        answer[source] = {
            "planned_s": float(routers["least-loaded"]["mean_s"]),
            "ecmp_s": float(routers["ecmp"]["mean_s"]),
            "ecmp_std_s": float(routers["ecmp"]["std_s"]),
            "spray_s": float(routers["spray"]["mean_s"]),
        }
    return answer


def _buffer_label(value: Optional[int]) -> str:
    if value is None:
        return "unbounded"
    if value == 0:
        return "0"
    if value < 1024 * 1024:
        return "{} KiB".format(value // 1024)
    return "{} MiB".format(value // (1024 * 1024))


def _draw_buffer_sweep(
    fig_dir: Path,
    rows: Sequence[Mapping[str, object]],
    packet_baselines: Mapping[str, Mapping[str, float]],
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    order: List[Optional[int]] = [*BUFFER_BYTES, None]
    labels = [_buffer_label(value) for value in order]
    colors = {"backpressure": "#277da1", "retransmit": "#d1495b"}
    for source in DATASETS:
        fig, (ax_time, ax_speed) = plt.subplots(1, 2, figsize=(11.2, 4.25))
        for mechanism in ("backpressure", "retransmit"):
            selected = {
                row["buffer_bytes"]: row
                for row in rows
                if row["source"] == source
                and row["mechanism"] in (mechanism, "unbounded")
            }
            times = [float(selected[value]["completion_time_s"]) * 1e3 for value in order]
            speedups = [
                float(selected[value]["speedup_vs_packet_plan"]) for value in order
            ]
            ax_time.plot(
                range(len(order)), times, marker="o", linewidth=2,
                label=mechanism, color=colors[mechanism],
            )
            ax_speed.plot(
                range(len(order)), speedups, marker="o", linewidth=2,
                label=mechanism, color=colors[mechanism],
            )
        plan_ms = packet_baselines[source]["planned_s"] * 1e3
        ecmp_ms = packet_baselines[source]["ecmp_s"] * 1e3
        ax_time.axhline(plan_ms, color="#333333", linestyle="--", label="planned path")
        ax_time.axhline(ecmp_ms, color="#777777", linestyle=":", label="ECMP")
        ax_speed.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2)
        for ax in (ax_time, ax_speed):
            ax.set_xticks(range(len(order)), labels, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.22)
        ax_time.set_ylabel("Whole-model communication time (ms)")
        ax_time.set_xlabel("Reorder buffer per message")
        ax_time.legend(fontsize=8)
        ax_speed.set_ylabel("Speedup over planned single path")
        ax_speed.set_xlabel("Reorder buffer per message")
        ax_speed.legend(fontsize=8)
        fig.suptitle("{}: finite-buffer packet spraying".format(SOURCE_LABELS[source]))
        fig.tight_layout()
        fig.savefig(fig_dir / "{}_buffer_sweep.png".format(source), dpi=180)
        plt.close(fig)


def _draw_granularity(
    fig_dir: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    palette = {"flame": "#3a6ea5", "olmoe": "#c65d21"}
    for source in DATASETS:
        chosen = sorted(
            (row for row in rows if row["source"] == source),
            key=lambda row: int(row["unit_bytes"]),
        )
        x = [int(row["unit_bytes"]) / 1024 for row in chosen]
        axes[0].plot(
            x,
            [float(row["completion_time_s"]) * 1e3 for row in chosen],
            marker="o", linewidth=2, label=SOURCE_LABELS[source], color=palette[source],
        )
        axes[1].plot(
            x,
            [float(row["max_receiver_reorder_bytes"]) / (1024 * 1024) for row in chosen],
            marker="o", linewidth=2, label=SOURCE_LABELS[source], color=palette[source],
        )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([value / 1024 for value in GRANULARITIES])
        ax.set_xticklabels([str(value // 1024) for value in GRANULARITIES])
        ax.set_xlabel("Packet / flowlet size (KiB)")
        ax.grid(axis="y", alpha=0.22)
        ax.legend()
    axes[0].set_ylabel("Whole-model communication time (ms)")
    axes[1].set_ylabel("Peak aggregate reorder memory / GPU (MiB)")
    fig.suptitle("Spraying granularity: completion time and reordering")
    fig.tight_layout()
    fig.savefig(fig_dir / "granularity_sweep.png", dpi=180)
    plt.close(fig)


def _report(
    summary: Mapping[str, object],
    buffer_rows: Sequence[Mapping[str, object]],
    granularity_rows: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# Packet/flowlet spraying with finite reorder buffers",
        "",
        "This is a new analytical experiment over the closed Stage-1 P95 admitted",
        "last-checkpoint matrices. Each matrix cell is one ordered message. The model",
        "uses directed FIFO links, store-and-forward serialization, 500 ns per Clos",
        "hop, 100 ns for an intra-host transfer, and phase barriers between dispatch",
        "and combine. Whole-model time is the sum over 16 FLAME or 32 OLMoE phases.",
        "",
        "The primary sweep uses 64 KiB flowlets. `backpressure` is a per-message",
        "credit window with a 6 us credit RTT. `retransmit` drops excess out-of-order",
        "flowlets, waits for the original phase burst to drain, and replays them after",
        "50 us on one ordered recovery path. The latter is intentionally conservative.",
        "",
        "## Main result",
        "",
        "| Model | Planned | ECMP (8 seeds) | Spray, unbounded | Spray speedup vs plan | Peak reorder / message | Peak reorder / GPU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    sources = summary["sources"]  # type: ignore[index]
    for source in DATASETS:
        block = sources[source]  # type: ignore[index]
        packet = block["packet"]
        unbounded = block["unbounded_spray"]
        lines.append(
            "| {} | {:.2f} ms | {:.2f} ± {:.2f} ms | {:.2f} ms | {:.3f}x | {:.2f} MiB | {:.2f} MiB |".format(
                SOURCE_LABELS[source],
                packet["planned_s"] * 1e3,
                packet["ecmp_s"] * 1e3,
                packet["ecmp_std_s"] * 1e3,
                unbounded["completion_time_s"] * 1e3,
                packet["planned_s"] / unbounded["completion_time_s"],
                unbounded["max_message_reorder_bytes"] / (1024 * 1024),
                unbounded["max_receiver_reorder_bytes"] / (1024 * 1024),
            )
        )
    lines += ["", "## Buffer sweep (64 KiB flowlets)", ""]
    for source in DATASETS:
        lines += ["### {}".format(SOURCE_LABELS[source]), ""]
        lines.append(
            "| Policy | Buffer/message | Time (ms) | Speedup vs plan | Retransmitted | Peak/GPU (MiB) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in buffer_rows:
            if row["source"] != source:
                continue
            lines.append(
                "| {} | {} | {:.2f} | {:.3f}x | {:.2%} | {:.2f} |".format(
                    row["mechanism"],
                    row["buffer_label"],
                    float(row["completion_time_s"]) * 1e3,
                    float(row["speedup_vs_packet_plan"]),
                    float(row["extra_transmitted_fraction"]),
                    float(row["max_receiver_reorder_bytes"]) / (1024 * 1024),
                )
            )
        lines.append("")
    lines += [
        "## Granularity sweep (unbounded reorder buffer)",
        "",
        "| Model | Unit | Time (ms) | Out-of-order packets | Peak/message (MiB) | Peak/GPU (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in granularity_rows:
        lines.append(
            "| {} | {} KiB | {:.2f} | {:.1%} | {:.2f} | {:.2f} |".format(
                SOURCE_LABELS[str(row["source"])],
                int(row["unit_bytes"]) // 1024,
                float(row["completion_time_s"]) * 1e3,
                float(row["out_of_order_fraction"]),
                float(row["max_message_reorder_bytes"]) / (1024 * 1024),
                float(row["max_receiver_reorder_bytes"]) / (1024 * 1024),
            )
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The simulator accounts for serialization, FIFO queueing, path-dependent",
        "arrival order, finite receiver memory, credit delay, and retransmitted bytes.",
        "ECMP is the mean of eight deterministic hash seeds; the planned and spray",
        "policies are deterministic. The simulator does not model headers, switch",
        "buffer limits, PFC/DCQCN, ACK traffic, loss",
        "detection, or compute/communication overlap. The results are therefore a",
        "transport-level analytical estimate, not a packet-accurate RoCE claim.",
        "",
    ]
    return "\n".join(lines)


def run_packet_spray_experiment(out: Path) -> Dict[str, object]:
    """Execute the experiment and write machine-readable results plus figures."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = RunConfig.load(str(DEFAULT_TOPOLOGY))
    topology = config.topology
    matrices = {source: _load_phases(source, out) for source in DATASETS}

    all_pairs = np.ones((topology.num_ranks, topology.num_ranks), dtype=np.int64)
    np.fill_diagonal(all_pairs, 0)
    planned_paths = freeze_paths(all_pairs, config, bytes_per_slot=1)
    routes = {
        "planned": route_table(
            topology, "planned", planned_paths=planned_paths
        ),
        "spray": route_table(topology, "spray"),
    }
    fluid = _fluid_baselines()

    phase_rows: List[Dict[str, object]] = []
    packet_baselines: Dict[str, Dict[str, float]] = {}
    unbounded_by_source: Dict[str, Dict[str, object]] = {}
    for source, phases in matrices.items():
        print("packet-spray baseline", source, flush=True)
        packet_baselines[source] = {}
        for routing in ("planned", "spray"):
            aggregate, details = _run_whole(
                phases, topology, routes[routing], _model()
            )
            packet_baselines[source]["{}_s".format(routing)] = float(
                aggregate["completion_time_s"]
            )
            if routing == "spray":
                unbounded_by_source[source] = aggregate
            for phase, result in enumerate(details):
                phase_rows.append(
                    {
                        "source": source,
                        "phase": phase,
                        "routing": routing,
                        "seed": 0,
                        **result.as_dict(),
                    }
                )
        ecmp_totals = []
        for seed in range(ECMP_SEEDS):
            aggregate, details = _run_whole(
                phases,
                topology,
                route_table(topology, "ecmp", seed=seed),
                _model(),
            )
            ecmp_totals.append(float(aggregate["completion_time_s"]))
            for phase, result in enumerate(details):
                phase_rows.append(
                    {
                        "source": source,
                        "phase": phase,
                        "routing": "ecmp",
                        "seed": seed,
                        **result.as_dict(),
                    }
                )
        packet_baselines[source]["ecmp_s"] = float(np.mean(ecmp_totals))
        packet_baselines[source]["ecmp_std_s"] = float(np.std(ecmp_totals))
    _write_csv(out / "phase_results.csv", phase_rows)

    buffer_rows: List[Dict[str, object]] = []
    for source, phases in matrices.items():
        unbounded = unbounded_by_source[source]
        packet_plan = packet_baselines[source]["planned_s"]
        packet_ecmp = packet_baselines[source]["ecmp_s"]
        for mechanism in ("backpressure", "retransmit"):
            for buffer_bytes in BUFFER_BYTES:
                print(
                    "packet-spray buffer", source, mechanism, _buffer_label(buffer_bytes),
                    flush=True,
                )
                aggregate, _ = _run_whole(
                    phases,
                    topology,
                    routes["spray"],
                    _model(policy=mechanism, buffer_bytes=buffer_bytes),
                )
                penalty = float(aggregate["completion_time_s"]) / float(
                    unbounded["completion_time_s"]
                )
                adjusted_fluid = fluid[source]["spray_s"] * penalty
                buffer_rows.append(
                    {
                        "source": source,
                        "mechanism": mechanism,
                        "buffer_bytes": buffer_bytes,
                        "buffer_label": _buffer_label(buffer_bytes),
                        **aggregate,
                        "penalty_vs_unbounded_spray": penalty,
                        "speedup_vs_packet_plan": packet_plan
                        / float(aggregate["completion_time_s"]),
                        "speedup_vs_packet_ecmp": packet_ecmp
                        / float(aggregate["completion_time_s"]),
                        "fluid_spray_with_measured_penalty_s": adjusted_fluid,
                        "speedup_vs_fluid_plan_after_penalty": fluid[source]["planned_s"]
                        / adjusted_fluid,
                    }
                )
        aggregate = unbounded_by_source[source]
        buffer_rows.append(
            {
                "source": source,
                "mechanism": "unbounded",
                "buffer_bytes": None,
                "buffer_label": "unbounded",
                **aggregate,
                "penalty_vs_unbounded_spray": 1.0,
                "speedup_vs_packet_plan": packet_plan
                / float(aggregate["completion_time_s"]),
                "speedup_vs_packet_ecmp": packet_ecmp
                / float(aggregate["completion_time_s"]),
                "fluid_spray_with_measured_penalty_s": fluid[source]["spray_s"],
                "speedup_vs_fluid_plan_after_penalty": fluid[source]["planned_s"]
                / fluid[source]["spray_s"],
            }
        )
    _write_csv(out / "buffer_sweep.csv", buffer_rows)

    granularity_rows: List[Dict[str, object]] = []
    for source, phases in matrices.items():
        for unit_bytes in GRANULARITIES:
            print(
                "packet-spray granularity", source, unit_bytes // 1024, "KiB",
                flush=True,
            )
            if unit_bytes == UNIT_BYTES:
                aggregate = unbounded_by_source[source]
            else:
                aggregate, _ = _run_whole(
                    phases,
                    topology,
                    routes["spray"],
                    _model(unit_bytes=unit_bytes),
                )
            granularity_rows.append(
                {"source": source, "unit_bytes": unit_bytes, **aggregate}
            )
    _write_csv(out / "granularity_sweep.csv", granularity_rows)

    sensitivity_rows: List[Dict[str, object]] = []
    sensitivity_buffer = 256 * 1024
    for source, phases in matrices.items():
        for credit_rtt in (3.0, 6.0, 12.0):
            aggregate, _ = _run_whole(
                phases,
                topology,
                routes["spray"],
                _model(
                    policy="backpressure",
                    buffer_bytes=sensitivity_buffer,
                    credit_rtt_us=credit_rtt,
                ),
            )
            sensitivity_rows.append(
                {
                    "source": source,
                    "mechanism": "backpressure",
                    "buffer_bytes": sensitivity_buffer,
                    "control_delay_us": credit_rtt,
                    **aggregate,
                }
            )
        for timeout in (10.0, 50.0, 100.0):
            aggregate, _ = _run_whole(
                phases,
                topology,
                routes["spray"],
                _model(
                    policy="retransmit",
                    buffer_bytes=sensitivity_buffer,
                    retransmit_timeout_us=timeout,
                ),
            )
            sensitivity_rows.append(
                {
                    "source": source,
                    "mechanism": "retransmit",
                    "buffer_bytes": sensitivity_buffer,
                    "control_delay_us": timeout,
                    **aggregate,
                }
            )
    _write_csv(out / "penalty_sensitivity.csv", sensitivity_rows)

    summary: Dict[str, object] = {
        "configuration": {
            "topology": str(DEFAULT_TOPOLOGY),
            "primary_unit_bytes": UNIT_BYTES,
            "buffer_bytes": list(BUFFER_BYTES),
            "granularities_bytes": list(GRANULARITIES),
            "link_latency_ns": LINK_LATENCY_NS,
            "intra_host_latency_ns": INTRA_HOST_LATENCY_NS,
            "credit_rtt_us": CREDIT_RTT_US,
            "retransmit_timeout_us": RETRANSMIT_TIMEOUT_US,
            "ecmp_seeds": ECMP_SEEDS,
            "queue_model": "directed FIFO, store-and-forward",
            "phase_model": "dispatch/combine serialized; no phase overlap",
        },
        "sources": {
            source: {
                "phases": len(matrices[source]),
                "packet": packet_baselines[source],
                "fluid_reference": fluid[source],
                "unbounded_spray": unbounded_by_source[source],
            }
            for source in DATASETS
        },
    }
    _write_json(out / "summary.json", summary)
    _draw_buffer_sweep(out / "figures", buffer_rows, packet_baselines)
    _draw_granularity(out / "figures", granularity_rows)
    report = _report(summary, buffer_rows, granularity_rows)
    (out / "REPORT.md").write_text(report)
    reports = Path(__file__).resolve().parents[1] / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "packet-spray-analytical.md").write_text(report)
    print(json.dumps(summary, indent=2)[:5000])
    return summary
