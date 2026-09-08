"""Packet/flowlet-level analytical model for Clos path spraying."""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from puppeteer.planner.router import (
    ClosEcmpRouter,
    enumerate_clos_routes,
)
from puppeteer.planner.state import NetworkState
from puppeteer.topology import Route, Topology


Path = Tuple[str, ...]
RouteTable = Dict[Tuple[int, int], Tuple[Path, ...]]


@dataclass(frozen=True)
class PacketModelConfig:
    """Configuration of one analytical phase simulation."""

    unit_bytes: int = 64 * 1024
    link_latency_ns: float = 500.0
    intra_host_latency_ns: float = 100.0
    buffer_bytes: Optional[int] = None
    overflow_policy: str = "unbounded"
    credit_rtt_us: float = 6.0
    retransmit_timeout_us: float = 50.0
    max_transmissions_per_unit: int = 256

    def validate(self) -> None:
        if self.unit_bytes <= 0:
            raise ValueError("unit_bytes must be positive")
        if self.link_latency_ns < 0 or self.intra_host_latency_ns < 0:
            raise ValueError("link latencies must be non-negative")
        if self.buffer_bytes is not None and self.buffer_bytes < 0:
            raise ValueError("buffer_bytes must be non-negative or None")
        if self.overflow_policy not in ("unbounded", "backpressure", "retransmit"):
            raise ValueError(
                "overflow_policy must be unbounded, backpressure, or retransmit"
            )
        if self.overflow_policy != "unbounded" and self.buffer_bytes is None:
            raise ValueError("a finite-buffer policy requires buffer_bytes")
        if self.credit_rtt_us < 0 or self.retransmit_timeout_us < 0:
            raise ValueError("control delays must be non-negative")
        if self.max_transmissions_per_unit < 1:
            raise ValueError("max_transmissions_per_unit must be at least one")


@dataclass(frozen=True)
class PacketSimulationResult:
    """Summary returned by :func:`simulate_phase`."""

    completion_time_s: float
    messages: int
    original_packets: int
    transmissions: int
    retransmissions: int
    dropped_arrivals: int
    original_bytes: int
    transmitted_bytes: int
    out_of_order_packets: int
    out_of_order_fraction: float
    max_reorder_distance_packets: int
    max_message_reorder_bytes: int
    mean_message_peak_reorder_bytes: float
    max_receiver_reorder_bytes: int
    buffer_bytes: Optional[int]
    overflow_policy: str
    unit_bytes: int

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class _Message:
    src: int
    dst: int
    size_bytes: int
    packets: int
    routes: Tuple[Path, ...]
    expected: int = 0
    next_to_send: int = 0
    buffered_bytes: int = 0
    peak_buffered_bytes: int = 0
    peak_distance: int = 0
    completion_s: float = 0.0


def route_table(
    topology: Topology,
    policy: str,
    *,
    seed: int = 0,
    planned_paths: Optional[Mapping[Tuple[int, int], Route]] = None,
) -> RouteTable:
    """Build the path choices used by the packet model."""

    kind = policy.replace("_", "-").lower()
    n = int(topology.num_ranks)
    out: RouteTable = {}
    ecmp = ClosEcmpRouter(topology, seed=seed) if kind == "ecmp" else None
    state = NetworkState(topology)
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            if kind == "spray":
                choices = enumerate_clos_routes(topology, src, dst)
            elif kind == "ecmp":
                assert ecmp is not None
                choices = [ecmp.route(src, dst, state)]
            elif kind in ("planned", "least-loaded"):
                if planned_paths is None or (src, dst) not in planned_paths:
                    raise ValueError(
                        "planned routing requires a frozen path for ({}, {})".format(
                            src, dst
                        )
                    )
                choices = [planned_paths[(src, dst)]]
            else:
                raise ValueError("unknown routing policy {!r}".format(policy))
            out[(src, dst)] = tuple(tuple(route.links) for route in choices)
    return out


def _packet_size(message_bytes: int, seq: int, unit_bytes: int) -> int:
    start = seq * unit_bytes
    return min(unit_bytes, message_bytes - start)


def simulate_phase(
    matrix_bytes: np.ndarray,
    topology: Topology,
    routes: RouteTable,
    config: PacketModelConfig = PacketModelConfig(),
) -> PacketSimulationResult:
    """Simulate one all-to-all phase and return completion/reordering metrics."""

    config.validate()
    matrix = np.asarray(matrix_bytes)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix_bytes must be square")
    if matrix.shape[0] != int(topology.num_ranks):
        raise ValueError(
            "matrix has {} ranks, topology has {}".format(
                matrix.shape[0], topology.num_ranks
            )
        )
    if np.any(matrix < 0):
        raise ValueError("matrix_bytes cannot contain negative values")

    messages: List[_Message] = []
    by_source: Dict[int, List[int]] = {src: [] for src in range(matrix.shape[0])}
    for src in range(matrix.shape[0]):
        for dst in range(matrix.shape[1]):
            size = int(matrix[src, dst])
            if src == dst or size <= 0:
                continue
            choices = routes.get((src, dst))
            if not choices:
                raise ValueError("no route choices for ({}, {})".format(src, dst))
            msg_id = len(messages)
            messages.append(
                _Message(
                    src=src,
                    dst=dst,
                    size_bytes=size,
                    packets=int(math.ceil(size / config.unit_bytes)),
                    routes=tuple(tuple(path) for path in choices),
                )
            )
            by_source[src].append(msg_id)

    if not messages:
        return PacketSimulationResult(
            completion_time_s=0.0,
            messages=0,
            original_packets=0,
            transmissions=0,
            retransmissions=0,
            dropped_arrivals=0,
            original_bytes=0,
            transmitted_bytes=0,
            out_of_order_packets=0,
            out_of_order_fraction=0.0,
            max_reorder_distance_packets=0,
            max_message_reorder_bytes=0,
            mean_message_peak_reorder_bytes=0.0,
            max_receiver_reorder_bytes=0,
            buffer_bytes=config.buffer_bytes,
            overflow_policy=config.overflow_policy,
            unit_bytes=config.unit_bytes,
        )

    events: List[Tuple[float, int, int, int, int, int, int]] = []
    ticket = 0
    transmissions = 0
    retransmissions = 0
    transmitted_bytes = 0
    transmission_counts: Dict[Tuple[int, int], int] = {}

    pseudo_prefix = "@scaleup:"
    capacities = {
        link_id: float(link.capacity_bps)
        for link_id, link in topology.links.items()
    }
    link_available: Dict[str, float] = {}
    fabric_latency_s = config.link_latency_ns * 1e-9
    scaleup_latency_s = config.intra_host_latency_ns * 1e-9

    def effective_path(msg: _Message, seq: int, attempt: int) -> Path:
        index = seq % len(msg.routes) if attempt == 0 else 0
        path = msg.routes[index]
        if path:
            return path
        pseudo = "{}{}->{}".format(pseudo_prefix, msg.src, msg.dst)
        capacities[pseudo] = float(topology.scale_up_bps)
        return (pseudo,)

    def schedule(msg_id: int, seq: int, ready_s: float, attempt: int = 0) -> None:
        nonlocal ticket, transmissions, retransmissions, transmitted_bytes
        msg = messages[msg_id]
        size = _packet_size(msg.size_bytes, seq, config.unit_bytes)
        key = (msg_id, seq)
        count = transmission_counts.get(key, 0) + 1
        transmission_counts[key] = count
        if count > config.max_transmissions_per_unit:
            raise RuntimeError(
                "packet ({}, {}) exceeded {} transmissions; finite-buffer retry "
                "did not converge".format(
                    msg_id, seq, config.max_transmissions_per_unit
                )
            )
        ticket += 1
        heapq.heappush(events, (ready_s, ticket, msg_id, seq, 0, size, attempt))
        transmissions += 1
        transmitted_bytes += size
        if attempt:
            retransmissions += 1

    if config.overflow_policy == "backpressure":
        assert config.buffer_bytes is not None
        initial_window = 1 + config.buffer_bytes // config.unit_bytes
    else:
        initial_window = max(msg.packets for msg in messages)

    for src in range(matrix.shape[0]):
        ids = sorted(by_source[src], key=lambda i: messages[i].dst)
        if not ids:
            continue
        depth = max(min(initial_window, messages[i].packets) for i in ids)
        for seq in range(depth):
            for msg_id in ids:
                msg = messages[msg_id]
                if seq < min(initial_window, msg.packets):
                    schedule(msg_id, seq, 0.0)
                    msg.next_to_send = seq + 1

    buffered: List[Dict[int, int]] = [dict() for _ in messages]
    receiver_buffer = [0 for _ in range(matrix.shape[0])]
    max_receiver_buffer = 0
    out_of_order = 0
    max_distance = 0
    dropped = 0
    pending_retries: List[set[int]] = [set() for _ in messages]
    recovery_round = 0
    last_arrival_s = 0.0
    completed = 0
    completion = 0.0
    credit_delay_s = config.credit_rtt_us * 1e-6
    retry_delay_s = config.retransmit_timeout_us * 1e-6

    while completed < len(messages):
        if not events:
            if config.overflow_policy != "retransmit":
                break
            retry_ids = [
                (msg_id, seq)
                for msg_id, seqs in enumerate(pending_retries)
                for seq in sorted(seqs)
                if seq >= messages[msg_id].expected
            ]
            if not retry_ids:
                break
            recovery_round += 1
            ready_s = last_arrival_s + retry_delay_s
            for msg_id, seq in retry_ids:
                pending_retries[msg_id].discard(seq)
                schedule(msg_id, seq, ready_s, recovery_round)

        ready_s, _, msg_id, seq, hop, size, attempt = heapq.heappop(events)
        msg = messages[msg_id]
        path = effective_path(msg, seq, attempt)
        link_id = path[hop]
        start_s = max(ready_s, link_available.get(link_id, 0.0))
        finish_s = start_s + size * 8.0 / capacities[link_id]
        link_available[link_id] = finish_s
        latency_s = scaleup_latency_s if link_id.startswith(pseudo_prefix) else fabric_latency_s
        arrival_s = finish_s + latency_s
        last_arrival_s = max(last_arrival_s, arrival_s)

        if hop + 1 < len(path):
            ticket += 1
            heapq.heappush(
                events,
                (arrival_s, ticket, msg_id, seq, hop + 1, size, attempt),
            )
            continue

        if seq < msg.expected or seq in buffered[msg_id]:
            continue

        old_expected = msg.expected
        if seq == msg.expected:
            msg.expected += 1
            while msg.expected in buffered[msg_id]:
                released = buffered[msg_id].pop(msg.expected)
                msg.buffered_bytes -= released
                receiver_buffer[msg.dst] -= released
                msg.expected += 1
        else:
            if attempt == 0:
                out_of_order += 1
            distance = seq - msg.expected
            msg.peak_distance = max(msg.peak_distance, distance)
            max_distance = max(max_distance, distance)
            limit = config.buffer_bytes
            fits = limit is None or msg.buffered_bytes + size <= limit
            if fits:
                buffered[msg_id][seq] = size
                msg.buffered_bytes += size
                receiver_buffer[msg.dst] += size
                msg.peak_buffered_bytes = max(
                    msg.peak_buffered_bytes, msg.buffered_bytes
                )
                max_receiver_buffer = max(
                    max_receiver_buffer, receiver_buffer[msg.dst]
                )
            elif config.overflow_policy == "retransmit":
                dropped += 1
                pending_retries[msg_id].add(seq)
            else:
                raise RuntimeError(
                    "reorder buffer overflow under {!r}".format(
                        config.overflow_policy
                    )
                )

        advanced = msg.expected - old_expected
        if config.overflow_policy == "backpressure" and advanced:
            for _ in range(advanced):
                if msg.next_to_send >= msg.packets:
                    break
                next_seq = msg.next_to_send
                msg.next_to_send += 1
                schedule(msg_id, next_seq, arrival_s + credit_delay_s)

        if msg.expected == msg.packets and msg.completion_s == 0.0:
            msg.completion_s = arrival_s
            completion = max(completion, arrival_s)
            completed += 1

    if completed != len(messages):
        raise RuntimeError(
            "simulation ended with {}/{} messages complete".format(
                completed, len(messages)
            )
        )

    original_packets = sum(msg.packets for msg in messages)
    original_bytes = sum(msg.size_bytes for msg in messages)
    peaks = [msg.peak_buffered_bytes for msg in messages]
    return PacketSimulationResult(
        completion_time_s=completion,
        messages=len(messages),
        original_packets=original_packets,
        transmissions=transmissions,
        retransmissions=retransmissions,
        dropped_arrivals=dropped,
        original_bytes=original_bytes,
        transmitted_bytes=transmitted_bytes,
        out_of_order_packets=out_of_order,
        out_of_order_fraction=out_of_order / original_packets,
        max_reorder_distance_packets=max_distance,
        max_message_reorder_bytes=max(peaks, default=0),
        mean_message_peak_reorder_bytes=float(np.mean(peaks)) if peaks else 0.0,
        max_receiver_reorder_bytes=max_receiver_buffer,
        buffer_bytes=config.buffer_bytes,
        overflow_policy=config.overflow_policy,
        unit_bytes=config.unit_bytes,
    )
