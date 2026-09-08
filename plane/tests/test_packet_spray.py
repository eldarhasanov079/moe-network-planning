import numpy as np
import pytest

from puppeteer.config import RunConfig

from moe_feeder.config import DEFAULT_TOPOLOGY
from moe_feeder.packet_spray import (
    PacketModelConfig,
    route_table,
    simulate_phase,
)


@pytest.fixture(scope="module")
def topology():
    return RunConfig.load(str(DEFAULT_TOPOLOGY)).topology


def _busy_matrix() -> np.ndarray:
    rng = np.random.default_rng(0)
    matrix = rng.integers(1, 10, size=(8, 8), dtype=np.int64) * 1_000
    np.fill_diagonal(matrix, 0)
    return matrix


def test_one_path_serializes_and_completes_after_last_packet(topology):
    matrix = np.zeros((8, 8), dtype=np.int64)
    matrix[0, 2] = 2_000
    routes = route_table(topology, "ecmp", seed=0)

    result = simulate_phase(
        matrix,
        topology,
        routes,
        PacketModelConfig(unit_bytes=1_000),
    )

    serialization = 1_000 * 8 / 100e9
    expected = 4 * (serialization + 500e-9) + serialization
    assert result.completion_time_s == pytest.approx(expected)
    assert result.original_packets == 2
    assert result.transmissions == 2
    assert result.out_of_order_packets == 0
    assert result.max_message_reorder_bytes == 0


def test_spray_records_out_of_order_arrivals(topology):
    result = simulate_phase(
        _busy_matrix(),
        topology,
        route_table(topology, "spray"),
        PacketModelConfig(unit_bytes=1_000),
    )

    assert result.out_of_order_packets > 0
    assert result.max_reorder_distance_packets > 0
    assert result.max_message_reorder_bytes > 0
    assert result.max_receiver_reorder_bytes >= result.max_message_reorder_bytes


def test_credit_backpressure_respects_per_message_buffer(topology):
    matrix = _busy_matrix()
    routes = route_table(topology, "spray")
    unbounded = simulate_phase(
        matrix,
        topology,
        routes,
        PacketModelConfig(unit_bytes=1_000),
    )
    bounded = simulate_phase(
        matrix,
        topology,
        routes,
        PacketModelConfig(
            unit_bytes=1_000,
            overflow_policy="backpressure",
            buffer_bytes=2_000,
            credit_rtt_us=6.0,
        ),
    )

    assert bounded.max_message_reorder_bytes <= 2_000
    assert bounded.dropped_arrivals == 0
    assert bounded.retransmissions == 0
    assert bounded.completion_time_s >= unbounded.completion_time_s


def test_finite_buffer_selective_recovery_retransmits(topology):
    result = simulate_phase(
        _busy_matrix(),
        topology,
        route_table(topology, "spray"),
        PacketModelConfig(
            unit_bytes=1_000,
            overflow_policy="retransmit",
            buffer_bytes=0,
            retransmit_timeout_us=10.0,
        ),
    )

    assert result.max_message_reorder_bytes == 0
    assert result.dropped_arrivals > 0
    assert result.retransmissions == result.dropped_arrivals
    assert result.transmissions > result.original_packets
    assert result.transmitted_bytes > result.original_bytes

