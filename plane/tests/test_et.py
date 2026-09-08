from pathlib import Path

import numpy as np

from moe_feeder.chakra import matrix_from_chakra
from moe_feeder.cli import main
from moe_feeder.et import write_comm_et, write_phased_et
from puppeteer.io.chakra.reader import ChakraProtobufReader


def test_write_comm_et_round_trips(tmp_path: Path):
    M = np.array(
        [
            [0, 3, 0],
            [1, 0, 2],
            [0, 4, 0],
        ],
        dtype=float,
    )
    n = write_comm_et(M, str(tmp_path), name="decided", bytes_per_slot=10)
    assert n == 3
    assert (tmp_path / "comm_groups.json").is_file()
    for rank in range(3):
        assert (tmp_path / "et" / "decided.{}.et".format(rank)).is_file()
    graph = ChakraProtobufReader().read(str(tmp_path))
    assert graph.num_ranks == 3

    recovered, stats = matrix_from_chakra(tmp_path)
    np.testing.assert_array_equal(recovered, M.astype(np.int64) * 10)
    assert stats["off_rank_flows"] == 4
    assert stats["off_rank_bytes"] == 100


def test_write_phased_et(tmp_path: Path):
    disp = np.array([[0, 4], [2, 0]], dtype=float)
    comb = disp.T
    n = write_phased_et([disp, comb], str(tmp_path), name="fwd", bytes_per_slot=8)
    assert n == 2
    assert (tmp_path / "et" / "fwd.0.et").is_file()
    assert (tmp_path / "et" / "fwd.1.et").is_file()


def test_profile_et_cli_produces_byte_reservation(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    out = tmp_path / "profile"
    M0 = np.array([[0, 10], [20, 0]], dtype=np.int64)
    M1 = np.array([[0, 14], [22, 0]], dtype=np.int64)
    write_comm_et(M0, str(first), name="first")
    write_comm_et(M1, str(second), name="second")

    assert main([
        "profile-et",
        "--trace", str(first),
        "--trace", str(second),
        "--reservation", "mean",
        "-o", str(out),
    ]) == 0
    np.testing.assert_array_equal(
        np.load(out / "reservation_bytes.npy"),
        np.array([[0, 12], [21, 0]], dtype=np.float64),
    )
