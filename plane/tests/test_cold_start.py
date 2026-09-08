import numpy as np

from moe_feeder.cold_start import TOP_K, topology_only_uniform
from moe_feeder.claim_v1 import source_bundle


def test_topology_only_uniform_uses_job_shape_not_trace_data():
    for source in ("flame", "olmoe"):
        bundle = source_bundle(source)
        matrix = topology_only_uniform(bundle)
        expected_total = bundle["n_tokens"] * TOP_K[source]
        assert matrix.shape == (8, 8)
        assert np.allclose(matrix, matrix[0, 0])
        assert matrix.sum() == expected_total
        assert np.allclose(matrix.sum(axis=1), expected_total / 8)


def test_uniform_capacity_factor_scales_all_cells():
    bundle = source_bundle("flame")
    base = topology_only_uniform(bundle)
    padded = topology_only_uniform(bundle, 1.25)
    assert np.allclose(padded, base * 1.25)
