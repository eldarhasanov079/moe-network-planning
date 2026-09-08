import numpy as np

from moe_feeder.config import DEFAULT_TOPOLOGY
from moe_feeder.netlinks import LinkModel
from moe_feeder.planner import paths_from_plan, plan_matrix


def test_link_accounting_and_bounds():
    model = LinkModel(str(DEFAULT_TOPOLOGY))
    assert len(model.links) == 48
    rng = np.random.default_rng(0)
    E = rng.uniform(1000, 2000, size=(8, 8))
    np.fill_diagonal(E, 0)
    M = E * rng.uniform(0.9, 1.1, size=E.shape)
    np.fill_diagonal(M, 0)
    payload = plan_matrix(E, str(DEFAULT_TOPOLOGY), bytes_per_slot=2048)
    from puppeteer.config import RunConfig

    paths = paths_from_plan(payload["plan"], RunConfig.load(str(DEFAULT_TOPOLOGY)).topology)
    frozen = model.frozen_routes(paths)
    for (s, d), opts in frozen.items():
        host_s, host_d = s // 2, d // 2
        assert (len(opts[0][0]) == 0) == (host_s == host_d)
    spray = model.spray_routes()
    ecmp = model.ecmp_routes(0)
    for routes in (frozen, spray, ecmp):
        load = model.link_bytes(M, routes, 2048)
        expect = sum(float(M[s, d]) * 2048 * sum(w * len(l) for l, w in routes[(s, d)]) for (s, d) in routes)
        assert abs(sum(load.values()) - expect) < 1e-3 * max(expect, 1)
    bud = model.budget(M, E, frozen, ecmp, 2048)
    assert 0.0 <= bud["coverage"] <= 1.0
    assert bud["lb_total_s"] >= bud["lb_reserved_s"] > 0
    assert bud["predicted_extension_s"] >= 0
    same = model.budget(E, E, frozen, ecmp, 2048)
    assert same["coverage"] == 1.0 and same["predicted_extension_s"] == 0.0
