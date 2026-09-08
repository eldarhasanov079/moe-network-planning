from puppeteer.config import RunConfig
from puppeteer.planner.router import ClosLeastLoadedRouter, FrozenPathRouter, build_router
from puppeteer.planner.state import NetworkState
from moe_feeder.config import DEFAULT_TOPOLOGY


def test_choose_dest_prefers_on_device_then_lighter_path():
    config = RunConfig.load(str(DEFAULT_TOPOLOGY))
    state = NetworkState(config.topology)
    router = ClosLeastLoadedRouter(config.topology)
    router.prepare(state)
    assert router.choose_dest(0, (0, 1), state) == 0
    pick = router.choose_dest(0, (1, 2), state)
    assert pick in (1, 2)


def test_frozen_path_router_uses_map():
    config = RunConfig.load(str(DEFAULT_TOPOLOGY))
    inner = ClosLeastLoadedRouter(config.topology)
    state = NetworkState(config.topology)
    inner.prepare(state)
    live = inner.route(0, 3, state)
    frozen = FrozenPathRouter(inner, {(0, 3): live})
    assert frozen.route(0, 3, state).links == live.links
    built = build_router(config.topology, frozen_paths={(0, 3): live})
    built.prepare(NetworkState(config.topology))
    assert built.route(0, 3, state).links == live.links
