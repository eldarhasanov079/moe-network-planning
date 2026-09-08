"""Token deflection arms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import FeederConfig


@dataclass
class DeflectionResult:
    indices: np.ndarray
    chosen_device: np.ndarray
    n: int
    eligible: int = 0
    deflected: int = 0
    cost: float = 0.0
    extra: Dict = field(default_factory=dict)

    @property
    def frac_deflected(self) -> float:
        return self.deflected / self.n if self.n else 0.0

    @property
    def frac_eligible(self) -> float:
        return self.eligible / self.n if self.n else 0.0

    def as_dict(self) -> Dict:
        return {
            "n": self.n,
            "eligible": self.eligible,
            "deflected": self.deflected,
            "frac_deflected": self.frac_deflected,
            "frac_eligible": self.frac_eligible,
            "cost": self.cost,
        }


def _devices(indices: np.ndarray, place: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return place[indices[:, 0]], place[indices[:, 1]]


def _margins(scores: Optional[np.ndarray], n: int) -> np.ndarray:
    if scores is None:
        return np.full(n, np.inf, dtype=np.float64)
    return (scores[:, 0] - scores[:, 1]).astype(np.float64)


def _rewrite_top1(indices: np.ndarray, chosen_device: np.ndarray, place: np.ndarray) -> np.ndarray:
    """Point the primary expert at one that lives on ``chosen_device`` (top-2 if needed)."""
    out = indices.copy()
    top1_dev = place[out[:, 0]]
    top2_dev = place[out[:, 1]]
    move = chosen_device != top1_dev
    use_top2 = move & (top2_dev == chosen_device)
    out[use_top2, 0] = out[use_top2, 1]
    return out


def _online(
    top1: np.ndarray,
    top2: np.ndarray,
    margin: np.ndarray,
    src: np.ndarray,
    tau: float,
    num_devices: int,
    local: bool,
) -> Tuple[np.ndarray, int, int, float]:
    n = len(top1)
    chosen = top1.copy()
    eligible = 0
    moved = 0
    cost = 0.0
    if local:
        load = np.zeros((int(src.max()) + 1 if len(src) else 1, num_devices), dtype=np.float64)
    else:
        load = np.zeros(num_devices, dtype=np.float64)
    for t in range(n):
        d0, d1 = int(top1[t]), int(top2[t])
        mg = float(margin[t])
        L = load[int(src[t])] if local else load
        if d1 != d0 and mg <= tau:
            eligible += 1
            if L[d1] < L[d0]:
                chosen[t] = d1
                moved += 1
                cost += mg
        L[int(chosen[t])] += 1.0
    return chosen, eligible, moved, cost


def _offline(
    top1: np.ndarray,
    top2: np.ndarray,
    margin: np.ndarray,
    num_devices: int,
) -> Tuple[np.ndarray, int, int, float]:
    """Hindsight peak-shave on devices. Upper bound, not deployable."""
    import heapq

    n = len(top1)
    L = np.bincount(top1, minlength=num_devices).astype(np.int64)
    pools: List[List] = [[] for _ in range(num_devices)]
    owner = top1.copy()
    for t in range(n):
        pools[int(top1[t])].append((float(margin[t]), int(top2[t]), t))
    for d in range(num_devices):
        heapq.heapify(pools[d])
    moved = 0
    cost = 0.0
    eligible = int(np.sum((top1 != top2)))
    while True:
        e = int(np.argmax(L))
        pool = pools[e]
        did = False
        while pool:
            mg, d, t = heapq.heappop(pool)
            if L[d] < L[e] - 1:
                L[e] -= 1
                L[d] += 1
                owner[t] = d
                moved += 1
                cost += mg
                did = True
                break
        if not did:
            break
    return owner, eligible, moved, cost


def _puppeteer_choose(
    top1: np.ndarray,
    top2: np.ndarray,
    margin: np.ndarray,
    src: np.ndarray,
    tau: float,
    topology_path: str,
) -> Tuple[np.ndarray, int, int, float]:
    from puppeteer.config import RunConfig
    from puppeteer.planner.router import ClosLeastLoadedRouter
    from puppeteer.planner.state import NetworkState

    config = RunConfig.load(topology_path)
    state = NetworkState(config.topology)
    router = ClosLeastLoadedRouter(config.topology)
    router.prepare(state)

    n = len(top1)
    chosen = top1.copy()
    eligible = 0
    batches: Dict[Tuple[int, int, int], List[int]] = {}
    for t in range(n):
        d0, d1 = int(top1[t]), int(top2[t])
        if d1 != d0 and float(margin[t]) <= tau:
            eligible += 1
            batches.setdefault((int(src[t]), d0, d1), []).append(t)

    moved = 0
    cost = 0.0
    for (s, d0, d1), tokens in sorted(batches.items()):
        pick = router.choose_dest(s, (d0, d1), state)
        route = router.route(s, pick, state)
        state.admit("batch:{}:{}:{}:{}".format(s, d0, d1, pick), route.links)
        for t in tokens:
            if pick != int(top1[t]):
                chosen[t] = pick
                moved += 1
                cost += float(margin[t])
    return chosen, eligible, moved, cost


def apply_deflection(
    cfg: FeederConfig,
    indices: np.ndarray,
    scores: Optional[np.ndarray],
    place: np.ndarray,
    src: np.ndarray,
) -> DeflectionResult:
    n = len(indices)
    top1_dev, top2_dev = _devices(indices, place)
    if cfg.deflection == "none" or scores is None:
        return DeflectionResult(
            indices=indices, chosen_device=top1_dev, n=n
        )

    margin = _margins(scores, n)
    if cfg.deflection == "online_global":
        chosen, elig, moved, cost = _online(
            top1_dev, top2_dev, margin, src, cfg.tau, cfg.num_devices, local=False
        )
    elif cfg.deflection == "online_local":
        chosen, elig, moved, cost = _online(
            top1_dev, top2_dev, margin, src, cfg.tau, cfg.num_devices, local=True
        )
    elif cfg.deflection == "offline":
        chosen, elig, moved, cost = _offline(top1_dev, top2_dev, margin, cfg.num_devices)
    elif cfg.deflection == "puppeteer_choose":
        chosen, elig, moved, cost = _puppeteer_choose(
            top1_dev, top2_dev, margin, src, cfg.tau, cfg.topology
        )
    else:
        raise ValueError("unknown deflection {!r}".format(cfg.deflection))

    rewritten = _rewrite_top1(indices, chosen, place)
    return DeflectionResult(
        indices=rewritten,
        chosen_device=chosen,
        n=n,
        eligible=elig,
        deflected=moved,
        cost=cost,
    )
