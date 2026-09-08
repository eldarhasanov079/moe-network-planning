"""Dispatch matrices from (possibly deflected) token indices."""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import numpy as np

from .config import EXPERIMENTS, FeederConfig
from .loaders import load_tokens


def _common():
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    from common.placement import expert_to_device, token_source_ranks
    from common.traffic_matrix import build_dispatch_matrix

    return expert_to_device, token_source_ranks, build_dispatch_matrix


def maps(cfg: FeederConfig, n_tokens: int) -> Tuple[np.ndarray, np.ndarray]:
    expert_to_device, token_source_ranks, _ = _common()
    place = expert_to_device(
        cfg.num_experts, cfg.num_devices, cfg.placement, seed=cfg.seed
    )
    src = token_source_ranks(n_tokens, cfg.num_src_ranks, cfg.sharding)
    return place, src


def indices_for_lens(indices: np.ndarray, traffic_lens: str) -> np.ndarray:
    if traffic_lens == "primary":
        return indices[:, :1]
    if traffic_lens == "membership":
        return indices
    raise ValueError("unknown traffic_lens {!r}".format(traffic_lens))


def build_matrix(
    cfg: FeederConfig,
    indices: np.ndarray,
    place: Optional[np.ndarray] = None,
    src: Optional[np.ndarray] = None,
) -> np.ndarray:
    _, _, build_dispatch_matrix = _common()
    if place is None or src is None:
        place, src = maps(cfg, len(indices))
    used = indices_for_lens(indices, cfg.traffic_lens)
    return build_dispatch_matrix(
        used,
        place,
        src,
        cfg.num_src_ranks,
        cfg.num_devices,
        dedup_device=cfg.dedup_device,
    )


def profile_window(
    cfg: FeederConfig, checkpoints: List
) -> Tuple[List[np.ndarray], List]:
    """One dispatch matrix per checkpoint (after optional deflection)."""
    from .deflection import apply_deflection

    history = []
    for ckpt in checkpoints:
        scores, indices = load_tokens(cfg, ckpt)
        place, src = maps(cfg, len(indices))
        result = apply_deflection(cfg, indices, scores, place, src)
        history.append(build_matrix(cfg, result.indices, place, src))
    return history, checkpoints
