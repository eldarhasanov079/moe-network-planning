"""Shared netmodel toolkit -- expert->device placement and token->source-rank sharding."""

from __future__ import annotations

import numpy as np

PLACEMENT_SCHEMES = ("contiguous", "roundrobin", "random")
SHARDING_SCHEMES = ("contiguous", "strided")


def expert_to_device(
    num_experts: int,
    num_devices: int,
    scheme: str = "contiguous",
    seed: int = 0,
) -> np.ndarray:
    """Return an int array ``place`` of shape (num_experts,): ``place[e]`` = device id."""
    if num_experts % num_devices != 0:
        raise ValueError(f"num_experts ({num_experts}) must be divisible by "
                         f"num_devices ({num_devices}).")
    epd = num_experts // num_devices

    if scheme == "contiguous":
        return np.repeat(np.arange(num_devices), epd).astype(np.int64)
    if scheme == "roundrobin":
        return (np.arange(num_experts) % num_devices).astype(np.int64)
    if scheme == "random":
        rng = np.random.default_rng(seed)
        slots = np.repeat(np.arange(num_devices), epd)
        rng.shuffle(slots)
        return slots.astype(np.int64)
    raise ValueError(f"unknown placement scheme {scheme!r}; choose from {PLACEMENT_SCHEMES}")


def token_source_ranks(
    n_tokens: int,
    num_src_ranks: int,
    scheme: str = "contiguous",
) -> np.ndarray:
    """Return an int array ``src`` of shape (n_tokens,): ``src[t]`` = source rank id."""
    if num_src_ranks <= 0:
        raise ValueError("num_src_ranks must be positive")
    if scheme == "contiguous":
        per = int(np.ceil(n_tokens / num_src_ranks))
        return (np.arange(n_tokens) // per).astype(np.int64)
    if scheme == "strided":
        return (np.arange(n_tokens) % num_src_ranks).astype(np.int64)
    raise ValueError(f"unknown sharding scheme {scheme!r}; choose from {SHARDING_SCHEMES}")
