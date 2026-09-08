"""Knobs for the feeder. None of these leak into Puppeteer's Clos event loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

RESERVATION_POLICIES = ("uniform", "mean", "p95", "p99", "worst", "volume")
DEFLECTION_ARMS = ("none", "online_global", "online_local", "offline", "puppeteer_choose")
TRAFFIC_LENSES = ("membership", "primary")
FREEZE_POLICIES = ("fixed_M", "stable", "reprofile_on_overflow")

WORKSPACE = Path(__file__).resolve().parents[2]
EXPERIMENTS = WORKSPACE / "moe-trace-experiments"
DEFAULT_TOPOLOGY = Path(__file__).resolve().parents[1] / "configs" / "topology_8gpu_clos.yaml"


@dataclass
class FeederConfig:
    source: str = "flame"  # flame | olmoe
    model: str = "flame-moe-290m"
    layer: str = "layer_02"
    n_tokens: int = 250_000
    num_devices: int = 8
    num_src_ranks: int = 8
    num_experts: int = 64
    placement: str = "contiguous"
    sharding: str = "contiguous"
    dedup_device: bool = False
    traffic_lens: str = "membership"
    deflection: str = "none"
    tau: float = 0.03
    reservation: str = "mean"
    freeze: str = "fixed_M"
    window: int = 4
    cosine_gate: float = 0.99
    overflow_reprofile: float = 0.05
    bytes_per_slot: int = 2048
    topology: str = str(DEFAULT_TOPOLOGY)
    seed: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def flame_checkpoints(model: str) -> List[int]:
    import sys

    sys.path.insert(0, str(EXPERIMENTS))
    import config as exp_config  # noqa: E402

    return list(exp_config.FLAME_MODELS[model]["checkpoints"])


def flame_layers(model: str) -> List[str]:
    import sys

    sys.path.insert(0, str(EXPERIMENTS))
    import config as exp_config  # noqa: E402

    return list(exp_config.FLAME_MODELS[model]["layers"])


def olmoe_checkpoints() -> List[str]:
    return ["5000", "120000", "245000", "490000", "final"]


def olmoe_layers() -> List[str]:
    return ["layer_{}".format(i) for i in range(16)]


def window_and_heldout(
    checkpoints: List, window: int, freeze: str = "fixed_M"
) -> tuple:
    """Split a checkpoint list into a profile window and a held-out tail."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if window >= len(checkpoints):
        return list(checkpoints[:-1]) or list(checkpoints), checkpoints[-1]
    return list(checkpoints[:window]), checkpoints[window]


def resolve_layer(cfg: FeederConfig) -> Optional[int]:
    """OLMoE layers are integer indices; FLAME uses names like layer_02."""
    if cfg.source == "olmoe":
        if cfg.layer.startswith("layer_"):
            return int(cfg.layer.split("_")[1])
        return int(cfg.layer)
    return None
