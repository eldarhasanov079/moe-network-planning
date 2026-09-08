"""Cache-first loaders. Reuse moe-trace-experiments/common; do not re-download if cached."""

from __future__ import annotations

import sys
from typing import Optional, Tuple

import numpy as np

from .config import EXPERIMENTS, FeederConfig, resolve_layer


def _ensure_common() -> None:
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))


def load_tokens(
    cfg: FeederConfig, checkpoint, layer: Optional[str] = None
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Return ``(scores | None, indices)`` with indices shaped ``(n, top_k)``."""
    _ensure_common()
    layer = layer or cfg.layer
    if cfg.source == "flame":
        from common.flame_loader import load_layer

        scores, indices = load_layer(
            int(checkpoint), layer, cfg.n_tokens, model=cfg.model, verbose=False
        )
        return scores.astype(np.float32), indices.astype(np.int64)
    if cfg.source == "olmoe":
        from common.olmoe_loader import load_checkpoint

        packed = load_checkpoint(str(checkpoint), cfg.n_tokens, verbose=False)
        li = resolve_layer(cfg)
        if li is None:
            raise ValueError("OLMoE requires a numeric layer")
        return None, packed[:, li, :].astype(np.int64)
    raise ValueError("unknown source {!r}".format(cfg.source))
