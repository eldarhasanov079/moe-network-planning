"""Token-level replay against a frozen reservation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .config import FeederConfig
from .deflection import _devices, _margins


@dataclass
class ReplayStats:
    n: int = 0
    would_overflow: int = 0
    deflected_to_fit: int = 0
    overflow_kept: int = 0
    overflow_no_alt: int = 0
    dropped: int = 0
    sent: Optional[np.ndarray] = None
    extra: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        n = self.n or 1
        return {
            "n": self.n,
            "would_overflow": self.would_overflow,
            "deflected_to_fit": self.deflected_to_fit,
            "overflow_kept": self.overflow_kept,
            "overflow_no_alt": self.overflow_no_alt,
            "dropped": self.dropped,
            "frac_deflected": self.deflected_to_fit / n,
            "frac_dropped": self.dropped / n,
            "frac_overflow": self.would_overflow / n,
        }


def replay_against_plan(
    cfg: FeederConfig,
    indices: np.ndarray,
    scores: Optional[np.ndarray],
    place: np.ndarray,
    src: np.ndarray,
    reservation_slots: np.ndarray,
    *,
    deflect: bool = True,
    drop: bool = False,
) -> ReplayStats:
    """Walk tokens in order. ``reservation_slots`` is the frozen envelope in slots."""
    top1, top2 = _devices(indices, place)
    margin = _margins(scores, len(indices))
    sent = np.zeros_like(reservation_slots, dtype=np.float64)
    stats = ReplayStats(n=len(indices))
    D = reservation_slots.shape[0]

    for t in range(len(indices)):
        s = int(src[t])
        d0, d1 = int(top1[t]), int(top2[t])
        if not (0 <= s < D and 0 <= d0 < D):
            continue
        fits = sent[s, d0] + 1.0 <= reservation_slots[s, d0] + 1e-9
        if fits:
            sent[s, d0] += 1.0
            continue
        stats.would_overflow += 1
        low = float(margin[t]) <= cfg.tau
        if deflect and low and d1 != d0 and sent[s, d1] + 1.0 <= reservation_slots[s, d1] + 1e-9:
            sent[s, d1] += 1.0
            stats.deflected_to_fit += 1
            continue
        if drop:
            stats.dropped += 1
            if deflect and low:
                stats.overflow_no_alt += 1
            continue
        sent[s, d0] += 1.0
        if low and deflect:
            stats.overflow_no_alt += 1
        else:
            stats.overflow_kept += 1

    stats.sent = sent
    return stats
