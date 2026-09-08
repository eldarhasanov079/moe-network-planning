"""Shared matplotlib styling so every figure looks consistent and clean."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt

PALETTE = [
    "#2c6fbb", "#d1495b", "#33a02c", "#ff7f00", "#6a3d9a",
    "#b15928", "#1f9e89", "#e31a1c", "#7f7f7f", "#17becf",
]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "figure.autolayout": True,
    })


def save(fig, path, *, close: bool = True) -> None:
    """Save a figure and (by default) close it to free memory."""
    fig.savefig(path, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    print(f"  saved figure -> {path}", flush=True)
