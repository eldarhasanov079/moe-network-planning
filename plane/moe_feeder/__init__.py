"""PLANE: router or Chakra traces → policy matrix → network plan."""

from .chakra import matrix_from_chakra
from .config import FeederConfig
from .deflection import apply_deflection
from .matrix import build_matrix, profile_window
from .policy import collapse_history
from .runtime import replay_against_plan

__version__ = "0.1.0"

__all__ = [
    "FeederConfig",
    "apply_deflection",
    "build_matrix",
    "collapse_history",
    "matrix_from_chakra",
    "profile_window",
    "replay_against_plan",
]
