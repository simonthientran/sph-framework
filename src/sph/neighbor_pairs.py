"""
Shared neighbor pair container to keep interaction data backend-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NeighborPairs:
    """Separates fluid-fluid and fluid-boundary neighbor data."""

    ff_i: np.ndarray
    ff_j: np.ndarray
    ff_r: np.ndarray
    ff_dist: np.ndarray
    fb_i: np.ndarray
    fb_j: np.ndarray
    fb_r: np.ndarray
    fb_dist: np.ndarray

    @classmethod
    def empty(cls, dim: int) -> "NeighborPairs":
        zero_idx = np.zeros(0, dtype=np.int32)
        zero_vec = np.zeros((0, dim), dtype=np.float64)
        zero_dist = np.zeros(0, dtype=np.float64)
        return cls(
            zero_idx,
            zero_idx,
            zero_vec,
            zero_dist,
            zero_idx,
            zero_idx,
            zero_vec,
            zero_dist,
        )
