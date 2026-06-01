"""Internal validation-boundary contracts shared by DFSPH and CUDA replay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sph.neighbor_pairs import NeighborPairs


@dataclass(slots=True)
class CUDAValidationStageInput:
    """Stable validation snapshot for one DFSPH stage input."""

    dt: float
    h: float
    rho0: float
    mass: float
    max_iter: int
    eta: float
    positions: np.ndarray
    velocities: np.ndarray
    densities: np.ndarray
    rho_self: np.ndarray
    rho_ff: np.ndarray
    rho_fb: np.ndarray
    k_factor: np.ndarray
    lambda_prev: np.ndarray
    boundary_velocities: np.ndarray
    pairs: NeighborPairs


@dataclass(slots=True)
class CUDAValidationStageOutput:
    """Stable validation reference for one DFSPH stage output."""

    velocities: np.ndarray
    lambda_final: np.ndarray
    iterations: int
    converged: bool
    metric: float


@dataclass(slots=True)
class CUDAValidationStageSnapshot:
    """Typed producer/consumer validation contract for one DFSPH stage."""

    stage: Literal["cd", "df"]
    stage_input: CUDAValidationStageInput
    stage_output: CUDAValidationStageOutput | None = None
