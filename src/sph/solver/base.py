"""Common solver interfaces used by the simulation orchestration layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sph.core.simulator import SimConfig
from sph.core.state import ParticleState


class SolverBase(ABC):
    """Minimal solver interface expected by :mod:`sph.core.simulation`."""

    def __init__(self, config: SimConfig):
        self.config = config

    @abstractmethod
    def step(self, state: ParticleState, particle_size: float) -> float:
        """Advance the simulation by one step and return ``dt``."""

