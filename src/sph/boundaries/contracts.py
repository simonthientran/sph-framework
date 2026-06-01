"""Generic internal boundary-representation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class BoundaryRepresentationSource:
    """Describes one scene boundary source independently of its representation."""

    boundary_type: str
    source_name: str


class BoundaryRepresentation(ABC):
    """
    Internal boundary-representation seam.

    Runtime consumers should depend on this contract and request the form they
    currently need from a representation, rather than assuming every boundary is
    directly described by mesh-sampled particles.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Representation kind, e.g. ``mesh_particles`` or future ``sdf``."""

    @property
    @abstractmethod
    def sources(self) -> tuple[BoundaryRepresentationSource, ...]:
        """Boundary sources that produced this representation."""

    @abstractmethod
    def project_particle_positions(
        self,
        dim: int,
        spacing: float,
        deduplicate: bool = False,
    ) -> np.ndarray:
        """
        Project the representation into the current particle-boundary bridge.

        This keeps the present runtime behavior intact while allowing future
        representations to provide different backing data.
        """
