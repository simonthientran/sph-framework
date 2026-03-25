"""
TimeStep base class for SPH time integration schemes.

Inspired by SPlisHSPlasH TimeStep architecture.
"""
from __future__ import annotations

from sph.fluid_model import FluidModel, BoundaryModel


class TimeStep:
    """
    Base class for time integration schemes.

    Subclasses implement specific SPH solvers (WCSPH, PCISPH, etc.).
    """

    def step(self, fluid: FluidModel, boundary: BoundaryModel | None, dt: float):
        """
        Advance the simulation by one time step.

        Args:
            fluid: Fluid model
            boundary: Boundary model (or None if no boundaries)
            dt: Time step size
        """
        raise NotImplementedError("Subclasses must implement step()")

    def _compute_densities(self, fluid: FluidModel, boundary: BoundaryModel | None):
        """Compute particle densities using SPH summation."""
        raise NotImplementedError

    def _clear_accelerations(self, fluid: FluidModel):
        """Reset accelerations to zero before force computation."""
        fluid.clear_accelerations()

    def _compute_non_pressure_forces(self, fluid: FluidModel):
        """Compute all non-pressure forces (viscosity, body forces, etc.)."""
        raise NotImplementedError
