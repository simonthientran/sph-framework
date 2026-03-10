import numpy as np

from sph.sph.kernels import kernel_constant


def compute_particle_mass(dx: float, rho0: float, dim: int) -> float:
    """
    Compute particle mass based on spacing (dx), reference density (rho0), and dimension.

    Rules:
      2D: m = rho0 * dx^2
      3D: m = rho0 * dx^3
    """
    if dim == 2:
        return float(rho0 * (dx**2))
    elif dim == 3:
        return float(rho0 * (dx**3))
    else:
        raise ValueError(f"Unsupported dimension for mass computation: {dim}")


def get_kernel_constants(dim: int, h: float) -> dict:
    """
    Return kernel normalization constants and properties for diagnostics.
    
    h = smoothing length
    """
    norm = kernel_constant(dim, h) if dim in (2, 3) else 0.0
    return {
        "normalization_constant": norm,
        "support_radius": 2.0 * h,
        "smoothing_length": h
    }
