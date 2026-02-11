from __future__ import annotations

import numpy as np


def cubic_spline_W(r: np.ndarray, h: float, dim: int) -> float:
    """
    Cubic spline smoothing kernel W(r,h) with compact support q in [0, 1].

    This is the parameterization from:
    "SPH Techniques for the Physics Based Simulation of Fluids and Solids"
    (Eq. (4)), where q = ||r||/h and the kernel is zero for q > 1.
    The normalization constants are:
      sigma1 = 4/(3h), sigma2 = 40/(7*pi*h^2), sigma3 = 8/(pi*h^3).
    """
    h = float(h)
    if h <= 0.0:
        raise ValueError("h must be > 0")

    if dim == 1:
        sigma = 4.0 / (3.0 * h)
    elif dim == 2:
        sigma = 40.0 / (7.0 * np.pi * h * h)
    elif dim == 3:
        sigma = 8.0 / (np.pi * h ** 3)
    else:
        raise ValueError("dim must be 1, 2 or 3")

    q = float(np.linalg.norm(r) / h)

    if q < 0.0:
        return 0.0
    if q <= 0.5:
        # 6(q^3 - q^2) + 1
        return sigma * (6.0 * (q ** 3 - q ** 2) + 1.0)
    if q <= 1.0:
        # 2(1 - q)^3
        return sigma * (2.0 * (1.0 - q) ** 3)
    return 0.0
