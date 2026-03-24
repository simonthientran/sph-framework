from __future__ import annotations

import numpy as np


def cubic_spline_W(r: np.ndarray, h: float, dim: int) -> float:
    """
    Cubic spline smoothing kernel W(r,h) with compact support q in [0, 1].

    Reference:
    - Document: "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
    - Equation: Eq. (4) (cubic spline kernel definition).

    Parameterization used here:
        q = ||r|| / h
        W(r,h) = sigma_d * piecewise(q)
    with normalization constants:
        sigma1 = 4/(3h), sigma2 = 40/(7*pi*h^2), sigma3 = 8/(pi*h^3)
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

    if q <= 0.5:
        # 6(q^3 - q^2) + 1
        return sigma * (6.0 * (q ** 3 - q ** 2) + 1.0)
    if q <= 1.0:
        # 2(1 - q)^3
        return sigma * (2.0 * (1.0 - q) ** 3)
    return 0.0


def cubic_spline_gradW(r: np.ndarray, h: float, dim: int) -> np.ndarray:
    """
    Gradient of the cubic spline kernel ∇W(r,h).

    Reference:
    - Document: "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
    - Kernel definition: Eq. (4). The derivative is taken analytically from the same piecewise polynomial.

    We apply the chain rule:
        q = ||r|| / h
        ∇W = (dW/dq) * (1/h) * r/||r||     for r != 0
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

    r = np.asarray(r, dtype=np.float64)
    rn = float(np.linalg.norm(r))

    # At r = 0 the direction is undefined; for symmetric kernels we set gradient to zero.
    if rn == 0.0:
        return np.zeros((dim,), dtype=np.float64)

    q = rn / h

    # Compact support: outside q>1 => gradient is zero
    if q > 1.0:
        return np.zeros((dim,), dtype=np.float64)

    # Eq. (4) piecewise polynomial derivatives w.r.t q:
    # for q <= 0.5: 6(q^3 - q^2) + 1  -> d/dq = 18 q^2 - 12 q
    # for 0.5 < q <= 1: 2(1 - q)^3    -> d/dq = -6(1 - q)^2
    if q <= 0.5:
        dW_dq = sigma * (18.0 * q * q - 12.0 * q)
    else:
        dW_dq = sigma * (-6.0 * (1.0 - q) ** 2)

    return (dW_dq / h) * (r / rn)
