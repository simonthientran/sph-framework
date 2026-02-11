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


def cubic_spline_gradW(r: np.ndarray, h: float, dim: int) -> np.ndarray:
    """
    Gradient of the cubic spline kernel W(r, h) used in the tutorial (Eq. (4)).

    We use the analytical derivative of the same piecewise polynomial:
        q = ||r|| / h
        dW/dr = (dW/dq) * (dq/dr)
    with:
        dq/dr = (1/h) * r/||r||   for r != 0

    The result is a vector of shape (dim,).
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

    # Ensure r has the right shape for the requested dimension.
    if r.shape != (dim,):
        r = r.reshape(dim,)

    rn = float(np.linalg.norm(r))

    # At the origin the kernel is radially symmetric, so gradient is zero.
    if rn == 0.0:
        return np.zeros((dim,), dtype=np.float64)

    q = rn / h

    # Outside compact support => gradient is zero
    if q > 1.0:
        return np.zeros((dim,), dtype=np.float64)

    # d/dq of Eq. (4):
    # for q <= 0.5: 6(q^3 - q^2) + 1  -> 18 q^2 - 12 q
    # for 0.5 < q <= 1: 2(1-q)^3      -> -6(1-q)^2
    if q <= 0.5:
        dW_dq = sigma * (18.0 * q * q - 12.0 * q)
    else:
        dW_dq = sigma * (-6.0 * (1.0 - q) ** 2)

    # chain rule: dW/dr = dW/dq * (1/h) * r/||r||
    return (dW_dq / h) * (r / rn)

def cubic_spline_gradW(r: np.ndarray, h: float, dim: int) -> np.ndarray:
    """
    Gradient of the cubic spline kernel (Eq. (4)).

    We use the analytical derivative of the *same* piecewise polynomial
    defined in Eq. (4) and apply the chain rule:
        q = ||r|| / h
        dW/dr = (dW/dq) * (dq/dr)
    with:
        dq/dr = (1/h) * r/||r||   (for r != 0)

    Reference:
    - Kernel definition: Eq. (4) in "SPH Techniques for the Physics Based Simulation of Fluids and Solids".
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

    if rn == 0.0:
        return np.zeros((dim,), dtype=np.float64)

    q = rn / h

    # Outside compact support => gradient is zero
    if q > 1.0:
        return np.zeros((dim,), dtype=np.float64)

    # d/dq of Eq. (4):
    # for q <= 0.5: 6(q^3 - q^2) + 1  -> 18 q^2 - 12 q
    # for 0.5 < q <= 1: 2(1-q)^3      -> -6(1-q)^2
    if q <= 0.5:
        dW_dq = sigma * (18.0 * q * q - 12.0 * q)
    else:
        dW_dq = sigma * (-6.0 * (1.0 - q) ** 2)

    # chain rule: dW/dr = dW/dq * (1/h) * r/||r||
    return (dW_dq / h) * (r / rn)
