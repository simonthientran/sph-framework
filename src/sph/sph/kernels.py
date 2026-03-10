from __future__ import annotations

import numpy as np


def kernel_constant(dim: int, h: float) -> float:
    """
    Cubic spline kernel normalization constant sigma for W(r,h) = sigma * f(q).

    Convention (Monaghan cubic spline, support radius = 2h):
      2D: W(r,h) = (10 / (7*pi*h^2)) * f(q)   -> sigma = 10/(7*pi*h^2)
      3D: W(r,h) = (1 / (pi*h^3)) * f(q)      -> sigma = 1/(pi*h^3)

    Neighbor search must use support_radius = 2h.
    """
    h = float(h)
    if h <= 0.0:
        raise ValueError("h must be > 0")
    if dim == 1:
        return 2.0 / (3.0 * h)
    if dim == 2:
        return 10.0 / (7.0 * np.pi * h * h)
    if dim == 3:
        return 1.0 / (np.pi * h**3)
    raise ValueError(f"dim must be 1, 2 or 3, got {dim}")


def cubic_spline_W(r: np.ndarray, h: float, dim: int) -> float:
    """
    Cubic spline smoothing kernel W(r,h) with support radius 2h.
    
    Parameters:
        r: vector to neighbor (pos_i - pos_j)
        h: smoothing length (support radius = 2h)
        dim: dimension (1, 2, or 3)
        
    Normalization constants (Monaghan cubic spline):
        1D: 2/(3h)
        2D: 10/(7*pi*h^2)
        3D: 1/(pi*h^3)
    """
    sigma = kernel_constant(dim, h)
    q = float(np.linalg.norm(r) / h)

    if q <= 1.0:
        # 1 - (3/2)q^2 + (3/4)q^3
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q ** 3)
    elif q <= 2.0:
        # (1/4)(2-q)^3
        return sigma * 0.25 * (2.0 - q) ** 3
    
    return 0.0


def cubic_spline_gradW(r: np.ndarray, h: float, dim: int) -> np.ndarray:
    """
    Gradient of the cubic spline kernel ∇W(r,h).
    
    q = r/h
    ∇W = (dW/dq) * (1/h) * (r/|r|)
    """
    sigma = kernel_constant(dim, h)
    r = np.asarray(r, dtype=np.float64)
    rn = float(np.linalg.norm(r))

    if rn == 0.0:
        return np.zeros((dim,), dtype=np.float64)

    q = rn / h

    if q > 2.0:
        return np.zeros((dim,), dtype=np.float64)

    # dW/dq
    if q <= 1.0:
        # d/dq (1 - 1.5q^2 + 0.75q^3) = -3q + 2.25q^2
        dW_dq = sigma * (-3.0 * q + 2.25 * q * q)
    else:
        # d/dq (0.25(2-q)^3) = 0.75(2-q)^2 * (-1)
        dW_dq = sigma * (-0.75 * (2.0 - q) ** 2)

    return (dW_dq / h) * (r / rn)
