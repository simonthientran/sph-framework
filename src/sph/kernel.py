"""
CubicSplineKernel — Fully vectorized SPH kernel.

Reference:
"SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
Equation (4): Cubic spline kernel definition.
"""
from __future__ import annotations

import numpy as np


class CubicSplineKernel:
    """
    Cubic spline smoothing kernel with compact support [0, 2h].

    Fully vectorized: accepts arrays, returns arrays.
    """

    def __init__(self, h: float, dim: int = 2):
        """
        Initialize kernel with smoothing length h.

        Args:
            h: Smoothing length
            dim: Spatial dimension (1, 2, or 3)
        """
        self.h = float(h)
        self.dim = dim
        self.support_radius = 2.0 * h

        # Normalization factor (sigma_d in the reference)
        if dim == 1:
            self.alpha = 2.0 / (3.0 * h)
        elif dim == 2:
            self.alpha = 10.0 / (7.0 * np.pi * h * h)
        elif dim == 3:
            self.alpha = 1.0 / (np.pi * h ** 3)
        else:
            raise ValueError("dim must be 1, 2, or 3")

    def W(self, dist: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate kernel W(r, h) for distance values.

        Args:
            dist: Distance ||r|| or array of distances, shape (N,)

        Returns:
            Kernel values W, same shape as input
        """
        # Handle scalar input
        scalar_input = np.isscalar(dist)
        if scalar_input:
            dist = np.array([dist], dtype=np.float64)
        else:
            dist = np.asarray(dist, dtype=np.float64)

        q = dist / self.h
        W = np.zeros_like(q, dtype=np.float64)

        mask1 = q < 1.0
        q1 = q[mask1]
        W[mask1] = self.alpha * (1.0 - 1.5 * q1 * q1 + 0.75 * q1 * q1 * q1)

        mask2 = (q >= 1.0) & (q < 2.0)
        q2 = q[mask2]
        t = 2.0 - q2
        W[mask2] = self.alpha * 0.25 * t * t * t

        return W[0] if scalar_input else W

    def grad_W(self, r_ij: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """
        Evaluate kernel gradient ∇W(r, h).

        Args:
            r_ij: Displacement vectors r_i - r_j, shape (N, dim)
            dist: Distances ||r_ij||, shape (N,)

        Returns:
            Gradient vectors ∇W, shape (N, dim)
        """
        r_ij = np.asarray(r_ij, dtype=np.float64)
        dist = np.asarray(dist, dtype=np.float64)

        # Ensure r_ij is 2D
        if r_ij.ndim == 1:
            r_ij = r_ij[np.newaxis, :]
            dist = np.array([dist])

        n_pairs = r_ij.shape[0]
        grad_W = np.zeros((n_pairs, self.dim), dtype=np.float64)

        # Avoid division by zero for dist = 0
        nonzero_mask = dist > 0.0
        if not np.any(nonzero_mask):
            return grad_W

        dist_nz = dist[nonzero_mask]
        r_nz = r_ij[nonzero_mask]

        q = dist_nz / self.h

        # Compact support: q >= 2.0 => gradient is zero
        valid_mask = q < 2.0
        if not np.any(valid_mask):
            return grad_W

        q_valid = q[valid_mask]
        r_valid = r_nz[valid_mask]
        dist_valid = dist_nz[valid_mask]

        # Piecewise derivative dW/dq
        dW_dq = np.zeros_like(q_valid, dtype=np.float64)

        mask1 = q_valid < 1.0
        dW_dq[mask1] = self.alpha * (-3.0 * q_valid[mask1] + 2.25 * q_valid[mask1] ** 2)

        mask2 = (q_valid >= 1.0)
        t = 2.0 - q_valid[mask2]
        dW_dq[mask2] = self.alpha * (-0.75 * t * t)

        # Compute gradient: (dW/dq) * (1/h) * r/||r||
        grad_valid = (dW_dq[:, np.newaxis] / self.h) * (r_valid / dist_valid[:, np.newaxis])

        # Place results back
        nonzero_indices = np.where(nonzero_mask)[0]
        valid_indices = nonzero_indices[valid_mask]
        grad_W[valid_indices] = grad_valid

        return grad_W
