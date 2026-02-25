from __future__ import annotations
import numpy as np

from sph.boundaries.base import BoundaryBase

class WallBoundary(BoundaryBase):
    """
    Implements a structural wall boundary using penalty forces / collision projection.
    Replaces the hardcoded domain constraints.
    """
    def __init__(self, domain_min: list[float], domain_max: list[float], slip_mode: str = "no-slip", 
                 restitution: float = 0.0, eps: float = 1e-4):
        self.domain_min = np.array(domain_min, dtype=np.float64) if domain_min is not None else None
        self.domain_max = np.array(domain_max, dtype=np.float64) if domain_max is not None else None
        self.slip_mode = slip_mode
        self.friction = 1.0 if slip_mode == "no-slip" else 0.0
        self.restitution = restitution
        self.eps = eps

    def apply_walls(self, state, cfg, debug: bool = False) -> None:
        if self.domain_min is None or self.domain_max is None:
            return

        fluid_ids = state.fluid_indices
        if fluid_ids.size == 0:
            return

        pos = state.pos
        vel = state.vel
        dim = state.dim
        dmin = self.domain_min
        dmax = self.domain_max
        eps = self.eps

        for d in range(dim):
            # Check periodic
            if cfg.periodic_axes and int(d) in set(int(a) for a in cfg.periodic_axes):
                L = float(dmax[d] - dmin[d])
                if L > 0.0:
                    x = pos[fluid_ids, d] - dmin[d]
                    pos[fluid_ids, d] = (x % L) + dmin[d]
                continue

            # MIN face
            mask_lo = pos[fluid_ids, d] < dmin[d]
            if np.any(mask_lo):
                idx = fluid_ids[mask_lo]
                pos[idx, d] = dmin[d] + eps
                n = np.zeros((dim,), dtype=np.float64)
                n[d] = 1.0
                
                v_n = (vel[idx] @ n)
                moving_out = v_n < 0.0
                if np.any(moving_out):
                    ids_move = idx[moving_out]
                    vn = v_n[moving_out][:, None] * n[None, :]
                    vt = vel[ids_move] - vn
                    vn_new = -self.restitution * vn
                    vt_new = (1.0 - self.friction) * vt
                    vel[ids_move] = vn_new + vt_new

            # MAX face
            mask_hi = pos[fluid_ids, d] > dmax[d]
            if np.any(mask_hi):
                idx = fluid_ids[mask_hi]
                pos[idx, d] = dmax[d] - eps
                n = np.zeros((dim,), dtype=np.float64)
                n[d] = -1.0
                
                v_n = (vel[idx] @ n)
                moving_out = v_n < 0.0
                if np.any(moving_out):
                    ids_move = idx[moving_out]
                    vn = v_n[moving_out][:, None] * n[None, :]
                    vt = vel[ids_move] - vn
                    vn_new = -self.restitution * vn
                    vt_new = (1.0 - self.friction) * vt
                    vel[ids_move] = vn_new + vt_new
