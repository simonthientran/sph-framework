from __future__ import annotations

import numpy as np

from sph.boundaries.base import BoundaryBase

_FACE_TO_AXIS_SIDE = {
    "xmin": (0, "min"),
    "xmax": (0, "max"),
    "ymin": (1, "min"),
    "ymax": (1, "max"),
    "zmin": (2, "min"),
    "zmax": (2, "max"),
}


class WallBoundary(BoundaryBase):
    """
    Axis-aligned wall collision boundary with configurable slip model.
    """

    def __init__(
        self,
        domain_min: list[float] | np.ndarray,
        domain_max: list[float] | np.ndarray,
        *,
        slip_mode: str = "no-slip",
        restitution: float = 0.0,
        eps: float = 1e-6,
        faces: list[str] | tuple[str, ...] | None = None,
    ):
        self.domain_min = np.asarray(domain_min, dtype=np.float64)
        self.domain_max = np.asarray(domain_max, dtype=np.float64)
        if self.domain_min.shape != self.domain_max.shape:
            raise ValueError("domain_min/domain_max shape mismatch")

        mode = str(slip_mode).lower()
        if mode not in {"no-slip", "free-slip"}:
            raise ValueError(f"unsupported slip mode: {slip_mode!r}")
        self.slip_mode = mode
        self.friction = 1.0 if mode == "no-slip" else 0.0
        self.restitution = float(restitution)
        self.eps = float(max(eps, 0.0))

        dim = int(self.domain_min.shape[0])
        default_faces = ["xmin", "xmax", "ymin", "ymax"] if dim == 2 else ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
        chosen = [str(f).lower() for f in (faces or default_faces)]
        for face in chosen:
            axis_side = _FACE_TO_AXIS_SIDE.get(face)
            if axis_side is None or axis_side[0] >= dim:
                raise ValueError(f"invalid wall face {face!r} for dim={dim}")
        self.faces = tuple(chosen)

    def apply_walls(self, state, cfg, *, debug: bool = False) -> None:
        fluid_ids = state.fluid_indices
        if fluid_ids.size == 0:
            return

        pos = state.pos
        vel = state.vel
        dim = state.dim
        periodic_axes = set(int(a) for a in getattr(cfg, "periodic_axes", ()))
        active_mask = pos[fluid_ids, 0] < 1e8
        if not np.any(active_mask):
            return
        active_fluid_ids = fluid_ids[active_mask]

        for face in self.faces:
            axis, side = _FACE_TO_AXIS_SIDE[face]
            if axis >= dim or axis in periodic_axes:
                continue
            if side == "min":
                mask = pos[active_fluid_ids, axis] < self.domain_min[axis]
                if np.any(mask):
                    self._resolve_face(
                        ids=active_fluid_ids[mask],
                        axis=axis,
                        side=side,
                        pos=pos,
                        vel=vel,
                    )
            else:
                mask = pos[active_fluid_ids, axis] > self.domain_max[axis]
                if np.any(mask):
                    self._resolve_face(
                        ids=active_fluid_ids[mask],
                        axis=axis,
                        side=side,
                        pos=pos,
                        vel=vel,
                    )

    def _resolve_face(self, *, ids: np.ndarray, axis: int, side: str, pos: np.ndarray, vel: np.ndarray) -> None:
        if side == "min":
            pos[ids, axis] = float(self.domain_min[axis] + self.eps)
            n = np.zeros((pos.shape[1],), dtype=np.float64)
            n[axis] = 1.0
        else:
            pos[ids, axis] = float(self.domain_max[axis] - self.eps)
            n = np.zeros((pos.shape[1],), dtype=np.float64)
            n[axis] = -1.0

        v_n = vel[ids] @ n
        moving_out = v_n < 0.0
        if not np.any(moving_out):
            return

        ids_move = ids[moving_out]
        vn = v_n[moving_out][:, None] * n[None, :]
        vt = vel[ids_move] - vn
        vn_new = -self.restitution * vn
        vt_new = (1.0 - self.friction) * vt
        vel[ids_move] = vn_new + vt_new
