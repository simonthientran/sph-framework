"""Numba-CUDA kernels for device-side pair preparation and pairwise SPH passes.

Supports both 2D (ndim=2) and 3D (ndim=3) via an explicit ``ndim`` kernel
argument.  All geometry buffers carry separate dx, dy, dz arrays; in 2D the
dz array is present but unused (its allocation is harmless).

The gradient device function ``_cubic_spline_grad_3d`` returns (gx, gy, gz)
for both 2D and 3D — in 2D the gz component is always 0.0.
"""
from __future__ import annotations

import math as _math
from dataclasses import dataclass

import numpy as np
from numba import cuda


@dataclass(slots=True)
class DevicePairIndexBuffers:
    """Device-side pair index buffers mirrored from canonical NeighborPairs."""

    ff_i: object
    ff_j: object
    fb_i: object
    fb_j: object

    @property
    def ff_count(self) -> int:
        return int(self.ff_i.size)

    @property
    def fb_count(self) -> int:
        return int(self.fb_i.size)


@dataclass(slots=True)
class DevicePairGeometryBuffers:
    """Device-side pair geometry reused across multiple pairwise kernels."""

    ff_dx: object
    ff_dy: object
    ff_dz: object
    ff_dist: object
    fb_dx: object
    fb_dy: object
    fb_dz: object
    fb_dist: object


@dataclass(slots=True)
class DeviceDensityBuffers:
    """Device-side density decomposition matching the CPU solver layout."""

    rho_self: object
    rho_ff: object
    rho_fb: object
    density: object


@dataclass(slots=True)
class DeviceKFactorBuffers:
    """Device-side DFSPH k-factor intermediates."""

    sum_grad_x: object
    sum_grad_y: object
    sum_grad_z: object
    sum_grad_sq: object
    k_factor: object


def _kernel_alpha_2d(h: float) -> float:
    return 10.0 / (7.0 * np.pi * h * h)


def _kernel_alpha_3d(h: float) -> float:
    return 1.0 / (np.pi * h * h * h)


@cuda.jit(device=True)
def _cubic_spline_w(dist: float, h: float, alpha: float) -> float:
    q = dist / h
    if q < 1.0:
        return alpha * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    if q < 2.0:
        t = 2.0 - q
        return alpha * 0.25 * t * t * t
    return 0.0


@cuda.jit(device=True)
def _cubic_spline_grad_3d(
    dist: float, dx: float, dy: float, dz: float, h: float, alpha: float,
) -> tuple:
    """Return (gx, gy, gz).  In 2D calls, dz should be 0.0."""
    if dist <= 1.0e-12:
        return 0.0, 0.0, 0.0
    q = dist / h
    if q < 1.0:
        dw_dq = alpha * (-3.0 * q + 2.25 * q * q)
    elif q < 2.0:
        t = 2.0 - q
        dw_dq = alpha * (-0.75 * t * t)
    else:
        return 0.0, 0.0, 0.0
    scale = dw_dq / (h * dist)
    return scale * dx, scale * dy, scale * dz


@cuda.jit
def fill_1d_kernel(arr, value):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = value


@cuda.jit
def fill_2d_components_kernel(arr_x, arr_y, value):
    idx = cuda.grid(1)
    if idx < arr_x.size:
        arr_x[idx] = value
        arr_y[idx] = value


@cuda.jit
def combine_density_kernel(rho_self, rho_ff, rho_fb, density):
    idx = cuda.grid(1)
    if idx < density.size:
        density[idx] = rho_self[idx] + rho_ff[idx] + rho_fb[idx]


@cuda.jit
def clip_upper_1d_kernel(arr, upper):
    idx = cuda.grid(1)
    if idx < arr.size and arr[idx] > upper:
        arr[idx] = upper


@cuda.jit
def finalize_k_factor_kernel(sum_grad_x, sum_grad_y, sum_grad_z, sum_grad_sq, k_factor):
    idx = cuda.grid(1)
    if idx < k_factor.size:
        denom = (
            sum_grad_x[idx] * sum_grad_x[idx]
            + sum_grad_y[idx] * sum_grad_y[idx]
            + sum_grad_z[idx] * sum_grad_z[idx]
            + sum_grad_sq[idx]
        )
        if denom < 1.0e-12:
            denom = 1.0e-12
        k_factor[idx] = 1.0 / denom


@cuda.jit(device=True)
def _viscosity_dimension_factor(ndim: int) -> float:
    return 2.0 * (float(ndim) + 2.0)


# ─────────────────── Pair geometry ───────────────────────────────────────────

@cuda.jit
def pair_geometry_ff_kernel(positions, idx_i, idx_j, periodic_length, use_periodic,
                            dx_out, dy_out, dz_out, dist_out, ndim):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    i = idx_i[pair_idx]
    j = idx_j[pair_idx]
    ddx = positions[i, 0] - positions[j, 0]
    if use_periodic:
        ddx -= periodic_length * round(ddx / periodic_length)
    ddy = positions[i, 1] - positions[j, 1]
    if ndim == 3:
        ddz = positions[i, 2] - positions[j, 2]
        local_dist = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
        dz_out[pair_idx] = ddz
    else:
        local_dist = (ddx * ddx + ddy * ddy) ** 0.5
        dz_out[pair_idx] = 0.0
    dx_out[pair_idx] = ddx
    dy_out[pair_idx] = ddy
    dist_out[pair_idx] = local_dist


@cuda.jit
def pair_geometry_fb_kernel(
    fluid_positions, boundary_positions, idx_i, idx_j,
    periodic_length, use_periodic, dx_out, dy_out, dz_out, dist_out, ndim,
):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    i = idx_i[pair_idx]
    j = idx_j[pair_idx]
    ddx = fluid_positions[i, 0] - boundary_positions[j, 0]
    if use_periodic:
        ddx -= periodic_length * round(ddx / periodic_length)
    ddy = fluid_positions[i, 1] - boundary_positions[j, 1]
    if ndim == 3:
        ddz = fluid_positions[i, 2] - boundary_positions[j, 2]
        local_dist = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
        dz_out[pair_idx] = ddz
    else:
        local_dist = (ddx * ddx + ddy * ddy) ** 0.5
        dz_out[pair_idx] = 0.0
    dx_out[pair_idx] = ddx
    dy_out[pair_idx] = ddy
    dist_out[pair_idx] = local_dist


# ─────────────────── Density kernels ─────────────────────────────────────────

@cuda.jit
def density_ff_kernel(idx_i, idx_j, dist, mass, h, alpha, rho_ff):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    w = _cubic_spline_w(dist[pair_idx], h, alpha)
    val = mass * w
    cuda.atomic.add(rho_ff, idx_i[pair_idx], val)
    cuda.atomic.add(rho_ff, idx_j[pair_idx], val)


@cuda.jit
def density_fb_kernel(idx_i, idx_j, dist, boundary_psi, h, alpha, rho_fb):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    w = _cubic_spline_w(dist[pair_idx], h, alpha)
    val = boundary_psi[idx_j[pair_idx]] * w
    cuda.atomic.add(rho_fb, idx_i[pair_idx], val)


# ─────────────────── K-factor kernels ────────────────────────────────────────

@cuda.jit
def k_factor_ff_kernel(idx_i, idx_j, dx, dy, dz, dist, h, alpha,
                       sum_grad_x, sum_grad_y, sum_grad_z, sum_grad_sq):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    gx, gy, gz = _cubic_spline_grad_3d(
        dist[pair_idx], dx[pair_idx], dy[pair_idx], dz[pair_idx], h, alpha,
    )
    grad_sq = gx * gx + gy * gy + gz * gz
    i = idx_i[pair_idx]
    j = idx_j[pair_idx]
    cuda.atomic.add(sum_grad_x, i, gx)
    cuda.atomic.add(sum_grad_y, i, gy)
    cuda.atomic.add(sum_grad_z, i, gz)
    cuda.atomic.add(sum_grad_x, j, -gx)
    cuda.atomic.add(sum_grad_y, j, -gy)
    cuda.atomic.add(sum_grad_z, j, -gz)
    cuda.atomic.add(sum_grad_sq, i, grad_sq)
    cuda.atomic.add(sum_grad_sq, j, grad_sq)


@cuda.jit
def k_factor_fb_kernel(idx_i, dx, dy, dz, dist, h, alpha,
                       sum_grad_x, sum_grad_y, sum_grad_z, sum_grad_sq):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    gx, gy, gz = _cubic_spline_grad_3d(
        dist[pair_idx], dx[pair_idx], dy[pair_idx], dz[pair_idx], h, alpha,
    )
    grad_sq = gx * gx + gy * gy + gz * gz
    i = idx_i[pair_idx]
    cuda.atomic.add(sum_grad_x, i, gx)
    cuda.atomic.add(sum_grad_y, i, gy)
    cuda.atomic.add(sum_grad_z, i, gz)
    cuda.atomic.add(sum_grad_sq, i, grad_sq)


# ─────────────────── Pressure correction kernels ─────────────────────────────

@cuda.jit
def apply_pressure_ff_kernel(idx_i, idx_j, dx, dy, dz, dist,
                             density, lambda_values, mass, h, alpha, dt, velocities, ndim):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    i = idx_i[pair_idx]
    j = idx_j[pair_idx]
    gx, gy, gz = _cubic_spline_grad_3d(
        dist[pair_idx], dx[pair_idx], dy[pair_idx], dz[pair_idx], h, alpha,
    )
    ri = density[i]
    rj = density[j]
    coeff = mass * (
        lambda_values[i] / (ri * ri + 1.0e-12)
        + lambda_values[j] / (rj * rj + 1.0e-12)
    )
    cuda.atomic.add(velocities, (i, 0), -dt * coeff * gx)
    cuda.atomic.add(velocities, (i, 1), -dt * coeff * gy)
    cuda.atomic.add(velocities, (j, 0), dt * coeff * gx)
    cuda.atomic.add(velocities, (j, 1), dt * coeff * gy)
    if ndim == 3:
        cuda.atomic.add(velocities, (i, 2), -dt * coeff * gz)
        cuda.atomic.add(velocities, (j, 2), dt * coeff * gz)


@cuda.jit
def apply_pressure_fb_kernel(idx_i, idx_j, dx, dy, dz, dist,
                             density, lambda_values, boundary_psi, h, alpha, dt, velocities, ndim):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    i = idx_i[pair_idx]
    gx, gy, gz = _cubic_spline_grad_3d(
        dist[pair_idx], dx[pair_idx], dy[pair_idx], dz[pair_idx], h, alpha,
    )
    ri = density[i]
    coeff = boundary_psi[idx_j[pair_idx]] * (lambda_values[i] / (ri * ri + 1.0e-12))
    cuda.atomic.add(velocities, (i, 0), -dt * coeff * gx)
    cuda.atomic.add(velocities, (i, 1), -dt * coeff * gy)
    if ndim == 3:
        cuda.atomic.add(velocities, (i, 2), -dt * coeff * gz)


# ─────────────────── Residual kernels (CD) ───────────────────────────────────

@cuda.jit
def cd_residual_ff_kernel(idx_i, idx_j, dx, dy, dz, dist, velocities, mass, h, alpha, dt, residual, ndim):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    i = idx_i[pair_idx]
    j = idx_j[pair_idx]
    gx, gy, gz = _cubic_spline_grad_3d(
        dist[pair_idx], dx[pair_idx], dy[pair_idx], dz[pair_idx], h, alpha,
    )
    dv_x = velocities[i, 0] - velocities[j, 0]
    dv_y = velocities[i, 1] - velocities[j, 1]
    val = dt * mass * (dv_x * gx + dv_y * gy)
    if ndim == 3:
        dv_z = velocities[i, 2] - velocities[j, 2]
        val += dt * mass * dv_z * gz
    cuda.atomic.add(residual, i, val)
    cuda.atomic.add(residual, j, -val)


@cuda.jit
def cd_residual_fb_kernel(idx_i, idx_j, dx, dy, dz, dist, velocities,
                          boundary_velocities, boundary_psi, h, alpha, dt, residual, ndim):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    i = idx_i[pair_idx]
    j = idx_j[pair_idx]
    gx, gy, gz = _cubic_spline_grad_3d(
        dist[pair_idx], dx[pair_idx], dy[pair_idx], dz[pair_idx], h, alpha,
    )
    dv_x = velocities[i, 0] - boundary_velocities[j, 0]
    dv_y = velocities[i, 1] - boundary_velocities[j, 1]
    val = dt * boundary_psi[j] * (dv_x * gx + dv_y * gy)
    if ndim == 3:
        dv_z = velocities[i, 2] - boundary_velocities[j, 2]
        val += dt * boundary_psi[j] * dv_z * gz
    cuda.atomic.add(residual, i, val)


# ─────────────────── Residual kernels (DF) ───────────────────────────────────

@cuda.jit
def df_residual_ff_kernel(idx_i, idx_j, dx, dy, dz, dist, velocities, mass, h, alpha, residual, ndim):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    i = idx_i[pair_idx]
    j = idx_j[pair_idx]
    gx, gy, gz = _cubic_spline_grad_3d(
        dist[pair_idx], dx[pair_idx], dy[pair_idx], dz[pair_idx], h, alpha,
    )
    dv_x = velocities[i, 0] - velocities[j, 0]
    dv_y = velocities[i, 1] - velocities[j, 1]
    val = mass * (dv_x * gx + dv_y * gy)
    if ndim == 3:
        dv_z = velocities[i, 2] - velocities[j, 2]
        val += mass * dv_z * gz
    cuda.atomic.add(residual, i, val)
    cuda.atomic.add(residual, j, -val)


@cuda.jit
def df_residual_fb_kernel(idx_i, idx_j, dx, dy, dz, dist, velocities,
                          boundary_velocities, boundary_psi, h, alpha, residual, ndim):
    pair_idx = cuda.grid(1)
    if pair_idx >= idx_i.size:
        return
    i = idx_i[pair_idx]
    j = idx_j[pair_idx]
    gx, gy, gz = _cubic_spline_grad_3d(
        dist[pair_idx], dx[pair_idx], dy[pair_idx], dz[pair_idx], h, alpha,
    )
    dv_x = velocities[i, 0] - boundary_velocities[j, 0]
    dv_y = velocities[i, 1] - boundary_velocities[j, 1]
    val = boundary_psi[j] * (dv_x * gx + dv_y * gy)
    if ndim == 3:
        dv_z = velocities[i, 2] - boundary_velocities[j, 2]
        val += boundary_psi[j] * dv_z * gz
    cuda.atomic.add(residual, i, val)


# ─────────────────── Lambda / metric reduce kernels ──────────────────────────

@cuda.jit
def zero_scalar_kernel(arr):
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        arr[0] = 0.0


@cuda.jit
def cd_lambda_reduce_kernel(residual, density_ref, rho0, dt, k_factor, lambda_next, metric_sum):
    idx = cuda.grid(1)
    if idx >= residual.size:
        return
    rho_star = density_ref[idx] + residual[idx]
    rho_err = rho_star - rho0
    if rho_err < 0.0:
        rho_err = 0.0
    lambda_next[idx] = (rho_err / (dt * dt + 1.0e-12)) * k_factor[idx]
    if rho0 > 0.0:
        cuda.atomic.add(metric_sum, 0, rho_err / rho0)


@cuda.jit
def df_lambda_reduce_kernel(residual, rho0, dt, k_factor, lambda_next, metric_sum):
    idx = cuda.grid(1)
    if idx >= residual.size:
        return
    lambda_next[idx] = (residual[idx] / (dt + 1.0e-12)) * k_factor[idx]
    if rho0 > 0.0:
        value = residual[idx]
        if value < 0.0:
            value = -value
        cuda.atomic.add(metric_sum, 0, value * dt / rho0)


@cuda.jit
def fill_2d_kernel(arr, value):
    idx = cuda.grid(1)
    if idx < arr.shape[0]:
        for d in range(arr.shape[1]):
            arr[idx, d] = value


# ─────────────────── Adami boundary-state extrapolation ──────────────────────

@cuda.jit
def boundary_vel_scatter_kernel(fb_i, fb_j, fb_dist, fluid_velocities, h, alpha,
                                bnd_v_num_x, bnd_v_num_y, bnd_v_num_z, bnd_v_den, ndim):
    pair_idx = cuda.grid(1)
    if pair_idx >= fb_i.size:
        return
    i = fb_i[pair_idx]
    j = fb_j[pair_idx]
    w = _cubic_spline_w(fb_dist[pair_idx], h, alpha)
    cuda.atomic.add(bnd_v_num_x, j, fluid_velocities[i, 0] * w)
    cuda.atomic.add(bnd_v_num_y, j, fluid_velocities[i, 1] * w)
    if ndim == 3:
        cuda.atomic.add(bnd_v_num_z, j, fluid_velocities[i, 2] * w)
    cuda.atomic.add(bnd_v_den, j, w)


@cuda.jit
def boundary_vel_finalize_kernel(bnd_v_num_x, bnd_v_num_y, bnd_v_num_z, bnd_v_den,
                                 prior, boundary_velocities, ndim):
    b = cuda.grid(1)
    if b >= boundary_velocities.shape[0]:
        return
    den = bnd_v_den[b] + prior
    if den > 1.0e-6:
        boundary_velocities[b, 0] = -bnd_v_num_x[b] / den
        boundary_velocities[b, 1] = -bnd_v_num_y[b] / den
        if ndim == 3:
            boundary_velocities[b, 2] = -bnd_v_num_z[b] / den
    else:
        boundary_velocities[b, 0] = 0.0
        boundary_velocities[b, 1] = 0.0
        if ndim == 3:
            boundary_velocities[b, 2] = 0.0


# ─────────────────── Morris (1997) Laplacian viscosity ───────────────────────

@cuda.jit
def viscosity_ff_kernel(ff_i, ff_j, ff_dx, ff_dy, ff_dz, ff_dist,
                        velocities, mass, nu, h, alpha, rho0, accelerations, ndim):
    pair_idx = cuda.grid(1)
    if pair_idx >= ff_i.size:
        return
    i = ff_i[pair_idx]
    j = ff_j[pair_idx]
    d = ff_dist[pair_idx]
    ddx = ff_dx[pair_idx]
    ddy = ff_dy[pair_idx]
    ddz = ff_dz[pair_idx]
    gx, gy, gz = _cubic_spline_grad_3d(d, ddx, ddy, ddz, h, alpha)
    rdgw = ddx * gx + ddy * gy + ddz * gz
    denom = d * d + 0.01 * h * h
    coeff = _viscosity_dimension_factor(ndim) * nu * mass / rho0 * rdgw / denom

    v_ij_x = velocities[i, 0] - velocities[j, 0]
    v_ij_y = velocities[i, 1] - velocities[j, 1]
    cuda.atomic.add(accelerations, (i, 0), coeff * v_ij_x)
    cuda.atomic.add(accelerations, (i, 1), coeff * v_ij_y)
    cuda.atomic.add(accelerations, (j, 0), -coeff * v_ij_x)
    cuda.atomic.add(accelerations, (j, 1), -coeff * v_ij_y)
    if ndim == 3:
        v_ij_z = velocities[i, 2] - velocities[j, 2]
        cuda.atomic.add(accelerations, (i, 2), coeff * v_ij_z)
        cuda.atomic.add(accelerations, (j, 2), -coeff * v_ij_z)


@cuda.jit
def viscosity_fb_kernel(fb_i, fb_j, fb_dx, fb_dy, fb_dz, fb_dist, fluid_velocities,
                        boundary_velocities, boundary_mass, nu, h, alpha,
                        boundary_rho0, accelerations, ndim):
    pair_idx = cuda.grid(1)
    if pair_idx >= fb_i.size:
        return
    i = fb_i[pair_idx]
    j = fb_j[pair_idx]
    d = fb_dist[pair_idx]
    ddx = fb_dx[pair_idx]
    ddy = fb_dy[pair_idx]
    ddz = fb_dz[pair_idx]
    gx, gy, gz = _cubic_spline_grad_3d(d, ddx, ddy, ddz, h, alpha)
    rdgw = ddx * gx + ddy * gy + ddz * gz
    denom = d * d + 0.01 * h * h
    coeff = _viscosity_dimension_factor(ndim) * nu * boundary_mass / boundary_rho0 * rdgw / denom

    v_ij_x = fluid_velocities[i, 0] - boundary_velocities[j, 0]
    v_ij_y = fluid_velocities[i, 1] - boundary_velocities[j, 1]
    cuda.atomic.add(accelerations, (i, 0), coeff * v_ij_x)
    cuda.atomic.add(accelerations, (i, 1), coeff * v_ij_y)
    if ndim == 3:
        v_ij_z = fluid_velocities[i, 2] - boundary_velocities[j, 2]
        cuda.atomic.add(accelerations, (i, 2), coeff * v_ij_z)


# ─────────────────── Velocity prediction ─────────────────────────────────────

@cuda.jit
def velocity_predict_kernel(velocities, accelerations, gravity, dt, ndim):
    i = cuda.grid(1)
    if i >= velocities.shape[0]:
        return
    velocities[i, 0] += dt * (accelerations[i, 0] + gravity[0])
    velocities[i, 1] += dt * (accelerations[i, 1] + gravity[1])
    if ndim == 3:
        velocities[i, 2] += dt * (accelerations[i, 2] + gravity[2])


# ─────────────────── Position integration + periodic wrap ────────────────────

@cuda.jit
def position_integrate_kernel(positions, velocities, dt, ndim):
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    positions[i, 0] += dt * velocities[i, 0]
    positions[i, 1] += dt * velocities[i, 1]
    if ndim == 3:
        positions[i, 2] += dt * velocities[i, 2]


@cuda.jit
def xsph_kernel(velocities, densities,
                idx_i, idx_j,
                dist,
                mass, h, alpha,
                xsph_eps,
                xsph_out,
                n_pairs):
    k = cuda.grid(1)
    if k >= n_pairs:
        return

    i = idx_i[k]
    j = idx_j[k]
    d = dist[k]
    if d >= 2.0 * h:
        return

    q = d / h
    if q < 1.0:
        w = alpha * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    elif q < 2.0:
        t = 2.0 - q
        w = alpha * 0.25 * t * t * t
    else:
        return

    rho_ij = 0.5 * (densities[i] + densities[j])
    if rho_ij <= 1.0e-12:
        return

    coeff = xsph_eps * mass / rho_ij * w
    for dim_idx in range(velocities.shape[1]):
        delta = coeff * (velocities[j, dim_idx] - velocities[i, dim_idx])
        cuda.atomic.add(xsph_out, (i, dim_idx), delta)
        cuda.atomic.add(xsph_out, (j, dim_idx), -delta)


@cuda.jit
def add_velocity_delta_kernel(velocities, delta, ndim):
    i = cuda.grid(1)
    if i >= velocities.shape[0]:
        return
    velocities[i, 0] += delta[i, 0]
    velocities[i, 1] += delta[i, 1]
    if ndim == 3:
        velocities[i, 2] += delta[i, 2]


@cuda.jit
def periodic_wrap_x_kernel(positions, x_min, x_max):
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    L_x = x_max - x_min
    x = positions[i, 0]
    while x < x_min:
        x += L_x
    while x >= x_max:
        x -= L_x
    positions[i, 0] = x


@cuda.jit
def neighbor_count_kernel(pair_indices, counts):
    """Atomically increment counts[pair_indices[idx]] for each pair entry."""
    idx = cuda.grid(1)
    if idx >= pair_indices.size:
        return
    cuda.atomic.add(counts, pair_indices[idx], 1)


def launch_config(size: int, threads_per_block: int = 128) -> tuple[int, int]:
    if size <= 0:
        return 1, threads_per_block
    blocks = (size + threads_per_block - 1) // threads_per_block
    return blocks, threads_per_block
