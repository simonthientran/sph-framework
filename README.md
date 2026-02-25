# SPH Framework

Modular Smoothed Particle Hydrodynamics (SPH) framework in Python.

## Goals
- Clean architecture (solver/forces/neighbors swapbar)
- Deterministic runs (reproducible simulations)
- Scene system (JSON)
- Export (CSV + VTK)
- Performance-ready design (NumPy first, Numba later)

## Repo Structure
- `src/sph/` core framework
- `scenes/` simulation scenes (JSON)
- `docs/` architecture + ADR decisions
- `tests/` unit/regression tests
# sph-framework

## PCISPH stability parameters (control logic)
These knobs do **not** change the PCISPH equations; they only change control logic around them:

- **Active-set**
  - `min_neighbors_for_pressure`: neighbor threshold for pressure solve (default 7)
  - `inactive_hold_steps`: require N consecutive under-threshold steps before pressure-skip
  - `force_active_if_density_low`: keep particles pressure-active when `rho*` is low (default true)
  - `force_active_rho_min`: threshold for forcing active (e.g. `0.95 * rho0`)

- **Negative pressure handling**
  - `negative_pressure_mode`: `"none" | "hard_zero" | "soft_cap"`
  - `negative_pressure_soft_factor`: $\alpha$ for `"soft_cap"` dynamic cap
  - `negative_pressure_cap`: optional fixed cap (overrides dynamic cap)

- **Boundary response (AABB collision)**
  - `boundary.eps`: push-out epsilon to avoid exact-on-wall teleports
  - `boundary.restitution`, `boundary.friction`: normal reflection + tangential damping

## Results export (CSV + XLSX)
See `docs/results_export.md` for how to enable structured per-step results export to:
- `<out_dir>/<base_name>_steps.csv`
- `<out_dir>/<base_name>_vxprof.csv`
- `<out_dir>/<base_name>.xlsx` (sheets: `steps`, `vx_profile`, `meta`)
