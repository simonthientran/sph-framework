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

## Pipe-flow vx-profile validation
- Run simulation:
  - `python -m sph.core.bootstrap scenes/examples/pipe_flow_1phase_2d.json`
- Output profile CSV (with metadata header): `out/pipe_flow_2d/vx_profile_bins.csv`
- Generate validation plot:
  - `python tools/plot_vx_profile.py --input out/pipe_flow_2d/vx_profile_bins.csv --output out/pipe_flow_2d/vx_profile.png`

## STL/CAD boundary import
- Scene schema supports:
  - `geometry.meshes[].path` (or `file`)
  - transform: `scale`, `translate`, `rotate_euler_deg`
  - units: `units_hint` (`"m"`, `"mm"`, `"cm"`)
  - boundary generation: `boundary_spacing`, `layers`, `layer_mode`
- Run with geometry diagnostics:
  - `python -m sph.core.bootstrap scenes/examples/pipe_from_stl.json --debug-geom`

## Solver verification mode (CI gate)
- Run deterministic verification with pass/fail exit code:
  - `python -m sph.core.bootstrap scenes/verification/hydrostatic_2d_wcsph.json --verify`
  - `python -m sph.core.bootstrap scenes/verification/poiseuille_2d_wcsph.json --verify`
- Reports are written to `out/verification/*.json`.

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
