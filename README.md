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
