# Design Note: Benchmarks & Metrics

## Overview
The goal of Milestone M1 is to provide a **trustable** single‑phase benchmark suite for internal flows (Poiseuille and Couette) that can be used to validate the SPH framework against analytical solutions.  The suite consists of:

1. **Benchmark scene files** (`scenes/examples/poiseuille_2d.json`, `scenes/examples/couette_2d.json`).
2. **Analytic profile utilities** (`src/sph/diagnostics/benchmark_metrics.py`).
3. **Metric collection** – L2‑error of the velocity profile, mass drift, timestep statistics, neighbor statistics.
4. **Export pipeline** – optional VTK/CSV writer (`src/sph/io/exporter.py`).
5. **Improved CFD‑style boundary conditions** – no‑slip/slip walls with viscous damping and sponge‑layer inflow/outflow.

## Workflow
During each simulation step the bootstrap code:
```python
metrics.update(state, step)
exporter.maybe_write(step, state)
```
`BenchmarkMetrics` samples the velocity field in bins across the channel height, compares it to the analytical solution, and accumulates statistics.  At the end of the run a concise summary is printed.

## Export Format
* **VTK** – written with `pyvista`/`meshio` if available, otherwise a simple CSV fallback.
* Files are stored under `out/<scene_name>/step_XXXX.{vtk|csv}`.

## Boundary Conditions
* **WallBoundary** – no‑slip enforced by applying a tangential damping force proportional to the fluid viscosity (`mu`).
* **InflowBoundary** – injects particles with prescribed velocity and density in a buffer region.
* **OutflowBoundary** – sponge layer that gradually damps velocity and marks particles inactive after they cross a configurable thickness.

## Usage
```bash
# Run Poiseuille benchmark (2000 steps)
python -m sph.core.bootstrap scenes/examples/poiseuille_2d.json

# Run Couette benchmark
python -m sph.core.bootstrap scenes/examples/couette_2d.json
```
Metrics are printed every `metrics.interval` steps (default 100).  Exported files can be visualised with ParaView or PyVista.

---
*All new modules are deliberately small and have unit‑tests under `tests/`.  The design keeps the existing solver code untouched and adds functionality via the `Bootstrap` orchestration layer.*
