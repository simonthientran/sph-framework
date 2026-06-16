# SPH Framework — Particle-Based CFD for Internal Flow

## Overview

This repository implements a modular Smoothed Particle Hydrodynamics (SPH) framework with a focus on:

- **Single-phase internal flow (pipe/channel flow)**
- **Physics-based validation (Poiseuille flow)**
- **Extensibility towards modern incompressible SPH solvers (IISPH, DFSPH)**
- **Clean software architecture for research and HPC extensions**

The long-term goal is to evolve this project towards a **research-grade SPH simulation framework, with emphasis on:

- correctness
- reproducibility
- extensibility
- performance

---

## Project Goals

### Short-Term Goals
- Stable 2D single-phase internal flow simulation
- Fully filled channel / pipe flow
- Reproducible benchmark case
- Validation against analytical Poiseuille profile

### Mid-Term Goals
- Implement incompressible solvers:
  - PCISPH
  - IISPH
  - DFSPH (primary target)
- Improve boundary handling
- Improve numerical stability

### Long-Term Goals
- GPU acceleration (CuPy / CUDA)
- High-performance neighbor search
- Multi-phase extension
- Integration of ML-based methods (e.g. Neural Pressure Projection)
- Research-grade CFD/SPH toolchain

---

## Core Concepts

### What is SPH?

Smoothed Particle Hydrodynamics (SPH) is a **mesh-free Lagrangian method** where:

- Fluid is represented by particles
- Fields are approximated using kernel interpolation

General form:


A(x_i) = Σ_j m_j / ρ_j * A_j * W(x_i - x_j, h)


---

### Governing Equations

We discretize the Navier-Stokes equations:

#### Continuity (density)

ρ_i = Σ_j m_j W_ij


#### Momentum

dv/dt = -1/ρ ∇p + ν ∇²v + f


---

### Pressure Model (WCSPH)


p_i = k (ρ_i - ρ₀)


Weakly compressible approximation.

---

### Pressure Force (symmetric form)


a_i = - Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij


---

### Viscosity (pairwise form)


a_i = Σ_j m_j * (4ν (r_ij · ∇W_ij)) / ((ρ_i + ρ_j)(|r_ij|² + ε)) * (v_i - v_j)


---

### Kernel

Cubic spline kernel is used:

- compact support
- standard in SPH literature
- stable and efficient

---

## Current Simulation Focus

### Reference Case: 2D Periodic Channel Flow

- Domain: rectangular
- Periodic in x-direction
- Solid walls in y-direction
- Fully filled with fluid
- Driven by constant body force (gravity in x)

This corresponds to **Poiseuille flow**.

---

## Validation Target

Analytical solution for laminar flow:


u(y) = (1 / (2μ)) * dp/dx * (y(H - y))


Expected behavior:

- Parabolic velocity profile
- Zero velocity at walls
- Maximum velocity at centerline

---

## Repository Structure


src/sph/
core/
bootstrap.py # main simulation entry
state.py # particle state container

neighbors/
    spatial_hash.py     # neighbor search (grid-based)

sph/
    density.py
    pressure.py
    viscosity.py
    xsph.py
    kernels.py

visualization/
    animate_particles.py

scenes/
examples/
pipe_flow_1phase_periodic_2d.json

out/
pipe_flow_1phase_periodic/
csv/
vtk/
profiles/
anim/


---

## Simulation Pipeline

1. Load scene (JSON)
2. Initialize particles
3. Build neighbor search
4. Loop over time:
   - Density computation
   - Pressure computation
   - Forces (pressure, viscosity, external)
   - Optional XSPH
   - Time integration
5. Export results

---

## How to Run

### 1. Setup environment

```bash
cd sph-framework
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run simulation
python -m sph.core.bootstrap scenes/examples/pipe_flow_1phase_periodic_2d.json

4. Create animation
python -m sph.visualization.animate_particles \
    out/pipe_flow_1phase_periodic/csv \
    --color-by vmag \
    --save out/pipe_flow_1phase_periodic/anim/pipe_flow.gif
   
Important Design Decisions

1. Periodic Domain Handling
Implemented via minimum-image convention
Critical for internal flow correctness
2. Fluid-only XSPH
Avoid artificial damping from boundary particles
Improves internal flow behavior
3. Explicit WCSPH baseline
Used as initial solver
Will be replaced by incompressible solvers
Current Limitations
WCSPH → compressibility artifacts
No true inlet/outlet boundaries yet
Boundary handling still simplified
No adaptive timestep control beyond CFL
CPU-only

Next Steps (Critical)
1. Solver Upgrade
Implement IISPH or DFSPH
Reduce density error
Improve stability
2. Boundary Conditions
Improved wall treatment
Inlet/outlet (non-periodic flow)
3. Validation
Automated Poiseuille comparison
L2 error tracking
4. Performance
Vectorization
GPU acceleration
Better neighbor structures
For Developers / LLMs Continuing This Project
If you are extending this code:

You MUST:

Keep physics consistency across modules
Use neighbor_search.relative_vector(...) for ALL interactions
Not mix multiple scene types during debugging
Validate changes using the pipe flow benchmark
Priority Order
Correctness > Performance
Single benchmark > Multiple unfinished features
Reproducibility > Complexity
Key References
Monaghan, J. J. (1992) — SPH fundamentals
Ihmsen et al. (2014) — SPH tutorial
Price (2012) — Astrophysical SPH
Müller et al. (2003) — Particle-based fluids
Vision

This project is intended to evolve into:

A modular, research-oriented SPH simulation framework bridging classical CFD, HPC, and modern AI-driven simulation methods.

Author Notes

This project is actively developed as part of an engineering learning and research trajectory focused on:

CFD / SPH
High-performance computing
AI-assisted simulation

The codebase is intentionally structured to be:

readable
extensible
scientifically grounded
License

TBD
