# Single-Phase CUDA Baseline

This document freezes the current single-phase CUDA baseline so future work has
one agreed reference point for demos, regression checks, and later performance
phases.

The authoritative in-code definition lives in
`src/sph/validation/baseline_registry.py`. This document is the project-facing
engineering summary of that frozen baseline.

## Scope

- Solver: `dfsph`
- Primary backend under test: `numba_cuda`
- Reference correctness backend: `numba_cpu`
- Benchmark family: periodic-like 2D pipe/channel flow
- Performance phase status: frozen

This is not a physics retuning document. It does not change solver parameters,
scene setup, or UI behavior.

## Frozen Benchmark Scenes

### Base scene

- File: `scenes/examples/pipe_flow_2d.json`
- Name: `Pipe Flow 2D Benchmark`
- Intended use: `single_phase_benchmark`
- Fluid particles: `2250`
- Boundary particles: `756`
- Fluid spacing: `0.008`
- Support radius: `0.0208`
- Domain: `1.0 x 0.2`
- Time control: CFL mode with `dt_max=1.0e-04`
- Expected steady-state behavior:
- `dt` holds at `1.0e-04`
- `iter_cd=1`
- `iter_df=1`
- density mean stays near `rho0=1000`

### Dense scene

- File: `scenes/examples/pipe_flow_2d_dense.json`
- Name: `Pipe Flow 2D Dense (profiling)`
- Intended use: `single_phase_benchmark`
- Fluid particles: `5800`
- Boundary particles: `1206`
- Fluid spacing: `0.005`
- Support radius: `0.013`
- Domain: `1.0 x 0.2`
- Time control: CFL mode with `dt_max=1.0e-04`
- Expected steady-state behavior:
- `dt` holds at `1.0e-04`
- `iter_cd=1`
- `iter_df=1`
- density mean stays near `rho0=1000`

## Reference Hardware and Profiling Conditions

- Reference GPU: RTX 4050
- CUDA mode only: `NUMBA_ENABLE_CUDASIM=0`
- Warmup for stage profiling: `5` steps
- Steady-state profiling window: `500` steps
- Python entry point: `.venv/bin/python`
- `PYTHONPATH=src`

Absolute timings are hardware-dependent. Treat the ranges below as the frozen
reference for the RTX 4050 baseline, and treat stage ordering as the stronger
cross-machine signal.

## Frozen CUDA Baseline

### Base scene reference

- Scene: `pipe_flow_2d.json`
- Backend: `numba_cuda`
- Solver: `dfsph`
- Steady-state wall time: expect about `5.7-6.2 ms/step`
- Reference mean wall time: `5.880 ms/step`
- Key stage timings:
- `pair_build`: `1.717 ms`
- `solve`: `1.031 ms`
- `density`: `0.405 ms`
- `k_factor`: `0.388 ms`
- `boundary_state`: `0.348 ms`
- `upload`: `0.317 ms`
- `neighbor_count`: `0.288 ms`
- `pair_geometry`: `0.258 ms`
- Internal pair-build ranking:
- `count_scan_scatter`: `0.323 ms`
- `ff_emit`: `0.285 ms`
- `fb_emit`: `0.267 ms`
- `hash_assign`: `0.165 ms`

### Dense scene reference

- Scene: `pipe_flow_2d_dense.json`
- Backend: `numba_cuda`
- Solver: `dfsph`
- Steady-state wall time: expect about `6.6-7.2 ms/step`
- Reference mean wall time: `6.866 ms/step`
- Key stage timings:
- `pair_build`: `2.046 ms`
- `solve`: `1.078 ms`
- `density`: `0.413 ms`
- `k_factor`: `0.401 ms`
- `pair_geometry`: `0.371 ms`
- `upload`: `0.367 ms`
- `boundary_state`: `0.345 ms`
- `neighbor_count`: `0.295 ms`
- Internal pair-build ranking:
- `ff_emit`: `0.432 ms`
- `fb_emit`: `0.380 ms`
- `count_scan_scatter`: `0.365 ms`
- `hash_assign`: `0.183 ms`

## What To Run

### CPU reference benchmark

Use this to verify solver behavior on the canonical scene with the CPU backend.
This is the reference behavior run, not the frozen performance target.

```bash
PYTHONPATH=src .venv/bin/python scripts/bench_density.py \
  scenes/examples/pipe_flow_2d.json 500 50 numba_cpu
```

### CUDA benchmark

Use this for the frozen single-phase CUDA baseline on the base scene.

```bash
PYTHONPATH=src .venv/bin/python scripts/bench_density.py \
  scenes/examples/pipe_flow_2d.json 500 50 numba_cuda
```

### Dense CUDA benchmark

Use this for the denser profiling companion scene.

```bash
PYTHONPATH=src .venv/bin/python scripts/bench_density.py \
  scenes/examples/pipe_flow_2d_dense.json 500 50 numba_cuda
```

### CUDA stage profiler

Use this when you need stage-level timing rather than the higher-level benchmark
summary.

```bash
NUMBA_ENABLE_CUDASIM=0 PYTHONPATH=src .venv/bin/python scripts/profile_cuda_stages.py \
  --scene scenes/examples/pipe_flow_2d.json --warmup 5 --steps 500

NUMBA_ENABLE_CUDASIM=0 PYTHONPATH=src .venv/bin/python scripts/profile_cuda_stages.py \
  --scene scenes/examples/pipe_flow_2d_dense.json --warmup 5 --steps 500
```

### Baseline wrapper

The project also provides preset wrappers for the same commands:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_baseline_benchmarks.py cpu-base
PYTHONPATH=src .venv/bin/python scripts/run_baseline_benchmarks.py cuda-base
PYTHONPATH=src .venv/bin/python scripts/run_baseline_benchmarks.py cuda-dense
PYTHONPATH=src .venv/bin/python scripts/run_baseline_benchmarks.py cuda-profile-base
PYTHONPATH=src .venv/bin/python scripts/run_baseline_benchmarks.py cuda-profile-dense
```

## What Output Matters

For `bench_density.py`:

- backend name
- solver name
- particle counts
- `dt`
- `iter_cd`
- `iter_df`
- `ms/step`
- `roll_ms`
- density quality summary near `rho0`
- solver health summary

For `profile_cuda_stages.py`:

- total steady-state wall time
- `pair_build`
- `solve`
- second-tier stages
- pair-build sub-phase ordering

## Current Bottleneck Ranking

Overall stage ranking on the frozen CUDA baseline:

1. `pair_build`
2. `solve`
3. `density`, `k_factor`, `upload`, and `pair_geometry` second tier

Internal `pair_build` ranking:

- Base scene:
- `count_scan_scatter`
- `ff_emit`
- `fb_emit`
- `hash_assign`
- Dense scene:
- `ff_emit`
- `fb_emit`
- `count_scan_scatter`
- `hash_assign`

## Successful Optimizations Already Landed

- Device-side neighbor build replaced CPU-built pair upload.
- CD to DF dynamic state ownership was tightened so device-resident state is
  reused instead of re-uploaded.
- Pair-geometry buffers were made reusable, removing per-step materialization
  allocation overhead.
- Hash-grid assign path removed dead writes on the `numba_scan` route.
- Count-side shared-memory histogram reduced global atomics in cell counting.
- Scatter-side block reservation reduced global cursor atomic pressure.

These are now part of the frozen baseline.

## Rejected Directions

These were measured on the real GPU and should not be repeated blindly:

- FF pair-counter batching with thread-local mini-buffers.
- FF same-cell versus cross-cell split kernels.
- FF full cell-centric shared-memory tiled traversal.
- Count/scan/scatter copy-removal ideas that reused range buffers and regressed
  the dense benchmark.

Rule for future work: if an idea is already listed here, do not retry it
without a materially different implementation strategy and a clear reason.

## What Is Working Well

- CUDA single-phase path is functionally correct for the frozen benchmark.
- Neighbor pairs are device-resident.
- Pair-set equality against CPU reference has been validated.
- Stage-level profiling exists and is usable for real-device work.
- The benchmark scenes are stable enough to use as demo and regression anchors.

## What Is Not Perfect Yet

- `pair_build` is still the top overall bottleneck.
- `solve` is still the second-largest stage.
- Dense-scene FF emission remains expensive.
- The baseline is performance-stable enough for comparison, but not yet the end
  state of the GPU architecture.

## What We Are Deliberately Not Doing In This Phase

- No more kernel micro-optimization loops.
- No solver retuning.
- No physics redesign.
- No UI or presentation redesign tied to solver changes.

This phase is about freezing the baseline, documenting it, and making the
benchmark workflow repeatable.

## Exit Criteria For This Phase

The current phase is complete when all of the following are true:

- The benchmark scenes and commands are documented.
- The CUDA baseline timings are documented.
- The current bottlenecks are documented.
- Successful and rejected optimization directions are documented.
- Future work starts from this frozen baseline instead of ad hoc kernel edits.

## Next Phase

The next major phase is not another immediate kernel-tuning sprint. It should
be one of:

- a deliberate new performance phase that starts from this frozen baseline and
  targets `pair_build` again with a fresh plan, or
- demo, reporting, and presentation work that makes the current CUDA baseline
  easier to show and compare.

If and when performance work resumes, start with:

1. a new hypothesis for `pair_build`
2. a measurement plan before code changes
3. exact pair-set equality as a gate
4. comparison against this frozen document
