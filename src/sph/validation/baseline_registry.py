"""Canonical frozen benchmark baseline for the current single-phase CUDA phase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


BenchmarkKind = Literal["bench", "profile"]


@dataclass(frozen=True, slots=True)
class TimingRange:
    """Reference timing window for a frozen benchmark measurement."""

    minimum_ms: float
    maximum_ms: float
    reference_ms: float


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Named average timing in milliseconds."""

    name: str
    avg_ms: float


@dataclass(frozen=True, slots=True)
class FrozenBenchmarkScene:
    """Canonical scene metadata and reference timings for the frozen baseline."""

    key: str
    relative_scene_path: str
    scene_name: str
    intended_use: str
    backend_name: str
    solver_name: str
    fluid_particles: int
    boundary_particles: int
    fluid_spacing: float
    support_radius: float
    domain_width: float
    domain_height: float
    dt_expected: float
    iter_cd_expected: int
    iter_df_expected: int
    steady_state_wall_ms: TimingRange
    key_cuda_stages_ms: tuple[StageTiming, ...]
    pair_build_subphases_ms: tuple[StageTiming, ...]
    pair_build_ranking: tuple[str, ...]

    def scene_path(self, repo_root: Path) -> Path:
        return Path(repo_root) / self.relative_scene_path


@dataclass(frozen=True, slots=True)
class BenchmarkPreset:
    """Command preset for the frozen benchmark workflow."""

    name: str
    kind: BenchmarkKind
    scene_key: str
    description: str
    backend_name: str | None = None
    steps: int = 500
    warmup: int | None = None
    log_every: int | None = None
    require_real_cuda: bool = False


REFERENCE_GPU = "RTX 4050"
REFERENCE_CUDA_MODE = "NUMBA_ENABLE_CUDASIM=0"
REFERENCE_PYTHON_ENTRY = ".venv/bin/python"
PERFORMANCE_PHASE_STATUS = "frozen"


FROZEN_BASELINE_SCENES: dict[str, FrozenBenchmarkScene] = {
    "base": FrozenBenchmarkScene(
        key="base",
        relative_scene_path="scenes/examples/pipe_flow_2d.json",
        scene_name="Pipe Flow 2D Benchmark",
        intended_use="single_phase_benchmark",
        backend_name="numba_cuda",
        solver_name="dfsph",
        fluid_particles=2250,
        boundary_particles=756,
        fluid_spacing=0.008,
        support_radius=0.0208,
        domain_width=1.0,
        domain_height=0.2,
        dt_expected=1.0e-04,
        iter_cd_expected=1,
        iter_df_expected=1,
        steady_state_wall_ms=TimingRange(minimum_ms=5.7, maximum_ms=6.2, reference_ms=5.880),
        key_cuda_stages_ms=(
            StageTiming("pair_build", 1.717),
            StageTiming("solve", 1.031),
            StageTiming("density", 0.405),
            StageTiming("k_factor", 0.388),
            StageTiming("boundary_state", 0.348),
            StageTiming("upload", 0.317),
            StageTiming("neighbor_count", 0.288),
            StageTiming("pair_geometry", 0.258),
        ),
        pair_build_subphases_ms=(
            StageTiming("count_scan_scatter", 0.323),
            StageTiming("ff_emit", 0.285),
            StageTiming("fb_emit", 0.267),
            StageTiming("hash_assign", 0.165),
        ),
        pair_build_ranking=("count_scan_scatter", "ff_emit", "fb_emit", "hash_assign"),
    ),
    "dense": FrozenBenchmarkScene(
        key="dense",
        relative_scene_path="scenes/examples/pipe_flow_2d_dense.json",
        scene_name="Pipe Flow 2D Dense (profiling)",
        intended_use="single_phase_benchmark",
        backend_name="numba_cuda",
        solver_name="dfsph",
        fluid_particles=5800,
        boundary_particles=1206,
        fluid_spacing=0.005,
        support_radius=0.013,
        domain_width=1.0,
        domain_height=0.2,
        dt_expected=1.0e-04,
        iter_cd_expected=1,
        iter_df_expected=1,
        steady_state_wall_ms=TimingRange(minimum_ms=6.6, maximum_ms=7.2, reference_ms=6.866),
        key_cuda_stages_ms=(
            StageTiming("pair_build", 2.046),
            StageTiming("solve", 1.078),
            StageTiming("density", 0.413),
            StageTiming("k_factor", 0.401),
            StageTiming("pair_geometry", 0.371),
            StageTiming("upload", 0.367),
            StageTiming("boundary_state", 0.345),
            StageTiming("neighbor_count", 0.295),
        ),
        pair_build_subphases_ms=(
            StageTiming("ff_emit", 0.432),
            StageTiming("fb_emit", 0.380),
            StageTiming("count_scan_scatter", 0.365),
            StageTiming("hash_assign", 0.183),
        ),
        pair_build_ranking=("ff_emit", "fb_emit", "count_scan_scatter", "hash_assign"),
    ),
}


FROZEN_BASELINE_OVERALL_BOTTLENECKS = (
    "pair_build",
    "solve",
    "density / k_factor / upload / pair_geometry",
)


SUCCESSFUL_OPTIMIZATION_DIRECTIONS = (
    "Device-side neighbor build replaced CPU-built pair upload.",
    "CD to DF dynamic state ownership was tightened so device-resident state is reused instead of re-uploaded.",
    "Pair-geometry buffers were made reusable, removing per-step materialization allocation overhead.",
    "Hash-grid assign path removed dead writes on the numba_scan route.",
    "Count-side shared-memory histogram reduced global atomics in cell counting.",
    "Scatter-side block reservation reduced global cursor atomic pressure.",
)


REJECTED_OPTIMIZATION_DIRECTIONS = (
    "FF pair-counter batching with thread-local mini-buffers.",
    "FF same-cell versus cross-cell split kernels.",
    "FF full cell-centric shared-memory tiled traversal.",
    "Count/scan/scatter copy-removal ideas that reused range buffers and regressed the dense benchmark.",
)


BASELINE_RUN_PRESETS: dict[str, BenchmarkPreset] = {
    "cpu-base": BenchmarkPreset(
        name="cpu-base",
        kind="bench",
        scene_key="base",
        description="CPU reference benchmark on the base pipe-flow scene.",
        backend_name="numba_cpu",
        steps=500,
        log_every=50,
    ),
    "cuda-base": BenchmarkPreset(
        name="cuda-base",
        kind="bench",
        scene_key="base",
        description="CUDA benchmark on the base pipe-flow scene.",
        backend_name="numba_cuda",
        steps=500,
        log_every=50,
    ),
    "cuda-dense": BenchmarkPreset(
        name="cuda-dense",
        kind="bench",
        scene_key="dense",
        description="CUDA benchmark on the dense pipe-flow profiling scene.",
        backend_name="numba_cuda",
        steps=500,
        log_every=50,
    ),
    "cuda-profile-base": BenchmarkPreset(
        name="cuda-profile-base",
        kind="profile",
        scene_key="base",
        description="CUDA stage profile on the base pipe-flow scene.",
        steps=500,
        warmup=5,
        require_real_cuda=True,
    ),
    "cuda-profile-dense": BenchmarkPreset(
        name="cuda-profile-dense",
        kind="profile",
        scene_key="dense",
        description="CUDA stage profile on the dense pipe-flow profiling scene.",
        steps=500,
        warmup=5,
        require_real_cuda=True,
    ),
}


def frozen_scene(scene_key: str) -> FrozenBenchmarkScene:
    try:
        return FROZEN_BASELINE_SCENES[scene_key]
    except KeyError as exc:
        supported = ", ".join(sorted(FROZEN_BASELINE_SCENES))
        raise KeyError(f"Unknown frozen baseline scene '{scene_key}'. Supported: {supported}") from exc


def benchmark_preset(preset_name: str) -> BenchmarkPreset:
    try:
        return BASELINE_RUN_PRESETS[preset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(BASELINE_RUN_PRESETS))
        raise KeyError(f"Unknown baseline preset '{preset_name}'. Supported: {supported}") from exc


def baseline_scene_path(repo_root: Path, scene_key: str) -> Path:
    return frozen_scene(scene_key).scene_path(repo_root)


def build_preset_command(
    repo_root: Path,
    python_bin: str,
    preset_name: str,
    steps: int | None = None,
    warmup: int | None = None,
    log_every: int | None = None,
    base_env: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str], BenchmarkPreset]:
    repo_root = Path(repo_root)
    preset = benchmark_preset(preset_name)
    scene = frozen_scene(preset.scene_key)
    env = dict(base_env or {})
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not current_pythonpath else f"{src_path}:{current_pythonpath}"

    if preset.kind == "bench":
        resolved_steps = int(steps if steps is not None else preset.steps)
        resolved_log_every = int(log_every if log_every is not None else (preset.log_every or 50))
        cmd = [
            python_bin,
            str(repo_root / "scripts/bench_density.py"),
            str(scene.scene_path(repo_root)),
            str(resolved_steps),
            str(resolved_log_every),
            str(preset.backend_name),
        ]
        return cmd, env, preset

    resolved_steps = int(steps if steps is not None else preset.steps)
    resolved_warmup = int(warmup if warmup is not None else (preset.warmup or 5))
    env["NUMBA_ENABLE_CUDASIM"] = "0"
    cmd = [
        python_bin,
        str(repo_root / "scripts/profile_cuda_stages.py"),
        "--scene",
        str(scene.scene_path(repo_root)),
        "--warmup",
        str(resolved_warmup),
        "--steps",
        str(resolved_steps),
    ]
    return cmd, env, preset


__all__ = [
    "BASELINE_RUN_PRESETS",
    "FROZEN_BASELINE_OVERALL_BOTTLENECKS",
    "FROZEN_BASELINE_SCENES",
    "PERFORMANCE_PHASE_STATUS",
    "REFERENCE_CUDA_MODE",
    "REFERENCE_GPU",
    "REFERENCE_PYTHON_ENTRY",
    "REJECTED_OPTIMIZATION_DIRECTIONS",
    "SUCCESSFUL_OPTIMIZATION_DIRECTIONS",
    "BenchmarkPreset",
    "FrozenBenchmarkScene",
    "StageTiming",
    "TimingRange",
    "baseline_scene_path",
    "benchmark_preset",
    "build_preset_command",
    "frozen_scene",
]
