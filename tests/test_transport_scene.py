from __future__ import annotations

from pathlib import Path

from sph.core.simulation import SimulationRunner
from sph.simulator_new import Simulator


SCENE_TRANSPORT = Path("scenes/examples/box_transport_3d.json").resolve()


def test_transport_scene_combines_inflow_and_outflow():
    sim = Simulator(SCENE_TRANSPORT)
    initial_count = sim.fluid.n
    blocked_seen = 0.0

    for _ in range(6):
        sim.step()
        blocked_seen = max(blocked_seen, sim.last_solver_stats.get("emitter_budget_blocked_count", 0.0))
    stats = sim.last_solver_stats

    assert sim.fluid.n != initial_count
    assert stats["cumulative_emitted_particles"] > 0.0
    assert stats["removed_particles"] > 0.0
    assert stats["cumulative_removed_particles"] >= stats["removed_particles"]
    assert blocked_seen >= 1.0


def test_transport_scene_reports_cumulative_metrics():
    runner = SimulationRunner(SCENE_TRANSPORT, backend_name="numba_cpu")
    cumulative_emitted = 0.0
    cumulative_removed = 0.0
    blocked_seen = 0.0

    for _ in range(6):
        result = runner.step()
        metrics = result.runtime.solver_metrics
        cumulative_emitted = metrics.get("cumulative_emitted_particles", 0.0)
        cumulative_removed = metrics.get("cumulative_removed_particles", 0.0)
        blocked_seen = max(blocked_seen, metrics.get("emitter_budget_blocked_count", 0.0))

    assert cumulative_emitted > 0.0
    assert cumulative_removed > 0.0
    assert blocked_seen >= 1.0
    assert "transport_particle_delta" in result.runtime.solver_metrics
