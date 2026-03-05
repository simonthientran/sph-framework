import numpy as np

from sph.core.diagnostics import InstabilityDetector, StepMetrics
from sph.core.state import ParticleState


def _state() -> ParticleState:
    pos = np.array([[0.0, 0.0], [0.02, 0.0]], dtype=np.float64)
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    mass = np.ones((2,), dtype=np.float64)
    rho = np.full((2,), 1000.0, dtype=np.float64)
    p = np.zeros((2,), dtype=np.float64)
    is_boundary = np.zeros((2,), dtype=np.bool_)
    return ParticleState(dim=2, pos=pos, vel=vel, acc=acc, mass=mass, rho=rho, p=p, is_boundary=is_boundary)


def test_instability_detector_flags_multiple_conditions():
    d = InstabilityDetector(rho0=1000.0, neigh_min_threshold=5, rho_min_frac=0.7, rho_max_frac=1.5, v_limit=100.0)
    s = _state()
    s.rho[:] = np.array([600.0, 1700.0], dtype=np.float64)
    s.vel[:] = np.array([[120.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    m = StepMetrics(
        step=1,
        time=0.0,
        dt=1e-4,
        v_max=120.0,
        rho_min=600.0,
        rho_avg=1150.0,
        rho_max=1700.0,
        p_min=0.0,
        p_avg=0.0,
        p_max=0.0,
        neigh_min=3,
        neigh_avg=10.0,
        neigh_max=20,
    )
    flags = d.evaluate(m, s)
    assert "LOW_NEIGHBORS" in flags
    assert "LOW_DENSITY" in flags
    assert "HIGH_DENSITY" in flags
    assert "HIGH_VELOCITY" in flags

