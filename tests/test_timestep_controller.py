import numpy as np

from sph.core.diagnostics import StepMetrics
from sph.core.timestep import TimeStepController


def _metrics(dt: float, v_max: float, step: int = 1) -> StepMetrics:
    return StepMetrics(
        step=step,
        time=step * dt,
        dt=dt,
        v_max=v_max,
        rho_min=1000.0,
        rho_avg=1000.0,
        rho_max=1000.0,
        p_min=0.0,
        p_avg=0.0,
        p_max=0.0,
        neigh_min=20,
        neigh_avg=25.0,
        neigh_max=30,
    )


def test_timestep_controller_clamps_and_ramp_up_limit():
    c = TimeStepController(
        use_cfl=True,
        cfl=0.4,
        h=0.045,
        dt_min=1e-5,
        dt_max=5e-4,
        ramp_up_max=1.2,
    )
    m1 = _metrics(dt=2e-3, v_max=1.0, step=1)
    dt1 = c.update(m1)
    assert np.isclose(dt1, 5e-4)
    assert "CLAMP_MAX" in m1.dt_reason_codes

    m2 = _metrics(dt=5e-4, v_max=50.0, step=2)
    dt2 = c.update(m2)
    assert np.isclose(dt2, 0.4 * 0.045 / 50.0)
    assert "CFL_V" in m2.dt_reason_codes

    m3 = _metrics(dt=5e-4, v_max=2.0, step=3)
    dt3 = c.update(m3)
    assert dt3 <= dt2 * 1.2 + 1e-14
    assert "RAMP_UP_LIMIT" in m3.dt_reason_codes

