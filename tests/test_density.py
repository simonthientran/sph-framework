import numpy as np

from sph.core.state_builder import build_fluid_block
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.density import compute_density_summation


def test_density_summation_reasonable_in_block_interior():
    """
    Density is reconstructed by summation:
        rho_i = sum_j m_j W_ij
    (used in the tutorial's basic SPH simulation loop, Algorithm 1 / Eq. (11)).

    We do a sanity check: for a regular block of particles, an interior particle
    should have rho roughly close to rho0 (within a tolerance).
    """
    scene = {
        "meta": {"name": "test_block", "version": 1, "seed": 0, "dimensions": 2},
        "fluid": {
            "type": "block",
            "min": [0.0, 0.0],
            "max": [0.2, 0.2],
            "spacing": 0.02,
            "initial_velocity": [0.0, 0.0],
        },
        "material": {"rho0": 1000.0, "viscosity": 0.01, "eos": {"type": "tait", "gamma": 7.0, "c0": 20.0}},
        "neighbors": {"type": "spatial_hash", "support_radius": 0.04},
        "time": {"mode": "cfl", "dt_fixed": 0.0005, "cfl": 0.4, "steps": 1},
        "domain": {"type": "box", "min": [0.0, 0.0], "max": [1.0, 1.0]},
    }

    state = build_fluid_block(scene)
    h = float(scene["neighbors"]["support_radius"])
    rho0 = float(scene["material"]["rho0"])

    ns = SpatialHash(support_radius=h, dim=state.dim)
    ns.build(state.pos)

    rho = compute_density_summation(state=state, neighbor_search=ns, h=h)

    # pick an interior-ish particle: closest to the block center
    center = np.array([0.1, 0.1], dtype=np.float64)
    i = int(np.argmin(np.linalg.norm(state.pos - center[None, :], axis=1)))

    # interior should be reasonably close. Tolerance is intentionally not too strict
    # because exact value depends on h/spacing ratio and kernel discretization error.
    assert np.isfinite(rho[i])
    assert np.isclose(rho[i], rho0, rtol=0.20, atol=0.0)
