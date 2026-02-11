import numpy as np

from sph.neighbors.spatial_hash import SpatialHash


def test_neighbor_query_only_returns_particles_within_support_radius():
    """
    Neighbor search should only return particles within distance <= h
    (compact support neighborhood assumption used throughout the tutorial).
    """
    dim = 2
    h = 1.0

    # 3 points: p0 at origin, p1 within radius, p2 outside radius
    pos = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [2.0, 0.0],
    ], dtype=np.float64)

    ns = SpatialHash(support_radius=h, dim=dim)
    ns.build(pos)

    nbs = ns.query(0, pos)
    assert 1 in nbs
    assert 2 not in nbs


def test_neighbor_query_is_deterministic_for_same_positions():
    """
    For reproducible simulations we want deterministic neighbor lists
    given the same positions and build order.
    """
    dim = 2
    h = 1.0

    pos = np.array([
        [0.0, 0.0],
        [0.8, 0.0],
        [0.0, 0.8],
        [0.8, 0.8],
    ], dtype=np.float64)

    ns = SpatialHash(support_radius=h, dim=dim)
    ns.build(pos)
    nbs1 = ns.query(0, pos)

    ns.build(pos)
    nbs2 = ns.query(0, pos)

    assert nbs1 == nbs2
