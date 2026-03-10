import numpy as np

from sph.sph.kernels import cubic_spline_W


def test_cubic_spline_support_is_compact():
    """
    Cubic spline kernel has compact support: W(r,h)=0 for ||r||/h > 2.
    Convention: support_radius = 2h (Monaghan cubic spline).
    """
    h = 0.04
    dim = 2

    # inside support (q < 2)
    w_inside = cubic_spline_W(np.array([0.5 * h, 0.0]), h=h, dim=dim)
    assert w_inside > 0.0

    # at boundary q=2
    w_boundary = cubic_spline_W(np.array([2.0 * h, 0.0]), h=h, dim=dim)
    assert w_boundary >= 0.0

    # outside support (q > 2)
    w_outside = cubic_spline_W(np.array([2.01 * h, 0.0]), h=h, dim=dim)
    assert w_outside == 0.0


def test_cubic_spline_is_symmetric():
    """
    Kernel depends only on ||r||, therefore W(r)=W(-r).
    """
    h = 0.04
    dim = 2
    r = np.array([0.013, -0.007])

    w1 = cubic_spline_W(r, h=h, dim=dim)
    w2 = cubic_spline_W(-r, h=h, dim=dim)
    assert np.isclose(w1, w2, rtol=0.0, atol=1e-14)


def test_cubic_spline_non_negative():
    """
    Smoothing kernels used for density summation must be non-negative.
    """
    h = 0.04
    dim = 2

    # random samples inside the support (q <= 2)
    rng = np.random.default_rng(0)
    for _ in range(100):
        r = rng.uniform(-2 * h, 2 * h, size=(dim,))
        if np.linalg.norm(r) <= 2 * h:
            assert cubic_spline_W(r, h=h, dim=dim) >= 0.0
