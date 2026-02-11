import numpy as np

from sph.sph.kernels import cubic_spline_W


def test_cubic_spline_support_is_compact():
    """
    Cubic spline kernel has compact support: W(r,h)=0 for ||r||/h > 1.
    This matches the tutorial's cubic spline definition (Eq. (4)).
    """
    h = 0.04
    dim = 2

    # inside support
    w_inside = cubic_spline_W(np.array([0.5 * h, 0.0]), h=h, dim=dim)
    assert w_inside > 0.0

    # exactly at boundary q=1 is allowed (may be 0 depending on the formula branch)
    w_boundary = cubic_spline_W(np.array([1.0 * h, 0.0]), h=h, dim=dim)
    assert w_boundary >= 0.0

    # outside support
    w_outside = cubic_spline_W(np.array([1.01 * h, 0.0]), h=h, dim=dim)
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

    # random samples inside the support
    rng = np.random.default_rng(0)
    for _ in range(100):
        r = rng.uniform(-h, h, size=(dim,))
        if np.linalg.norm(r) <= h:
            assert cubic_spline_W(r, h=h, dim=dim) >= 0.0
