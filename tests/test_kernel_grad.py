import numpy as np

from sph.sph.kernels import cubic_spline_gradW


def test_gradW_zero_outside_support():
    h = 0.04
    dim = 2
    g = cubic_spline_gradW(np.array([1.01 * h, 0.0]), h=h, dim=dim)
    assert np.allclose(g, 0.0)


def test_gradW_antisymmetric():
    h = 0.04
    dim = 2
    r = np.array([0.013, -0.007])
    g1 = cubic_spline_gradW(r, h=h, dim=dim)
    g2 = cubic_spline_gradW(-r, h=h, dim=dim)
    assert np.allclose(g1, -g2, atol=1e-12)
