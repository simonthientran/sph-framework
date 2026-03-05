import numpy as np

from sph.geometry.mesh_sampler import sample_triangle_surface


def test_sample_triangle_surface_points_inside_triangle():
    np.random.seed(123)
    triangles = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )

    pts = sample_triangle_surface(triangles, spacing=0.1)
    assert pts.shape[1] == 3
    assert pts.shape[0] > 0

    # For this right triangle, valid points satisfy x>=0, y>=0, x+y<=1, z=0.
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    eps = 1e-12
    assert np.all(x >= -eps)
    assert np.all(y >= -eps)
    assert np.all((x + y) <= 1.0 + eps)
    assert np.all(np.abs(z) <= eps)

