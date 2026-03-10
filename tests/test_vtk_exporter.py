import numpy as np

from sph.core.state import ParticleState
from sph.io.exports import ExportManager
from sph.io.vtk_export import export_particles_vtk_legacy


def _small_state() -> ParticleState:
    pos = np.array([[0.0, 0.0], [0.02, 0.0], [0.04, 0.01]], dtype=np.float64)
    vel = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]], dtype=np.float64)
    acc = np.zeros_like(pos)
    mass = np.ones((3,), dtype=np.float64)
    rho = np.array([1000.0, 995.0, 1010.0], dtype=np.float64)
    p = np.array([0.0, 10.0, -5.0], dtype=np.float64)
    is_boundary = np.array([False, False, True], dtype=np.bool_)
    return ParticleState(dim=2, pos=pos, vel=vel, acc=acc, mass=mass, rho=rho, p=p, is_boundary=is_boundary)


def test_vtk_export_contains_point_count_and_neighbor_field(tmp_path):
    state = _small_state()
    out = tmp_path / "particles_000001.vtk"
    export_particles_vtk_legacy(out, state, neighbor_count=np.array([10, 12, 8], dtype=np.int64))
    text = out.read_text(encoding="utf-8")
    assert "POINTS 3 float" in text
    assert "SCALARS neighbor_count int 1" in text
    assert "VECTORS velocity float" in text


def test_export_manager_writes_file_for_matching_step(tmp_path):
    state = _small_state()
    m = ExportManager(vtk_enabled=True, vtk_every=2, vtk_dir=tmp_path)
    m.maybe_export_initial(state)
    m.maybe_export(step=1, state=state, neighbor_count=np.array([1, 2, 3], dtype=np.int64))
    m.maybe_export(step=2, state=state, neighbor_count=np.array([1, 2, 3], dtype=np.int64))
    assert (tmp_path / "particles_000000.vtk").exists()
    assert not (tmp_path / "particles_000001.vtk").exists()
    assert (tmp_path / "particles_000002.vtk").exists()

