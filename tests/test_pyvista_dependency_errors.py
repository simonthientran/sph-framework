from __future__ import annotations

import pytest


def test_scene_viewer_raises_clear_error_without_pyvista() -> None:
    try:
        import pyvista  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("PyVista is installed; missing-dependency path is not active.")

    from sph.visualization.pyvista_viewer import PyVistaSceneViewer

    with pytest.raises(ImportError, match="PyVista is required for 3-D visualization"):
        PyVistaSceneViewer("scenes/examples/box_fill_3d.json")


def test_playback_viewer_raises_clear_error_without_pyvista() -> None:
    try:
        import pyvista  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("PyVista is installed; missing-dependency path is not active.")

    from sph.visualization.pyvista_playback import PyVistaPlaybackViewer

    with pytest.raises(ImportError, match="PyVista is required for 3-D visualization"):
        PyVistaPlaybackViewer("/tmp/nonexistent.vtk")
