"""
Generate test STL files for SPH boundary testing.
Run once: python scripts/create_test_geometry.py
"""
from __future__ import annotations

from pathlib import Path

import trimesh

assets = Path(__file__).parent.parent / "assets"
assets.mkdir(exist_ok=True)

# 1. Simple box container (1m × 1m × 1m), shifted to [0, 1]^3
box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
box.apply_translation([0.5, 0.5, 0.5])
box.export(str(assets / "box.stl"))
print(f"Created {assets / 'box.stl'}  (watertight={box.is_watertight})")

# 2. Cylinder pipe (for pipe flow)
cylinder = trimesh.creation.cylinder(radius=0.1, height=1.0, sections=32)
cylinder.export(str(assets / "pipe.stl"))
print(f"Created {assets / 'pipe.stl'}  (watertight={cylinder.is_watertight})")

# 3. Sphere container
sphere = trimesh.creation.icosphere(radius=0.5, subdivisions=3)
sphere.export(str(assets / "sphere.stl"))
print(f"Created {assets / 'sphere.stl'}  (watertight={sphere.is_watertight})")

print("All test geometries created in assets/")
