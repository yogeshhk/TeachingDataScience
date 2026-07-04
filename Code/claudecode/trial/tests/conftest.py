import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import trimesh


def _write_stl(mesh: trimesh.Trimesh) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    tmp.close()
    mesh.export(tmp.name)
    return Path(tmp.name)


@pytest.fixture
def thin_wall_mesh_path():
    """A 10x10x0.05 slab -- watertight and manifold, but far thinner
    than the default 1.0 min_wall_thickness."""
    mesh = trimesh.creation.box(extents=[10, 10, 0.05])
    path = _write_stl(mesh)
    yield path
    path.unlink(missing_ok=True)
