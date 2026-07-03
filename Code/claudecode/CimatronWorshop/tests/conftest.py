"""Shared pytest fixtures: synthetic mesh files only, no real STL samples."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh


def _write_stl(mesh: trimesh.Trimesh) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    tmp.close()
    mesh.export(tmp.name)
    return Path(tmp.name)


@pytest.fixture
def valid_box_mesh_path():
    """A watertight, manifold, non-degenerate 10x10x10 box."""
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    path = _write_stl(mesh)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def non_watertight_mesh_path():
    """A box with one triangle removed, leaving an open hole."""
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    mesh.faces = mesh.faces[1:]
    mesh.remove_unreferenced_vertices()
    path = _write_stl(mesh)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def degenerate_face_mesh_path():
    """A box with an extra zero-area (self-referential) triangle appended."""
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    faces = np.vstack([mesh.faces, [0, 0, 0]])
    degenerate_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=faces, process=False)
    path = _write_stl(degenerate_mesh)
    yield path
    path.unlink(missing_ok=True)
