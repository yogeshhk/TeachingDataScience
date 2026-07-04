"""Tests for core -- synthetic fixtures only, no real STL files."""

from __future__ import annotations

import math

import pytest

from core import inspect_mesh, load_mesh


def test_valid_box_has_no_issues(valid_box_mesh_path):
    mesh = load_mesh(valid_box_mesh_path)
    report = inspect_mesh(mesh)

    assert report.issues == []
    assert report.is_valid
    assert report.triangle_count == 12
    assert math.isclose(report.volume, 1000.0, rel_tol=1e-6)


def test_non_watertight_mesh_flagged(non_watertight_mesh_path):
    mesh = load_mesh(non_watertight_mesh_path)
    report = inspect_mesh(mesh)

    assert "not_watertight" in report.issues
    assert not report.is_valid


def test_degenerate_face_mesh_flagged(degenerate_face_mesh_path):
    mesh = load_mesh(degenerate_face_mesh_path)
    report = inspect_mesh(mesh)

    assert "degenerate_faces" in report.issues
    assert not report.is_valid


def test_load_mesh_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.stl"
    with pytest.raises(FileNotFoundError):
        load_mesh(missing)


def test_thin_walls_flagged(thin_wall_mesh_path):
    mesh = load_mesh(thin_wall_mesh_path)
    report = inspect_mesh(mesh)

    assert "thin_walls" in report.issues
    assert not report.is_valid
