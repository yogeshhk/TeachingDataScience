"""Tests for stlinspector.core -- synthetic fixtures only, no real STL files."""

from __future__ import annotations

import math

import pytest

from stlinspector.core import inspect


def test_valid_box_has_no_issues(valid_box_mesh_path):
    report = inspect(valid_box_mesh_path)

    assert report.issues == []
    assert report.is_valid
    assert report.triangle_count == 12
    assert math.isclose(report.volume, 1000.0, rel_tol=1e-6)


def test_non_watertight_mesh_flagged(non_watertight_mesh_path):
    report = inspect(non_watertight_mesh_path)

    assert "not_watertight" in report.issues
    assert not report.is_valid


def test_degenerate_face_mesh_flagged(degenerate_face_mesh_path):
    report = inspect(degenerate_face_mesh_path)

    assert "degenerate_faces" in report.issues
    assert not report.is_valid


def test_inspect_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.stl"
    with pytest.raises(FileNotFoundError):
        inspect(missing)
