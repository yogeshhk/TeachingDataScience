"""Geometry inspection engine for stlinspector.

Loads a triangle mesh and reports its basic geometric properties plus
any validation issues (not watertight, non-manifold edges, degenerate
faces). Never auto-repairs the mesh before inspecting it -- that would
hide the defects this tool exists to detect.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from config import Config, DEFAULT_CONFIG


@dataclass
class InspectionReport:
    """Result of inspecting a single mesh."""

    bounding_box: tuple[list[float], list[float]]
    volume: float
    surface_area: float
    triangle_count: int
    issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        return {
            "bounding_box": {
                "min": self.bounding_box[0],
                "max": self.bounding_box[1],
            },
            "volume": self.volume,
            "surface_area": self.surface_area,
            "triangle_count": self.triangle_count,
            "issues": self.issues,
            "is_valid": self.is_valid,
        }


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a triangle mesh from a part file.

    Args:
        path: Path to an STL (or other trimesh-supported) part file.

    Returns:
        The loaded triangle mesh.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file could not be parsed as a triangle mesh.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not parse a triangle mesh from: {path}")
    return mesh


def _has_non_manifold_edges(mesh: trimesh.Trimesh) -> bool:
    """True if any edge is shared by more than 2 faces."""
    edge_counts = Counter(map(tuple, mesh.edges_sorted))
    return any(count > 2 for count in edge_counts.values())


def _has_degenerate_faces(mesh: trimesh.Trimesh, threshold: float) -> bool:
    """True if any face has area below threshold (near-zero/zero area)."""
    if len(mesh.faces) == 0:
        return False
    return bool(np.any(mesh.area_faces < threshold))


def _check_thin_walls(mesh: trimesh.Trimesh, min_thickness: float) -> bool:
    """True if any wall is thinner than min_thickness.

    Casts a ray inward from each face's centroid (along its inward
    normal) and measures the distance to the mesh surface on the far
    side. A short hit distance means the local wall is thin there.
    """
    if len(mesh.faces) == 0:
        return False

    origins = mesh.triangles_center
    directions = -mesh.face_normals
    # Nudge origins off the surface along the ray direction so the
    # ray doesn't immediately re-intersect its own starting face.
    nudged_origins = origins + directions * 1e-6

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=nudged_origins,
        ray_directions=directions,
        multiple_hits=False,
    )
    if len(locations) == 0:
        return False

    thicknesses = np.linalg.norm(locations - origins[index_ray], axis=1)
    return bool(np.any(thicknesses < min_thickness))


def inspect_mesh(mesh: trimesh.Trimesh, config: Config = DEFAULT_CONFIG) -> InspectionReport:
    """Inspect an already-loaded mesh and report its properties and validation issues.

    Args:
        mesh: The triangle mesh to inspect (see load_mesh).
        config: Validation thresholds to apply.

    Returns:
        An InspectionReport describing the mesh and any issues found.
    """
    issues: list[str] = []
    if not mesh.is_watertight:
        issues.append("not_watertight")
    if _has_non_manifold_edges(mesh):
        issues.append("non_manifold_edges")
    if _has_degenerate_faces(mesh, config.degenerate_face_area_threshold):
        issues.append("degenerate_faces")
    if _check_thin_walls(mesh, config.min_wall_thickness):
        issues.append("thin_walls")

    bounds = mesh.bounds
    return InspectionReport(
        bounding_box=(bounds[0].tolist(), bounds[1].tolist()),
        volume=float(mesh.volume),
        surface_area=float(mesh.area),
        triangle_count=int(len(mesh.faces)),
        issues=issues,
    )
