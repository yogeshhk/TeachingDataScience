---
name: geometry-validation
description: >
  Domain knowledge for mesh validation thresholds and
  definitions. Use when adding or reviewing checks in
  core.py.
---
## Definitions
- Watertight: every edge is shared by exactly 2 faces,
  and the mesh has no boundary edges.
- Non-manifold edge: an edge shared by 3+ faces.
- Degenerate face: triangle area < 1e-8 (mesh units^2),
  or two shared vertices (zero-length edge).

## Units
Assume millimetres unless the file specifies otherwise.

## Tolerance
Use math.isclose with rel_tol=1e-6 for area/volume
comparisons in tests -- never exact float equality.
