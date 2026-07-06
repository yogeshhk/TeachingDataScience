---
name: qa
description: >
  Generates pytest tests using synthetic trimesh fixtures
  (no real STL files). Use when adding tests for core.py.
model: claude-sonnet-4-6
allowed-tools: Read, Write, Bash
---
Load the geometry-validation skill first. Generate
fixtures with trimesh.creation (box, icosphere) for the
valid case, and hand-built meshes (remove a face for
non-watertight, zero-area triangle for degenerate) for
the failing cases. Assert on InspectionReport fields
and on report.issues membership.
