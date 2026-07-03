# stlinspector -- Technical Specification

## Overview
stlinspector is a PoC geometry inspector and validator for STL part
files (STEP support deferred). It exposes a CLI and a Streamlit web app
over a shared inspection engine.

## Goals
- Parse an STL file with trimesh
- Report bounding box, volume, surface area, triangle count
- Flag not-watertight, non-manifold edges, degenerate faces

## Non-Goals
- STEP/IGES parsing (future work)
- Mesh repair (out of scope -- inspection only)

## Functional Requirements
- `core.inspect(path) -> InspectionReport`
- `cli.py`: `inspect <file> [--report out.json]`, non-zero exit code if
  any issue was found
- `app.py`: upload -> 3D preview (plotly) -> report table

## Data Models
    InspectionReport:
      file_path: str
      bounding_box: tuple[list[float], list[float]]
      volume: float
      surface_area: float
      triangle_count: int
      issues: list[str]

## API / CLI Contracts
- `load_mesh(path) -> trimesh.Trimesh`
- `inspect(path, config=DEFAULT_CONFIG) -> InspectionReport`
- CLI: `inspect part.stl --report out.json`

## Open Questions
- STEP file support: parser choice (deferred)
- Wall-thickness check: needs a signed-distance or ray-casting approach
  (left as a `/add-check` exercise)
