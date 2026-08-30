# stlinspector — Technical Specification

## Overview

`stlinspector` is a PoC tool for inspecting and validating STL mesh files. It loads a mesh, computes basic geometric properties, runs a set of manufacturability/validity checks, and surfaces the results through a CLI and a Streamlit app with 3D preview. Built end-to-end in a Claude Code workshop (see `LaTeX/ai_tools_claudecode_demo_cadcam.tex` in the parent repo).

Stack: Python 3.11, `trimesh`, `numpy`, Streamlit + Plotly (3D preview), Rich (CLI output).

## Goals

- Load a mesh file and compute bounding box, volume, surface area, and triangle count.
- Detect geometry defects: non-watertight meshes, non-manifold edges, degenerate faces, and thin walls.
- Provide a scriptable CLI that emits a JSON report.
- Provide an interactive Streamlit app: upload a file, preview it in 3D, view the report.

## Non-Goals

- No mesh repair/fixing — inspection/reporting only.
- No STEP/IGES parsing (future work).
- No batch/directory scanning in the CLI — single file per invocation.
- No authentication, persistence, or multi-user support in the Streamlit app.
- No packaging (`pyproject.toml`, editable install, console script) — `src/` holds plain scripts, run directly; only `tests/conftest.py` needs a `sys.path` shim.

## Functional Requirements

### `core.py`

- `load_mesh(path: str | Path) -> trimesh.Trimesh`
  - Loads via `trimesh.load(path, force="mesh")`.
  - Raises `FileNotFoundError` if `path` does not exist.
  - Raises `ValueError` if the file loads but isn't a triangle mesh.
- `inspect_mesh(mesh: trimesh.Trimesh, config: Config = DEFAULT_CONFIG) -> InspectionReport`
  - Takes an already-loaded mesh — two-step API, caller calls `load_mesh` first.
  - Never raises on a "bad" mesh — defects are reported via `issues`, not exceptions.

### Validation rules

| Rule | Check | Config threshold |
|---|---|---|
| `not_watertight` | `mesh.is_watertight is False` | — |
| `non_manifold_edges` | any edge shared by more than 2 faces | — |
| `degenerate_faces` | any face area below threshold | `degenerate_face_area_threshold` (default `1e-8`) |
| `thin_walls` | inward ray-cast from any face centroid hits the surface closer than threshold | `min_wall_thickness` (default `1.0`) |

A mesh with zero issues is considered valid (`is_valid` property). All four rules run independently — one failing rule doesn't block the others.

### `cli.py`

```
python src/cli.py <file> [--report out.json]
```

No console-script entry point — run directly.

- `<file>`: path to the mesh file to inspect (required, positional).
- `--report out.json`: optional path to write the full report as JSON (the only report format).
- Behavior:
  - Prints a Rich-formatted table (bounding box/volume/area/triangle count) to stdout, plus a red issue list or green "All checks passed".
  - Exit code `0` if no issues, `1` if any issue found, `2` on load failure (`FileNotFoundError`/`ValueError`).

### `app.py`

Streamlit single-page app:

1. **Upload**: `st.file_uploader` restricted to `.stl`. Written to a temp path.
2. **Inspect**: `load_mesh` then `inspect_mesh` on the same mesh object (also reused for the 3D preview — no redundant reload).
3. **3D preview**: Plotly `go.Mesh3d` (mesh vertices/faces), via `st.plotly_chart`.
4. **Report**: `st.metric` tiles for bounding box/volume/area/triangle count, `st.error` per issue or `st.success` if none.
5. Load/inspection errors shown via `st.error`, not raised — the app doesn't crash on bad input.

## Data Models

```python
@dataclass
class InspectionReport:
    bounding_box: tuple[list[float], list[float]]  # (min_xyz, max_xyz)
    volume: float
    surface_area: float
    triangle_count: int
    issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues
```

`issues` is a flat list of rule names (no per-issue severity field) — e.g. `["not_watertight", "thin_walls"]`. `InspectionReport.to_dict()` serializes this plus a computed `is_valid`, used for the `--report` JSON output.

## API / CLI Contracts

- `core.load_mesh(path: str | Path) -> trimesh.Trimesh`
- `core.inspect_mesh(mesh: trimesh.Trimesh, config: Config = DEFAULT_CONFIG) -> InspectionReport`
- CLI: `python src/cli.py <file> [--report out.json]` → exit codes `0` (valid), `1` (issues found), `2` (load failure)
- JSON report shape (`--report`, via `InspectionReport.to_dict()`):
  ```json
  {
    "bounding_box": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
    "volume": 0.0,
    "surface_area": 0.0,
    "triangle_count": 0,
    "issues": ["not_watertight"],
    "is_valid": false
  }
  ```
- The `inspection-report-summary` skill reads this JSON directly and produces a plain-English, severity-ranked summary for non-technical stakeholders — no separate Markdown report format.

## Open Questions

- Should `issues` carry per-rule severity (error vs. warning), or stay a flat list as it is now?
- Is a size/triangle-count cap needed for the Streamlit upload path, to avoid loading huge meshes in-browser?
- STEP file support: separate loader entirely, or extend trimesh's loader?
- `thin_walls` casts one ray per face — fine for these synthetic fixtures, but worth profiling before pointing it at a dense real-world mesh.
