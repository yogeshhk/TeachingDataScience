# stlinspector — Technical Spec

## Overview

`stlinspector` is a PoC tool for inspecting and validating 3D mesh files (STL, and eventually STEP). It loads a mesh, computes basic geometric properties, runs a set of manufacturability/validity checks, and surfaces the results through a CLI and a Streamlit app with 3D preview.

Stack: Python 3.11, `trimesh`, `numpy`, Streamlit + Plotly (3D preview), Rich (CLI output).

## Goals

- Load a mesh file and compute bounding box, volume, surface area, and triangle count.
- Detect common geometry defects: non-watertight meshes, non-manifold edges, degenerate faces.
- Provide a scriptable CLI (`inspect <file>`) that can emit a JSON report.
- Provide an interactive Streamlit app: upload a file, preview it in 3D, view the report as a table.

## Non-Goals

- No mesh repair/fixing — this is inspection/reporting only.
- No STEP/BREP parsing in this pass (STL only for v1); STEP support is a future extension, not implemented here.
- No batch/directory scanning in the CLI — single file per invocation.
- No authentication, persistence, or multi-user support in the Streamlit app.

## Functional Requirements

### `core.py`

- `load_mesh(path: str | Path) -> trimesh.Trimesh`
  - Loads the mesh via `trimesh.load(path, force="mesh")`.
  - Raises `FileNotFoundError` if `path` does not exist.
  - Raises `ValueError` if the file loads but yields zero vertices/faces (empty or unparseable mesh).

- `inspect_mesh(mesh: trimesh.Trimesh) -> InspectionReport`
  - Computes geometric properties and runs all validation rules (below).
  - Never raises on a "bad" mesh — defects are reported via `InspectionReport.issues`, not exceptions. Only I/O/parse failures in `load_mesh` raise.

### Validation rules

Each rule returns zero or more `Issue` entries appended to `InspectionReport.issues`. Rules run independently — one failing rule does not block the others.

| Rule | Check | Severity |
|---|---|---|
| `not-watertight` | `mesh.is_watertight is False` | error |
| `non-manifold-edges` | edges shared by ≠ 2 faces (`mesh.edges_sorted` grouped by count ≠ 2) | error |
| `degenerate-faces` | faces with near-zero area, i.e. `mesh.area_faces[i] < eps` (default `eps = 1e-9`), or faces with a repeated vertex index | warning |

A mesh with zero issues is considered valid. `not-watertight` and `non-manifold-edges` are correctness errors (volume/downstream operations are unreliable); `degenerate-faces` is a warning since it doesn't always block downstream use.

### `cli.py`

```
python cli.py <file> [--report out.json]
```

No console-script entry point is installed in this pass (that would be a packaging/DevOps step, out of scope here) — the CLI is invoked by running the script directly, not via a bare `inspect` command.

- `<file>`: path to the mesh file to inspect (required, positional).
- `--report out.json`: optional path to write the full `InspectionReport` as JSON. If omitted, no file is written.
- Behavior:
  - Always prints a Rich-formatted summary table to stdout (bbox, volume, surface area, triangle count, and a list of issues with severity).
  - Exit code `0` if no `error`-severity issues, `1` if at least one `error`-severity issue is present. `warning`-only reports still exit `0`.
  - On load failure (`FileNotFoundError`/`ValueError` from `load_mesh`), print the error to stderr and exit `2`.

### `app.py`

Streamlit single-page app:

1. **Upload**: `st.file_uploader` restricted to `.stl`. Uploaded file is written to a temp path and passed to `load_mesh`.
2. **3D preview**: render the mesh with Plotly `go.Mesh3d` (vertices/faces from the loaded `trimesh.Trimesh`), embedded via `st.plotly_chart`.
3. **Report table**: run `inspect_mesh`, render `InspectionReport` fields (bbox, volume, surface_area, triangle_count) as a summary block, and `issues` as an `st.dataframe` table (columns: rule, severity, message, optional face/edge indices).
4. Load or inspection errors are shown via `st.error` rather than raising, so the app doesn't crash on bad input.

## Data Models

```python
@dataclass
class Issue:
    rule: str            # e.g. "not-watertight", "non-manifold-edges", "degenerate-faces"
    severity: str        # "error" | "warning"
    message: str
    indices: list[int] | None = None   # offending face/edge indices, when applicable

@dataclass
class InspectionReport:
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]]  # (min_xyz, max_xyz)
    volume: float
    surface_area: float
    triangle_count: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)
```

`InspectionReport` must be JSON-serializable (via `dataclasses.asdict`) for the `--report` CLI flag and for any future JSON-report tooling (e.g. a summary skill consuming it directly — no separate Markdown report format).

## API / CLI Contracts

- `core.load_mesh(path: str | Path) -> trimesh.Trimesh`
- `core.inspect_mesh(mesh: trimesh.Trimesh) -> InspectionReport`
- CLI: `python cli.py <file> [--report out.json]` → exit codes `0` (valid), `1` (errors found), `2` (load failure)
- JSON report shape (written by `--report`): `dataclasses.asdict(InspectionReport)`, i.e.:
  ```json
  {
    "bbox": [[minx, miny, minz], [maxx, maxy, maxz]],
    "volume": 0.0,
    "surface_area": 0.0,
    "triangle_count": 0,
    "issues": [
      {"rule": "not-watertight", "severity": "error", "message": "...", "indices": null}
    ]
  }
  ```

## Open Questions

- Should `degenerate-faces` severity be configurable (some pipelines may treat it as an error)?
- Is a size/triangle-count cap needed for the Streamlit upload path to avoid loading huge meshes in-browser?
- Should the JSON report schema be versioned (e.g. a top-level `"schema_version"` key) given the PoC's report format has already drifted once against the paired teaching deck?
- STEP file support: separate loader entirely, or can `trimesh`'s loader be extended to cover it in v2?
