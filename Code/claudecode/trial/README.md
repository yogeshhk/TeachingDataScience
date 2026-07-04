# stlinspector

STL geometry inspector and validator (PoC). Loads a mesh, checks for
not-watertight shells, non-manifold edges, and degenerate faces, and
reports the results via a CLI or a Streamlit app.

See `docs/specs/stlinspector-spec.md` for the full spec.

## Install

```bash
pip install trimesh numpy streamlit plotly pytest rich
```

No packaging (no `pyproject.toml`, no editable install) is used in
this pass -- `src/` holds plain scripts, run directly.

## Run

```bash
python src/cli.py <file.stl> [--report out.json]
streamlit run src/app.py
```

## Test

```bash
pytest tests/ -v
```
