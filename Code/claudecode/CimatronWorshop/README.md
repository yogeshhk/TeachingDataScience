# stlinspector

STEP/STL geometry inspector and validator -- a PoC built end-to-end in a
Claude Code workshop (see `LaTeX/ai_tools_claudecode_demo_cimatron.tex` in
the parent repo for the full walkthrough).

## What it does
- Loads an STL part file (`trimesh`)
- Computes bounding box, volume, surface area, triangle count
- Flags: not watertight, non-manifold edges, degenerate faces, thin walls

## Install
```bash
conda env create -f environment.yml
conda activate stlinspector
```

No packaging (no `pyproject.toml`, no editable install, no console
script, no package folder) is used here -- `src/` holds plain scripts.
Running them directly puts `src/` on `sys.path` automatically; only
`tests/conftest.py` needs an explicit `sys.path` shim.

## CLI
```bash
python src/cli.py part.stl --report out.json
```

## Streamlit
```bash
streamlit run src/app.py
```

## Tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```
