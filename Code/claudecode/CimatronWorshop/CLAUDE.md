# CLAUDE.md

## Project: stlinspector
STL/STEP geometry inspector and validator (PoC).

## Tech Stack
- Python 3.11, trimesh, numpy
- Streamlit + plotly (3D preview), Rich (CLI output)
- pytest for tests (synthetic fixtures only -- no real STL sample files
  committed to the repo)

## Build Commands
- Install: `pip install -e ".[dev]"`
- Test:    `pytest tests/ -v`
- Lint:    `ruff check src/`

## Conventions
- Type hints on all functions
- Google-style docstrings on public functions
- core.py must not import Streamlit or any UI library
- Validation checks always operate on the raw loaded mesh -- never
  auto-repair before inspecting (that would hide the defects the tool
  is meant to detect)

## File Map
- src/stlinspector/core.py   : geometry inspection engine
- src/stlinspector/cli.py    : `inspect` CLI entry point
- src/stlinspector/app.py    : Streamlit web app
- src/stlinspector/config.py : Config dataclass (validation thresholds)
- tests/                     : pytest suite (synthetic fixtures only)
- docs/specs/                : feature specs
