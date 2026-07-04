# CLAUDE.md

## Project: stlinspector
STL/STEP geometry inspector and validator (PoC).

## Tech Stack
- Python 3.11, trimesh, numpy
- Streamlit + plotly (3D preview), Rich (CLI output)
- pytest for tests (synthetic fixtures only -- no real STL sample files
  committed to the repo)

## Build Commands
- Install: `conda env create -f environment.yml`
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
- src/core.py   : geometry inspection engine
- src/cli.py    : CLI entry point (`python src/cli.py`, no console script)
- src/app.py    : Streamlit web app
- src/config.py : Config dataclass (validation thresholds)
- tests/                     : pytest suite (synthetic fixtures only)
- docs/specs/                : feature specs
