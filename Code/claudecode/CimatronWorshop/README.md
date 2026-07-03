# stlinspector

STEP/STL geometry inspector and validator -- a PoC built end-to-end in a
Claude Code workshop (see `LaTeX/ai_tools_claudecode_demo_cimatron.tex` in
the parent repo for the full walkthrough).

> **Note:** the deck was revised after this code was built and now
> describes two things not yet implemented here: (1) `cli.py` writing a
> Markdown report to `reports/<stem>.md` alongside the JSON report, and
> (2) a `.claude/skills/inspection-report-summary/` skill that summarizes
> that report, replacing `.claude/skills/geometry-validation/` (still
> present below). The deck's DevOps Subagent frame is also currently
> commented out for the teaching session, but this repo's
> `.claude/agents/devops.md` and the `inspect` console-script entry point
> in `pyproject.toml` are untouched.

## What it does
- Loads an STL part file (`trimesh`)
- Computes bounding box, volume, surface area, triangle count
- Flags: not watertight, non-manifold edges, degenerate faces

## Install
```bash
conda env create -f environment.yml
conda activate stlinspector
pip install -e ".[dev]"
```

## CLI
```bash
inspect part.stl --report out.json
```

## Streamlit
```bash
streamlit run src/stlinspector/app.py
```

## Tests
```bash
pytest tests/ -v --cov=src/stlinspector --cov-report=term-missing
```
