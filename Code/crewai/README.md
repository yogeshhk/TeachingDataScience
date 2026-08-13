# CrewAI Examples

Three standalone CrewAI demo scripts (job matching, research blogging, trip planning),
moved here from `Code/agents/` to consolidate all CrewAI content in one place:

```bash
conda env create -f environment.yml
conda activate crewai-scripts
python crew_job_matching_open_source.py
```

`researcher/` is a separate, more complete CrewAI project with its own `pyproject.toml` +
`uv.lock` (modern uv workflow) — see its own README for setup.
