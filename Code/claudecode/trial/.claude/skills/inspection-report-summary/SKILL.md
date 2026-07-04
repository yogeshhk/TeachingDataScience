---
name: inspection-report-summary
description: >
  Summarizes an stlinspector InspectionReport (JSON)
  for a non-technical stakeholder. Use when asked to
  "summarize", "explain", or "triage" an inspection
  report.
---

## Input
Read the JSON report written by --report (e.g. out.json,
or the path given): bbox, volume, surface_area,
triangle_count, and a list of issues (if any).

## Output
- One-paragraph plain-English verdict: PASS / FAIL and why
- If issues present, rank by severity:
  1. not_watertight (blocks 3D printing / CAM)
  2. non_manifold_edges (blocks CAM toolpath generation)
  3. degenerate_faces (usually cosmetic, low risk)
- Suggest the next concrete action per issue found
  (e.g. "re-export from CAD with tighter tessellation")