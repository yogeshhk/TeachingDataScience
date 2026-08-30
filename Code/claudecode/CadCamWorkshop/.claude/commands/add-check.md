---
description: >
  Scaffold a new geometry validation rule in core.py,
  wired into InspectionReport.issues, plus a matching
  synthetic-fixture test. Pass the rule name as argument.
allowed-tools: Read, Edit, Write, Bash
---
Add a new validation check named "$ARGUMENTS" to
src/core.py:
1. Add a private _check_$ARGUMENTS(mesh) -> bool helper
2. Wire it into inspect_mesh(), appending "$ARGUMENTS" to
   issues when it returns True
3. Add a synthetic-mesh test in tests/test_core.py that
   triggers the new check
4. Run pytest tests/ -v and report the result
