name: refactor-patterns
description: >
  Patterns and criteria for safe refactoring of Python code.
  Use when asked to refactor or improve code quality.
---
## Refactoring Rules

### Safety First
- Never change public API signatures without updating
  all callers and the spec doc
- Run tests before and after to confirm no regression

### Python Improvements to Look For
- Replace dict literals for message objects with
  a TypedDict or dataclass
- Extract magic strings (model names, env var names)
  into constants at module level
- Replace print() with logging module (configurable level)
- Move configuration into a Config dataclass
  (not scattered kwargs)

### Code Smell Checklist
- Functions longer than 40 lines: consider splitting
- Duplicate code across cli.py and app.py: extract to utils
- Nested if/else depth > 3: refactor to early returns