# /project:refactor — Safe Refactoring

Restructures code safely with impact analysis across all call sites before touching anything.

## Usage
```
/project:refactor <file or function> [goal]
```

Examples:
- `/project:refactor src/mychatbot/chain.py extract build_chain into separate module`
- `/project:refactor src/mychatbot/cli.py reduce function length`

## Behavior

### Phase 1 — Impact Analysis (no edits yet)
1. Identify all callers of the target function/module using `Grep` and `Glob`.
2. List every file that imports or calls the code to be changed.
3. Identify any tests that cover the target code.
4. Report the full impact surface and wait for confirmation before proceeding.

### Phase 2 — Plan
1. Describe the proposed changes in plain language.
2. List every file that will be modified and why.
3. Show the new function/module signatures before writing code.
4. Wait for approval.

### Phase 3 — Apply
1. Make changes in dependency order (lowest-level modules first).
2. Update all call sites found in Phase 1.
3. Update imports across all affected files.
4. Do NOT change behavior — refactoring only, no new features.

### Phase 4 — Verify
1. Run `pytest tests/ -q` and report results.
2. If tests fail, show which test broke and why, then fix.

## Rules
- Never combine refactoring with feature changes in the same pass.
- Never rename public API functions without updating all call sites.
- Always run tests after applying changes.
