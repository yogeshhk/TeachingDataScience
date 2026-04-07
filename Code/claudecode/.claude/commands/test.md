# /project:test — Run Tests and Optionally Fix Failures

Runs the pytest suite, analyzes failures, and optionally applies fixes.

## Usage
```
/project:test              # Run full suite
/project:test <path>       # Run specific test file or directory
/project:test --fix        # Run suite and auto-fix failures
/project:test <path> --fix # Run specific tests and auto-fix
```

## Behavior

### Without `--fix`
1. Run: `pytest <path or tests/> -v --tb=short`
2. For each failure:
   - Show the test name, assertion that failed, and actual vs expected values.
   - Trace the failure to the source function (not the test itself).
   - Explain the root cause in plain language.
3. Summarize: X passed, Y failed, Z errors.
4. Ask whether to apply fixes.

### With `--fix`
1. Run the suite as above.
2. For each failure, determine whether:
   - The **test** is wrong (wrong expectation, stale fixture) → fix the test.
   - The **source code** is wrong → fix the source and note it.
3. Apply minimal fixes only — no unrelated changes.
4. Re-run the suite after fixes to confirm all pass.
5. Show a summary of what was fixed and why.

## Test Conventions (this project)
- Tests live in `tests/` mirroring `src/mychatbot/` structure.
- Use `Faker` for synthetic test data — never hardcode realistic PII.
- Fixtures are defined in `tests/conftest.py`.
- Mock external API calls (Groq) using `pytest-mock`.
- Test naming: `test_<function_name>_<scenario>` (e.g., `test_build_chain_returns_runnable`).
