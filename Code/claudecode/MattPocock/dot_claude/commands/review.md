# /project:review — Security-First Code Review

Performs a thorough code review with file:line references and actionable feedback.

## Usage
```
/project:review                    # Review all staged/uncommitted changes
/project:review <file>             # Review a specific file
/project:review src/mychatbot/     # Review a directory
```

## Review Checklist

### 1. Security (highest priority)
- [ ] No hardcoded secrets, tokens, or credentials
- [ ] All external inputs validated with Pydantic before use
- [ ] No sensitive data in log statements
- [ ] File paths sanitized before `open()`
- [ ] No SQL or shell injection vectors

### 2. Correctness
- [ ] Logic matches the stated intent (check docstrings and function names)
- [ ] Edge cases handled: empty inputs, None values, empty lists
- [ ] Error states handled and logged correctly
- [ ] Async functions properly awaited; no forgotten `await`

### 3. Project Conventions (see CLAUDE.md)
- [ ] Type hints on all function signatures
- [ ] Uses project logger, not `print()` or `logging.basicConfig()`
- [ ] Pydantic models for request/response schemas
- [ ] Conventional commit message if reviewing a commit

### 4. Test Coverage
- [ ] New functions have corresponding tests
- [ ] Edge cases are covered in tests
- [ ] Mocks used for external calls (Groq API)

## Output Format
For each issue found:
```
[SEVERITY] file.py:line — Description of issue
Suggestion: <specific fix>
```

Severity levels: `CRITICAL` (security), `HIGH` (correctness), `MEDIUM` (conventions), `LOW` (style).

Only flag `MEDIUM` and `LOW` issues if there are no `CRITICAL` or `HIGH` issues.
