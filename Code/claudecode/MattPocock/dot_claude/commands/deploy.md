# /project:deploy — Pre-Deployment Checklist

Runs a pre-deployment verification for the target environment.

## Usage
```
/project:deploy staging
/project:deploy production
/project:deploy local
```

## Checklist — All Environments

### Code Quality
- [ ] `pytest tests/ -q` passes with 0 failures
- [ ] No `TODO` or `FIXME` comments in changed files
- [ ] No debug code (`breakpoint()`, `pdb`, temporary `print()` statements)

### Security
- [ ] No secrets in source files or committed `.env`
- [ ] `.env.example` is up to date with all required variables
- [ ] `pip audit` shows no known vulnerabilities

### Dependencies
- [ ] `requirements.txt` is pinned and up to date
- [ ] No unused imports or packages

### Documentation
- [ ] `README.md` reflects current setup steps
- [ ] `CHANGELOG.md` updated with changes since last release

## Additional Checks — `production`
- [ ] Docker image builds successfully: `docker build -t mychatbot .`
- [ ] Container runs as non-root user (check `Dockerfile` for `USER appuser`)
- [ ] Health check endpoint responds: `GET /health`
- [ ] Environment variables documented in deployment runbook
- [ ] Rollback plan confirmed

## Output
Report each item as ✅ PASS or ❌ FAIL with a brief explanation.
Block deployment if any `CRITICAL` or `HIGH` items fail.
