---
name: security-review
description: >
  Auto-triggered security checklist. Activates when editing API
  endpoint files, configuration files, Dockerfiles, or dependency
  manifests. Runs a structured security review automatically —
  no slash command needed.
triggers:
  - "**/*router*.py"
  - "**/*endpoint*.py"
  - "**/*api*.py"
  - "**/*config*.py"
  - "**/Dockerfile*"
  - "**/*.env.example"
  - "**/requirements*.txt"
  - "**/pyproject.toml"
---

# Security Review Checklist

This skill runs automatically when the above file patterns are modified.

## Checklist (run through each item)

### Secrets and Credentials
- [ ] No hardcoded API keys, tokens, passwords, or secrets in source
- [ ] All secrets loaded from environment variables via `config.py`
- [ ] `.env.example` uses placeholder values only (e.g., `GROQ_API_KEY=your_key_here`)

### Input Validation
- [ ] All external inputs (HTTP body, query params, CLI args) validated with Pydantic
- [ ] No direct use of raw user input in file paths, shell commands, or SQL
- [ ] String inputs have reasonable length bounds enforced

### Logging Safety
- [ ] No API keys, tokens, passwords, session IDs, or PII in log statements
- [ ] Sensitive fields redacted or omitted in error messages returned to users

### Docker / Deployment (if Dockerfile modified)
- [ ] Container runs as non-root user (`USER appuser` present)
- [ ] Base image tag is pinned (not `python:latest`)
- [ ] No secrets passed as `ENV` in Dockerfile (use runtime injection)

### Dependencies (if requirements.txt or pyproject.toml modified)
- [ ] All packages pinned to specific versions
- [ ] No packages from unverified sources
- [ ] Run `pip audit` to check for known CVEs

## Reporting
For each item that FAILS, report:
```
[SECURITY] path/to/file.py:line — Description of issue
Fix: Specific remediation step
```

If all items PASS, report: "✅ Security checklist passed — no issues found."
