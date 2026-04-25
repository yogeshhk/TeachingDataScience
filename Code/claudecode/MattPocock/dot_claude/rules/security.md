---
paths:
  - "**/*"
---
# Security Rules (All Files)

These security rules apply to every file in the project.

## Secrets and Credentials
- **NEVER** hardcode API keys, tokens, passwords, or secrets in source files
- All secrets must come from environment variables via `src/mychatbot/config.py`
- Use `.env.example` to document required env vars (with placeholder values only)

## Input Validation
- Validate ALL external inputs with Pydantic before processing
- Sanitize user-provided strings before using in file paths, SQL, or shell commands
- Reject inputs that exceed reasonable length bounds

## Logging Safety
- Never log: API keys, tokens, passwords, session IDs, PII (names, emails, IPs)
- Use `***REDACTED***` or omit sensitive fields in log statements

## Docker / Deployment
- Containers must run as non-root user (add `USER appuser` in Dockerfile)
- Pin base image tags — no `FROM python:latest`
- Pin all dependency versions in `requirements.txt`

## Dependencies
- All Python packages must have pinned versions (`==` or bounded `~=`)
- No packages from unknown or unofficial sources
- Run `pip audit` before release

## File Access
- Never read or expose `.env`, `.env.*`, or credential files
- Validate file paths before open() to prevent path traversal
