# /project:fix — Root Cause Error Fix

Traces an error to its root cause and applies a minimal, targeted fix.

## Usage
```
/project:fix <error message or traceback>
```

Paste the full error or traceback after the command.

## Behavior

1. **Parse the error:** Identify the exception type, message, and the failing file + line number.
2. **Trace the execution path:** Search the codebase to map the call chain from the entry point (route handler / CLI entrypoint) down through middleware and into the module where the error originates.
3. **Identify root cause:** Distinguish between the error site (where the exception is raised) and the root cause (where the incorrect logic or missing guard lives).
4. **Propose a minimal fix:**
   - Change only the code that is directly responsible for the failure.
   - Do NOT refactor unrelated code.
   - Do NOT suggest style improvements in the same diff.
   - Do NOT touch files outside the call chain unless strictly necessary.
5. **Show the diff** and explain the fix in 2–3 sentences before applying.
6. **After applying:** Suggest the specific `pytest` command to verify the fix.

## Example
```
/project:fix
Traceback (most recent call last):
  File "src/mychatbot/cli.py", line 42, in chat_loop
    response = chain.invoke({"input": user_msg})
KeyError: 'input'
```

Expected output: traces `chain.invoke` to `chain.py`, identifies the key mismatch, applies a one-line fix, shows diff, suggests `pytest tests/test_chain.py -v`.
