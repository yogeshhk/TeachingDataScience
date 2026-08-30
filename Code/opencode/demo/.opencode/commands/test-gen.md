description: >
  Generate comprehensive pytest tests for a source file.
  Uses pytest-patterns skill. Pass the source file path
  as the argument.
agent: build
---
Generate comprehensive pytest tests for $ARGUMENTS.

Before writing tests:
1. Read the pytest-patterns skill for this project
2. Read the source file to understand all functions/classes
3. Read the existing test file (if any) to avoid duplicates

Write tests covering:
- Happy path for each public function
- Edge cases (empty input, None, boundary values)
- Error cases (missing env vars, invalid model names,
  API failures:  mock the LLM calls, no real API use)
- Use Faker to generate synthetic conversation data for
  multi-turn history tests

Output: complete updated test file, no placeholders.