description: >
Run the full test suite and summarize results
agent: build
---
Run the full test suite for this project:

1. Execute: `pytest tests/ -v --tb=short --cov=src/mychatbot
   --cov-report=term-missing 2>&1`
2. Parse the output and give me:
   - Pass/fail count
   - Coverage percentage per module
   - List of any failing tests with the error message
   - Three specific suggestions to improve coverage
     for any module below 85%