name: pytest-patterns
description: >
  Best practices and patterns for writing pytest tests
  for this project. Use when generating or reviewing
  Python unit tests.
license: MIT
---
## Pytest Patterns for mychatbot

### Fixtures
- Use `@pytest.fixture` for shared setup
- `mock_chain` fixture: returns a MagicMock callable
  that returns "Mocked response"
- `sample_messages` fixture: list of user/assistant dicts

### Mocking
- Mock `mychatbot.chain.ChatGroq` to avoid real API calls
- Use `pytest.raises` for exception testing
- Use `monkeypatch` for env var testing

### Synthetic Data (Faker)
- Import `faker.Faker` for realistic fake data
- Generate fake user messages, conversation histories
- Use `@pytest.mark.parametrize` for data-driven tests

### Coverage Target: 85%+