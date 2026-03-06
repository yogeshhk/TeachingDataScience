# AGENTS.md - Agentic Coding Guidelines

This file provides guidance for agents working on the mychatbot project.

## Project Overview

- **Project**: mychatbot
- **Type**: Python chatbot with CLI and web interfaces
- **Stack**: Python 3.11, LangChain, langchain-groq, Streamlit, Rich, pytest
- **LLM**: Groq (mixtral-8x7b-32768 default model)

## Commands

### Installation
```bash
pip install -r requirements.txt
pip install -e .  # For editable install
pip install -e ".[dev]"  # With dev dependencies
```

### Running the Application

**CLI Chatbot:**
```bash
python cli.py
```

**Web Chatbot:**
```bash
streamlit run app.py
```

### Testing

**Run all tests:**
```bash
pytest
```

**Run a single test file:**
```bash
pytest tests/test_chain.py
```

**Run a single test:**
```bash
pytest tests/test_chain.py::TestGetChatHistory::test_empty_messages
pytest -k "test_create_chain"
```

**Run with verbose output:**
```bash
pytest -v
```

### Linting and Formatting

**Format with Black:**
```bash
black .
```

**Lint with Ruff:**
```bash
ruff check .
```

**Type checking with MyPy:**
```bash
mypy .
```

### Environment Setup

1. Copy `.env.example` to `.env`
2. Add your `GROQ_API_KEY` from https://console.groq.com/

## Code Style Guidelines

### General Conventions

- **Python version**: 3.11+ (use built-in types like `list[str]` not `typing.List[str]`)
- **Line length**: 88 characters (Black default)
- **End-of-file newline**: Yes
- **Trailing commas**: Use where appropriate for better diffs

### Imports

Order imports in each file as follows (with blank lines between groups):

1. Standard library (`import os`, `from typing import ...`)
2. Third-party packages (`from dotenv import ...`, `from langchain_...`)
3. Local project imports (`from chain import ...`)

Alphabetize within each group.

Example:
```python
import os
from typing import Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

from chain import create_chain
```

### Type Hints

- **Always use type hints** on function signatures (parameters and return types)
- Use built-in types: `list[str]`, `dict[str, int]`, `str | None`
- Use `None` instead of `Optional[...]` for simple cases
- Add `# type: ignore[...]` comments only when necessary (e.g., for known library typing issues)

### Naming Conventions

- **Variables/functions**: `snake_case` (e.g., `user_input`, `get_chat_history`)
- **Classes**: `CamelCase` (e.g., `ChatGroq`, `MyCustomClass`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_MODEL`, `API_KEY`)
- **Private functions/variables**: Prefix with underscore (e.g., `_internal_func`)

### Docstrings

Use Google-style docstrings on all public functions and classes.

```python
def function_name(param1: str, param2: int) -> bool:
    """Short one-line description.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When this happens.
    """
```

### Error Handling

- Use specific exception types, not bare `except:`
- Include meaningful error messages
- Let exceptions propagate appropriately for unexpected errors

```python
# Good
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Avoid
if not api_key:
    raise Exception("Missing key")
```

### Testing Conventions

- Test files go in `tests/` directory
- Name test files as `test_<module>.py`
- Use `pytest` with class-based test organization
- Use `unittest.mock.patch` for mocking external dependencies
- Group tests by function/class with `Test<ClassName>` or `Test<FunctionName>` classes

Example test structure:
```python
class TestGetChatHistory:
    """Tests for get_chat_history function."""

    def test_empty_messages(self):
        """Test with empty message list."""
        result = get_chat_history([])
        assert result == []
```

### Environment Variables

- Never commit actual API keys to version control
- Use `.env` files for local development (add to `.gitignore`)
- Provide `.env.example` as a template
- Use `python-dotenv` for loading: `from dotenv import load_dotenv`

### Async Code

- This project uses synchronous code; use `async/await` only for I/O-bound operations if needed
- When using async with LangChain, prefer running in executor if not natively async

### Code Patterns

**Creating the chain:**
```python
chat_model, prompt = create_chain()
chain = prompt | chat_model
response = chain.invoke({"input": "Hello", "history": []})
```

**Handling chat history:**
```python
from chain import get_chat_history

history = get_chat_history(messages)  # Convert to list of dicts
```

## File Structure

```
mychatbot/
├── chain.py           # Reusable LangChain module (export: create_chain, invoke_chain)
├── cli.py             # CLI entry point
├── app.py             # Streamlit web app
├── tests/
│   └── test_chain.py  # Unit tests
├── pyproject.toml     # Project config (PEP 621)
├── requirements.txt   # Pip dependencies
├── .env.example       # Env var template
└── README.md          # Documentation
```

## Common Tasks

### Adding a new LLM model

Modify `chain.py`:
1. Update `DEFAULT_MODEL` constant
2. Or pass `model="new-model"` to `create_chain()`

### Adding a new feature

1. Add to appropriate module (`chain.py`, `cli.py`, or new module)
2. Add tests in `tests/`
3. Update README if user-facing
4. Run linting: `ruff check . && black . && mypy .`
