## Project: mychatbot

## Tech Stack
- Python 3.11
- LangChain 0.3+
- langchain-groq
- Streamlit 1.35+
- Rich (CLI output)
- pytest + pytest-mock

## Build Commands
- Install: `pip install -e ".[dev]"`
- Tests: `pytest tests/ -v`

## Conventions
- Type hints on all functions
- Docstrings (Google style) on all public functions
- No hardcoded API keys:  always read from env vars

## File Map
- tests/            :  pytest test suite