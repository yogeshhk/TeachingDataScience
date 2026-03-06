# MyChatbot

A Python chatbot project with CLI and web interfaces using LangChain and Groq LLM.

## Features

- **CLI Chatbot**: Terminal-based chatbot with Rich formatting
- **Web Chatbot**: Streamlit web interface
- **Reusable Chain**: Modular LangChain integration

## Requirements

- Python 3.11+
- Groq API key

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mychatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Get Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Create an account or sign in
3. Navigate to API Keys
4. Create a new API key
5. Add it to your `.env` file

## Usage

### CLI Chatbot

```bash
python cli.py
```

### Web Chatbot

```bash
streamlit run app.py
```

The web app will open at http://localhost:8501

## Testing

```bash
# Run all tests
pytest

# Run a single test
pytest tests/test_chain.py::TestGetChatHistory::test_empty_messages
pytest -k "test_create_chain"
```

## Project Structure

```
mychatbot/
├── chain.py          # Reusable LangChain module
├── cli.py            # CLI chatbot
├── app.py            # Streamlit web app
├── requirements.txt  # Dependencies
├── pyproject.toml    # Project configuration
├── .env.example      # Environment template
├── tests/
│   └── test_chain.py # Unit tests
└── README.md
```

## License

MIT
