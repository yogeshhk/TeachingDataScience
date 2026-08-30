# mychatbot Technical Specification

## Overview

**Project Name:** mychatbot  
**Project Type:** Conversational AI Chatbot Application  
**Core Functionality:** A chatbot application supporting both CLI and web-based interfaces, powered by LangChain and LLMs, with persistent conversation history and streaming responses.  
**Target Users:** Developers and end-users who want a customizable AI chatbot

---

## 1. CLI Chatbot Interface

### 1.1 Overview

The CLI interface provides a terminal-based chat experience for users who prefer command-line interaction.

### 1.2 Functional Requirements

- **Multi-turn Conversation**
  - The CLI must maintain context across multiple user inputs
  - Each user message is sent to the LLM along with conversation history
  - The assistant's responses are displayed after each user input

- **Conversation History**
  - Display chat history in the format: `User: <message>` and `Assistant: <response>`
  - History must be preserved within the session
  - Optional: Save history to a file for persistence between sessions

- **Exit Command**
  - Accept `/exit`, `/quit`, or `q` (case-insensitive) to terminate the session
  - Display a goodbye message before exiting
  - Clear any cached resources on exit

- **User Input**
  - Prompt users with a clear indicator (e.g., `> ` or `You: `)
  - Handle empty input gracefully (prompt again)
  - Support multi-line input if needed

### 1.3 Interface Example

```
Welcome to mychatbot CLI!
Type /quit, /exit, or q to exit.

User: What is Python?
Assistant: Python is a high-level, interpreted programming language known for its readability and versatility.

User: Tell me more about its features
Assistant: Python features include:
- Easy-to-read syntax
- Dynamic typing
- Extensive standard library
- Cross-platform compatibility
- Strong community support

User: /quit
Goodbye!
```

---

## 2. Streamlit Web Chatbot

### 2.1 Overview

The Streamlit web interface provides a browser-based chat experience with real-time streaming responses.

### 2.2 Functional Requirements

- **Session State Management**
  - Maintain `st.session_state` for conversation history
  - Store messages as a list of dictionaries: `[{"role": "user|assistant", "content": "..."}]`
  - Persist session state across Streamlit reruns
  - Provide a "Clear Chat" button to reset conversation

- **Streaming Responses**
  - Stream LLM tokens in real-time using LangChain's streaming callback
  - Display tokens as they arrive (no buffering until complete)
  - Show a typing indicator while waiting for first token

- **Chat UI Components**
  - `st.chat_message` for displaying user and assistant messages
  - `st.chat_input` for user text entry
  - Scrollable chat container that auto-scrolls to latest message

- **Initialization**
  - On first load, initialize empty conversation history
  - Optionally: Allow setting system prompt via sidebar or settings

### 2.3 Interface Example

```
+--------------------------------------------------+
|  mychatbot                              [Clear]  |
+--------------------------------------------------+
|                                                  |
|  User                                            |
|  What is LangChain?                              |
|                                                  |
|  Assistant                                        |
|  LangChain is a framework for building...        |
|                                                  |
|  +------------------------------------------+    |
|  | Type your message...              [Send]|    |
|  +------------------------------------------+    |
+--------------------------------------------------+
```

---

## 3. Shared LangChain Chain Module

### 3.1 Overview

The shared chain module contains the core logic for LLM interaction, including model configuration, prompt management, and memory.

### 3.2 Model Configuration

- **LLM Provider**
  - Support Groq as the primary LLM provider (via `langchain-groq`)
  - Model selection configurable via environment variable
  - Default model: `llama-3.1-70b-versatile` or similar

- **Configuration Parameters**
  - `GROQ_API_KEY`: Required API key for Groq
  - `MODEL_NAME`: LLM model identifier
  - `TEMPERATURE`: Sampling temperature (default: 0.7)
  - `MAX_TOKENS`: Maximum tokens in response (default: 1024)

### 3.3 System Prompt Injection

- **System Prompt**
  - Define a default system prompt explaining the assistant's persona
  - Allow customization via environment variable `SYSTEM_PROMPT`
  - Inject system prompt as the first message in the conversation chain

- **Default System Prompt**
  ```
  You are a helpful, friendly AI assistant. Answer questions
  clearly and concisely. If you don't know something, admit
  it honestly.
  ```

### 3.4 Memory Management

- **Conversation Memory**
  - Use LangChain's `ConversationBufferMemory` for storing chat history
  - Memory stores both human and AI messages
  - Configurable memory window (number of messages to retain)

- **Message Format**
  - Human messages labeled as `human`
  - AI messages labeled as `ai`
  - System messages labeled as `system`

- **Chain Construction**
  - Use `ConversationalChain` from LangChain
  - Structure: `PromptTemplate + LLM + Memory`
  - Input: `chat_history` + `input`

### 3.5 Code Structure

```python
from langchain_groq import ChatGroq
from langchain.chains import ConversationalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def create_chain(
    model_name: str = None,
    temperature: float = 0.7,
    system_prompt: str = None
) -> ConversationalChain:
    """Create a configured LangChain conversational chain."""
    # ... implementation
```

---

## 4. Environment Variable Requirements

### 4.1 Required Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | API key for Groq LLM service | Yes |

### 4.2 Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | LLM model to use | `llama-3.1-70b-versatile` |
| `TEMPERATURE` | Sampling temperature | `0.7` |
| `MAX_TOKENS` | Max response tokens | `1024` |
| `SYSTEM_PROMPT` | Custom system prompt | (Built-in default) |
| `MEMORY_WINDOW` | Number of messages to retain | `10` |

### 4.3 Loading Configuration

- Use `python-dotenv` to load `.env` files
- Provide clear error messages when required variables are missing
- Validate API key format before making requests

---

## 5. Error Handling Expectations

### 5.1 General Principles

- Never expose sensitive information (API keys, internal errors) to users
- Provide user-friendly error messages
- Log detailed errors for debugging
- Graceful degradation where possible

### 5.2 Specific Error Handling

| Error Type | User Message | Logging |
|------------|--------------|---------|
| Missing API Key | "Configuration error: GROQ_API_KEY not set" | Full error details |
| Invalid API Key | "Authentication failed. Please check your API key." | API response details |
| Rate Limiting | "Service busy. Please wait and try again." | Retry-after value |
| Network Error | "Connection failed. Please check your internet." | Exception stack trace |
| LLM Error | "Sorry, I encountered an error processing your request." | Error response from LLM |
| Empty Input | Re-prompt without error message | No logging needed |

### 5.3 Retry Strategy

- Implement exponential backoff for transient errors
- Maximum 3 retries before showing error to user
- Log each retry attempt

### 5.4 Recovery

- CLI: Allow user to continue after error
- Streamlit: Display error in chat, allow retry
- Clear invalid state on recovery

---

## 6. Non-Functional Requirements

### 6.1 Performance

- First response token within 2 seconds (network dependent)
- Streaming latency under 500ms per token
- Memory usage under 500MB for typical conversations

### 6.2 Security

- Never log API keys
- Validate all user input
- No hardcoded credentials

### 6.3 Reliability

- Handle network timeouts gracefully
- Recover from LLM errors without crashing
- Persist conversation history across restarts (optional enhancement)

---

## 7. Data Models

### 7.1 Message Format

```python
from typing import TypedDict

class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str
```

### 7.2 Configuration Model

```python
from typing import Optional
from pydantic import BaseModel, Field

class ChatConfig(BaseModel):
    groq_api_key: str = Field(..., min_length=1)
    model_name: str = "llama-3.1-70b-versatile"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    system_prompt: Optional[str] = None
    memory_window: int = Field(10, ge=1)
```

---

## 8. Open Questions

1. **Persistence:** Should conversation history be persisted to disk between sessions?
2. **Multi-user:** Should the web app support multiple users with separate histories?
3. **Model selection:** Should users be able to switch models at runtime?
4. **Streaming:** Should streaming be optional (toggle on/off)?
5. **File uploads:** Should the chatbot support document Q&A with file uploads?

---

## 9. Acceptance Criteria

### CLI Interface
- [ ] User can start CLI and see welcome message
- [ ] User can type messages and receive responses
- [ ] Conversation history is maintained within session
- [ ] `/exit`, `/quit`, and `q` commands terminate the session
- [ ] Errors display user-friendly messages

### Streamlit Interface
- [ ] Web page loads without errors
- [ ] User can send messages via chat input
- [ ] Responses stream in real-time
- [ ] Chat history persists during session
- [ ] Clear Chat button resets conversation

### LangChain Module
- [ ] Chain can be created with default configuration
- [ ] Custom system prompt can be injected
- [ ] Memory retains conversation context
- [ ] Configuration loads from environment variables

### Error Handling
- [ ] Missing API key shows clear error
- [ ] Network errors handled gracefully
- [ ] No sensitive data exposed in error messages
