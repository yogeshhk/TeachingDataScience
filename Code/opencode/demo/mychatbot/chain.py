"""Reusable LangChain module with Groq LLM integration."""

import os
from typing import Sequence

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from pydantic import SecretStr

load_dotenv()


DEFAULT_MODEL = "mixtral-8x7b-32768"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI assistant."


def get_chat_history(
    messages: Sequence[BaseMessage],
) -> list[dict[str, str]]:
    """Convert LangChain messages to list of dictionaries.

    Args:
        messages: Sequence of BaseMessage objects.

    Returns:
        List of message dictionaries with 'role' and 'content' keys.
    """
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
    return history


def create_chain(
    model: str | None = None,
    system_message: str | None = None,
    temperature: float = 0.7,
) -> tuple[ChatGroq, ChatPromptTemplate]:
    """Create a LangChain chat chain with Groq LLM.

    Args:
        model: Groq model name. Defaults to mixtral-8x7b-32768.
        system_message: System prompt message. Defaults to DEFAULT_SYSTEM_MESSAGE.
        temperature: LLM temperature. Defaults to 0.7.

    Returns:
        Tuple of (ChatGroq instance, ChatPromptTemplate).

    Raises:
        ValueError: If GROQ_API_KEY environment variable is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    chat_model = ChatGroq(
        model=model or DEFAULT_MODEL,
        api_key=SecretStr(api_key),
        temperature=temperature,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message or DEFAULT_SYSTEM_MESSAGE),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ]
    )

    return chat_model, prompt


def invoke_chain(
    input_text: str,
    chat_history: BaseChatMessageHistory | None = None,
    model: str | None = None,
    system_message: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Invoke the chat chain with user input.

    Args:
        input_text: User input message.
        chat_history: Optional chat history object.
        model: Groq model name.
        system_message: Optional system prompt.
        temperature: LLM temperature.

    Returns:
        AI response as string.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    chat_model, prompt = create_chain(
        model=model,
        system_message=system_message,
        temperature=temperature,
    )

    chain = prompt | chat_model

    history = chat_history.messages if chat_history else []

    response = chain.invoke(
        {
            "input": input_text,
            "history": history,
        }
    )

    if chat_history:
        chat_history.add_user_message(input_text)
        chat_history.add_ai_message(response.content)  # type: ignore[arg-type]

    return response.content  # type: ignore[return-value]
