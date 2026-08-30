"""LangChain conversational chain for mychatbot."""

import logging
import os
from typing import Callable, List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

from mychatbot.types import ChatConfig, Message

logger = logging.getLogger(__name__)

MAX_CHARS = 12000


class ChatChainError(Exception):
    """Base exception for ChatChain errors."""

    user_message: str = "Sorry, I encountered an error processing your request."


class AuthenticationError(ChatChainError):
    """Raised when authentication fails."""

    user_message = "Authentication failed. Please check your API key."


class RateLimitError(ChatChainError):
    """Raised when rate limited."""

    user_message = "Service busy. Please wait and try again."


class NetworkError(ChatChainError):
    """Raised when network fails."""

    user_message = "Connection failed. Please check your internet."


class RequestSizeError(ChatChainError):
    """Raised when request is too large."""

    user_message = "Your message is too long. Please shorten it and try again."


def _get_api_key() -> str:
    """Get GROQ_API_KEY from environment.

    Returns:
        The API key string.

    Raises:
        ValueError: If API key is not set.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it to your Groq API key."
        )
    return api_key


def _convert_messages_to_langchain(
    messages: list[dict[str, str]],
) -> list[BaseMessage]:
    """Convert message dicts to LangChain message objects.

    Args:
        messages: List of dicts with 'role' and 'content' keys.

    Returns:
        List of LangChain BaseMessage objects.
    """
    result: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "assistant":
            result.append(AIMessage(content=content))
        elif role == "system":
            from langchain_core.messages import SystemMessage

            result.append(SystemMessage(content=content))
        else:
            result.append(HumanMessage(content=content))

    return result


def _truncate_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Truncate messages if total character count exceeds MAX_CHARS.

    Preserves system messages. Removes oldest non-system messages first.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        Truncated list of messages.
    """
    total_chars = sum(len(msg.get("content", "")) for msg in messages)

    if total_chars <= MAX_CHARS:
        return messages

    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    other_messages = [msg for msg in messages if msg.get("role") != "system"]

    while (
        sum(len(msg.get("content", "")) for msg in other_messages)
        + sum(len(msg.get("content", "")) for msg in system_messages)
        > MAX_CHARS
        and other_messages
    ):
        other_messages.pop(0)

    logger.warning(
        f"Truncated conversation history from {total_chars} to "
        f"{sum(len(msg.get('', '')) for msg in system_messages + other_messages)} characters"
    )

    return system_messages + other_messages


def make_chain(
    model_name: str,
    system_prompt: str = "You are a helpful assistant",
    temperature: float = 0.7,
) -> Callable[[list[dict[str, str]]], str]:
    """Create a LangChain conversational chain.

    Creates a callable that takes a list of message dictionaries
    (with 'role' and 'content' keys) and returns the assistant's response.

    Args:
        model_name: The name of the Groq model to use.
        system_prompt: The system prompt to use for the assistant.
        temperature: The sampling temperature (0.0 to 2.0).

    Returns:
        A callable that takes list[dict] and returns str.

    Raises:
        ValueError: If GROQ_API_KEY is not set in environment.

    Example:
        >>> chain = make_chain("llama-3.1-70b-versatile")
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = chain(messages)
    """
    api_key = _get_api_key()

    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )

    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message = HumanMessagePromptTemplate.from_template("{input}")

    prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    chain = prompt | llm

    def invoke(messages: list[dict[str, str]]) -> str:
        """Invoke the chain with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                Example: [{"role": "user", "content": "Hello"},
                         {"role": "assistant", "content": "Hi there!"}]

        Returns:
            The assistant's response as a string.

        Raises:
            ChatChainError: If the API call fails.
        """
        try:
            messages = _truncate_messages(messages)

            chat_history = _convert_messages_to_langchain(messages[:-1])
            latest_message = (
                messages[-1] if messages else {"role": "user", "content": ""}
            )

            if not messages:
                raise ValueError("At least one message is required")

            input_text = latest_message.get("content", "")

            result = chain.invoke(
                {
                    "chat_history": chat_history,
                    "input": input_text,
                }
            )
            return str(result.content)  # type: ignore[return-value]
        except ValueError as e:
            if "GROQ_API_KEY" in str(e):
                raise
            logger.exception("ChatChain invoke failed")
            raise ChatChainError(str(e)) from e
        except Exception as e:
            logger.exception("ChatChain invoke failed")
            error_str = str(e).lower()

            if "api key" in error_str or "auth" in error_str:
                raise AuthenticationError(str(e)) from e

            if "rate" in error_str or "429" in error_str:
                raise RateLimitError(str(e)) from e

            if "request" in error_str and "large" in error_str:
                raise RequestSizeError(str(e)) from e

            if (
                "connection" in error_str
                or "timeout" in error_str
                or "network" in error_str
            ):
                raise NetworkError(str(e)) from e

            raise ChatChainError(str(e)) from e

    return invoke


class ChatChain:
    """Manages LangChain conversational chain with memory."""

    def __init__(self, config: ChatConfig) -> None:
        """Initialize the chat chain with configuration.

        Args:
            config: ChatConfig with LLM and chain settings.
        """
        self.config = config
        self._session_id = "default"

        self._llm = ChatGroq(
            model=config.model_name,
            api_key=config.groq_api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            streaming=True,
        )

        self._chat_history = InMemoryChatMessageHistory()

        system_prompt = config.system_prompt or "You are a helpful assistant."

        messages: list = [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)

        self._chain = prompt | self._llm

        self._chain_with_history = RunnableWithMessageHistory(
            self._chain,
            lambda _: self._chat_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def predict(self, user_input: str) -> str:
        """Send a message and get a blocking response.

        Args:
            user_input: The user's message.

        Returns:
            The assistant's response.

        Raises:
            ChatChainError: If the chain fails.
        """
        try:
            self._truncate_history()

            config = {"configurable": {"session_id": self._session_id}}
            result = self._chain_with_history.invoke(
                {"input": user_input},
                config=config,
            )
            return result.content
        except Exception as e:
            logger.exception("ChatChain predict failed")
            raise self._map_error(e)

    def predict_streaming(
        self, user_input: str, token_callback: Callable[[str], None]
    ) -> None:
        """Send a message and stream the response via callback.

        Args:
            user_input: The user's message.
            token_callback: Callback function called with each token.

        Raises:
            ChatChainError: If the chain fails.
        """
        try:
            self._truncate_history()

            config = {"configurable": {"session_id": self._session_id}}
            for chunk in self._chain_with_history.stream(
                {"input": user_input},
                config=config,
            ):
                if chunk.content:
                    token_callback(chunk.content)
        except Exception as e:
            logger.exception("ChatChain predict_streaming failed")
            raise self._map_error(e)

    def clear_memory(self) -> None:
        """Clear conversation history."""
        self._chat_history.clear()

    def get_history(self) -> List[Message]:
        """Get current conversation history.

        Returns:
            List of Message dicts with role and content.
        """
        messages = self._chat_history.messages
        history: List[Message] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "system"
            history.append({"role": role, "content": msg.content})
        return history

    def _map_error(self, error: Exception) -> ChatChainError:
        """Map an exception to a user-friendly ChatChainError.

        Args:
            error: The original exception.

        Returns:
            A ChatChainError with appropriate user message.
        """
        error_str = str(error).lower()

        if "api key" in error_str or "auth" in error_str:
            return AuthenticationError(str(error))

        if "rate" in error_str or "429" in error_str:
            return RateLimitError(str(error))

        if "request" in error_str and "large" in error_str:
            return RequestSizeError(str(error))

        if (
            "connection" in error_str
            or "timeout" in error_str
            or "network" in error_str
        ):
            return NetworkError(str(error))

        return ChatChainError(str(error))

    def _truncate_history(self) -> None:
        """Truncate chat history if total character count exceeds MAX_CHARS.

        Removes oldest messages first.
        """
        try:
            messages = self._chat_history.messages
            total_chars = sum(len(msg.content) for msg in messages)
        except (TypeError, AttributeError):
            return

        if total_chars <= MAX_CHARS:
            return

        while total_chars > MAX_CHARS and messages:
            removed = messages.pop(0)
            total_chars -= len(removed.content)

        logger.warning(f"Truncated conversation history to {total_chars} characters")
