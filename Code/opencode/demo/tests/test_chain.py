"""Tests for mychatbot.chain module."""

from unittest.mock import Mock, patch

import pytest

from mychatbot.chain import (
    AuthenticationError,
    ChatChain,
    ChatChainError,
    NetworkError,
    RateLimitError,
    RequestSizeError,
    _convert_messages_to_langchain,
    _get_api_key,
    make_chain,
)
from mychatbot.types import ChatConfig, Message


class TestGetApiKey:
    """Tests for _get_api_key function."""

    def test_returns_api_key_when_set(self) -> None:
        """Test that API key is returned when GROQ_API_KEY is set."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key-123"}):
            result = _get_api_key()
            assert result == "test-key-123"

    def test_raises_value_error_when_not_set(self) -> None:
        """Test that ValueError is raised when GROQ_API_KEY is not set."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                _get_api_key()

    def test_raises_value_error_when_empty(self) -> None:
        """Test that ValueError is raised when GROQ_API_KEY is empty string."""
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                _get_api_key()


class TestConvertMessagesToLangchain:
    """Tests for _convert_messages_to_langchain function."""

    def test_converts_human_message(self) -> None:
        """Test conversion of human message."""
        messages = [{"role": "user", "content": "Hello"}]
        result = _convert_messages_to_langchain(messages)
        assert len(result) == 1
        from langchain_core.messages import HumanMessage

        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    def test_converts_ai_message(self) -> None:
        """Test conversion of assistant message."""
        messages = [{"role": "assistant", "content": "Hi there"}]
        result = _convert_messages_to_langchain(messages)
        assert len(result) == 1
        from langchain_core.messages import AIMessage

        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Hi there"

    def test_converts_system_message(self) -> None:
        """Test conversion of system message."""
        messages = [{"role": "system", "content": "You are helpful"}]
        result = _convert_messages_to_langchain(messages)
        assert len(result) == 1
        from langchain_core.messages import SystemMessage

        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are helpful"

    def test_empty_list_returns_empty(self) -> None:
        """Test that empty list returns empty list."""
        result = _convert_messages_to_langchain([])
        assert result == []

    def test_default_role_is_user(self) -> None:
        """Test that missing role defaults to user."""
        messages = [{"content": "Test"}]
        result = _convert_messages_to_langchain(messages)
        assert len(result) == 1
        from langchain_core.messages import HumanMessage

        assert isinstance(result[0], HumanMessage)

    def test_default_content_is_empty(self) -> None:
        """Test that missing content defaults to empty string."""
        messages = [{"role": "user"}]
        result = _convert_messages_to_langchain(messages)
        assert result[0].content == ""

    def test_converts_multiple_messages(self) -> None:
        """Test conversion of multiple messages in order."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _convert_messages_to_langchain(messages)
        assert len(result) == 4
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)
        assert isinstance(result[3], HumanMessage)

    def test_unknown_role_defaults_to_user(self) -> None:
        """Test that unknown role defaults to user."""
        messages = [{"role": "unknown_role", "content": "Test"}]
        result = _convert_messages_to_langchain(messages)
        from langchain_core.messages import HumanMessage

        assert isinstance(result[0], HumanMessage)


@pytest.fixture
def mock_config() -> ChatConfig:
    """Create a mock ChatConfig."""
    return ChatConfig(
        groq_api_key="test_api_key",
        model_name="llama-3.1-70b-versatile",
        temperature=0.7,
        max_tokens=1024,
        system_prompt="You are a helpful assistant.",
        memory_window=10,
    )


@pytest.fixture
def chat_chain(mock_config: ChatConfig) -> ChatChain:
    """Create a ChatChain with mocked dependencies."""
    with (
        patch("mychatbot.chain.ChatGroq") as mock_groq,
        patch("mychatbot.chain.InMemoryChatMessageHistory") as mock_history_cls,
    ):
        mock_groq_instance = Mock()
        mock_groq.return_value = mock_groq_instance
        mock_history_instance = Mock()
        mock_history_cls.return_value = mock_history_instance

        chain = ChatChain(mock_config)
        chain._llm = mock_groq_instance
        chain._chat_history = mock_history_instance
        chain._chain_with_history = Mock()
        return chain


class TestMakeChain:
    """Tests for make_chain function."""

    def test_make_chain_returns_callable(self) -> None:
        """Test that make_chain returns a callable."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq_instance = Mock()
            mock_groq.return_value = mock_groq_instance

            chain = make_chain("llama-3.1-70b-versatile")
            assert callable(chain)

    def test_make_chain_with_custom_system_prompt(self) -> None:
        """Test that custom system prompt is used."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            chain = make_chain("llama-3.1-70b-versatile", system_prompt="Custom prompt")
            assert callable(chain)

    def test_make_chain_with_custom_temperature(self) -> None:
        """Test that custom temperature is passed to ChatGroq."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            make_chain("llama-3.1-70b-versatile", temperature=0.5)
            mock_groq.assert_called_once()
            call_kwargs = mock_groq.call_args.kwargs
            assert call_kwargs["temperature"] == 0.5

    def test_make_chain_raises_when_no_api_key(self) -> None:
        """Test that ValueError is raised when API key not set."""
        with patch("mychatbot.chain._get_api_key") as mock_get_key:
            mock_get_key.side_effect = ValueError("GROQ_API_KEY not set")

            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                make_chain("llama-3.1-70b-versatile")

    def test_make_chain_invokes_correctly(self) -> None:
        """Test that chain invocation works with messages."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.ChatPromptTemplate") as mock_prompt,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            mock_response = Mock()
            mock_response.content = "Test response"

            mock_prompt_instance = Mock()
            mock_prompt_instance.__or__ = Mock(
                return_value=Mock(invoke=Mock(return_value=mock_response))
            )
            mock_prompt.from_messages.return_value = mock_prompt_instance

            chain = make_chain("llama-3.1-70b-versatile")
            messages = [{"role": "user", "content": "Hello"}]
            result = chain(messages)
            assert result == "Test response"

    def test_make_chain_with_history(self) -> None:
        """Test that chain uses conversation history."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.ChatPromptTemplate") as mock_prompt,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            mock_response = Mock()
            mock_response.content = "Response with history"

            mock_prompt_instance = Mock()
            mock_prompt_instance.__or__ = Mock(
                return_value=Mock(invoke=Mock(return_value=mock_response))
            )
            mock_prompt.from_messages.return_value = mock_prompt_instance

            chain = make_chain("llama-3.1-70b-versatile")
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ]
            result = chain(messages)
            assert result == "Response with history"

    def test_make_chain_empty_messages_raises(self) -> None:
        """Test that empty message list raises ChatChainError."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            chain = make_chain("llama-3.1-70b-versatile")

            with pytest.raises(ChatChainError, match="At least one message"):
                chain([])

    def test_make_chain_raises_auth_error(self) -> None:
        """Test that auth errors are raised as AuthenticationError."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.ChatPromptTemplate") as mock_prompt,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            mock_prompt_instance = Mock()
            mock_chain = Mock(invoke=Mock(side_effect=Exception("API key invalid")))
            mock_prompt_instance.__or__ = Mock(return_value=mock_chain)
            mock_prompt.from_messages.return_value = mock_prompt_instance

            chain = make_chain("llama-3.1-70b-versatile")
            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(AuthenticationError):
                chain(messages)

    def test_make_chain_raises_rate_limit_error(self) -> None:
        """Test that rate limit errors are raised as RateLimitError."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.ChatPromptTemplate") as mock_prompt,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            mock_prompt_instance = Mock()
            mock_chain = Mock(
                invoke=Mock(side_effect=Exception("Rate limit exceeded 429"))
            )
            mock_prompt_instance.__or__ = Mock(return_value=mock_chain)
            mock_prompt.from_messages.return_value = mock_prompt_instance

            chain = make_chain("llama-3.1-70b-versatile")
            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(RateLimitError):
                chain(messages)

    def test_make_chain_raises_network_error(self) -> None:
        """Test that network errors are raised as NetworkError."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.ChatPromptTemplate") as mock_prompt,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            mock_prompt_instance = Mock()
            mock_chain = Mock(invoke=Mock(side_effect=Exception("Connection timeout")))
            mock_prompt_instance.__or__ = Mock(return_value=mock_chain)
            mock_prompt.from_messages.return_value = mock_prompt_instance

            chain = make_chain("llama-3.1-70b-versatile")
            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(NetworkError):
                chain(messages)

    def test_make_chain_raises_generic_error(self) -> None:
        """Test that generic errors are raised as ChatChainError."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.ChatPromptTemplate") as mock_prompt,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            mock_prompt_instance = Mock()
            mock_chain = Mock(invoke=Mock(side_effect=Exception("Some random error")))
            mock_prompt_instance.__or__ = Mock(return_value=mock_chain)
            mock_prompt.from_messages.return_value = mock_prompt_instance

            chain = make_chain("llama-3.1-70b-versatile")
            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(ChatChainError):
                chain(messages)

    def test_make_chain_raises_network_error(self) -> None:
        """Test that network errors are raised as NetworkError."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.ChatPromptTemplate") as mock_prompt,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            mock_prompt_instance = Mock()
            mock_chain = Mock(invoke=Mock(side_effect=Exception("Connection timeout")))
            mock_prompt_instance.__or__ = Mock(return_value=mock_chain)
            mock_prompt.from_messages.return_value = mock_prompt_instance

            chain = make_chain("llama-3.1-70b-versatile")
            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(NetworkError):
                chain(messages)

    def test_make_chain_raises_generic_error(self) -> None:
        """Test that generic errors are raised as ChatChainError."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.ChatPromptTemplate") as mock_prompt,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            mock_prompt_instance = Mock()
            mock_chain = Mock(invoke=Mock(side_effect=Exception("Some random error")))
            mock_prompt_instance.__or__ = Mock(return_value=mock_chain)
            mock_prompt.from_messages.return_value = mock_prompt_instance

            chain = make_chain("llama-3.1-70b-versatile")
            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(ChatChainError):
                chain(messages)

    def test_make_chain_raises_request_size_error(self) -> None:
        """Test that request size errors are raised as RequestSizeError."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.ChatPromptTemplate") as mock_prompt,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            mock_prompt_instance = Mock()
            mock_chain = Mock(
                invoke=Mock(side_effect=Exception("Request too large for model"))
            )
            mock_prompt_instance.__or__ = Mock(return_value=mock_chain)
            mock_prompt.from_messages.return_value = mock_prompt_instance

            chain = make_chain("llama-3.1-70b-versatile")
            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(RequestSizeError):
                chain(messages)


class TestChatChain:
    """Tests for ChatChain class."""

    def test_predict_returns_response(self, chat_chain: ChatChain) -> None:
        """Test that predict returns the assistant's response."""
        mock_response = Mock()
        mock_response.content = "Test response"
        chat_chain._chain_with_history.invoke.return_value = mock_response

        result = chat_chain.predict("Hello")
        assert result == "Test response"

    def test_predict_raises_on_error(self, chat_chain: ChatChain) -> None:
        """Test that predict raises ChatChainError on failure."""
        chat_chain._chain_with_history.invoke.side_effect = Exception("API error")

        with pytest.raises(ChatChainError):
            chat_chain.predict("Hello")

    def test_predict_streaming_calls_callback(self, chat_chain: ChatChain) -> None:
        """Test that predict_streaming calls callback with tokens."""
        mock_chunk1 = Mock()
        mock_chunk1.content = "Hello"
        mock_chunk2 = Mock()
        mock_chunk2.content = " world"

        chat_chain._chain_with_history.stream.return_value = iter(
            [mock_chunk1, mock_chunk2]
        )

        tokens: list[str] = []

        def callback(token: str) -> None:
            tokens.append(token)

        chat_chain.predict_streaming("Hello", callback)
        assert len(tokens) == 2
        assert "Hello" in tokens

    def test_predict_streaming_empty_content_ignored(
        self, chat_chain: ChatChain
    ) -> None:
        """Test that empty content chunks are ignored in streaming."""
        mock_chunk1 = Mock()
        mock_chunk1.content = ""
        mock_chunk2 = Mock()
        mock_chunk2.content = "Hello"

        chat_chain._chain_with_history.stream.return_value = iter(
            [mock_chunk1, mock_chunk2]
        )

        tokens: list[str] = []

        def callback(token: str) -> None:
            tokens.append(token)

        chat_chain.predict_streaming("Hello", callback)
        assert len(tokens) == 1
        assert tokens[0] == "Hello"

    def test_predict_streaming_raises_on_error(self, chat_chain: ChatChain) -> None:
        """Test that streaming raises ChatChainError on failure."""
        chat_chain._chain_with_history.stream.side_effect = Exception("Stream error")

        with pytest.raises(ChatChainError):
            chat_chain.predict_streaming("Hello", lambda x: None)

    def test_clear_memory(self, chat_chain: ChatChain) -> None:
        """Test that clear_memory clears the conversation history."""
        chat_chain.clear_memory()
        assert chat_chain._chat_history.clear.called

    def test_get_history_returns_list(self, chat_chain: ChatChain) -> None:
        """Test that get_history returns conversation history."""
        mock_msg = Mock()
        mock_msg.type = "ai"
        mock_msg.content = "Hello"
        chat_chain._chat_history.messages = [mock_msg]

        history = chat_chain.get_history()
        assert isinstance(history, list)

    def test_get_history_human_message(self, chat_chain: ChatChain) -> None:
        """Test that get_history returns user role for HumanMessage."""
        from langchain_core.messages import HumanMessage

        chat_chain._chat_history.messages = [HumanMessage(content="User input")]

        history = chat_chain.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "User input"

    def test_get_history_ai_message(self, chat_chain: ChatChain) -> None:
        """Test that get_history returns assistant role for AIMessage."""
        from langchain_core.messages import AIMessage

        chat_chain._chat_history.messages = [AIMessage(content="AI response")]

        history = chat_chain.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "assistant"
        assert history[0]["content"] == "AI response"

    def test_get_history_system_message(self, chat_chain: ChatChain) -> None:
        """Test that get_history returns system role for SystemMessage."""
        from langchain_core.messages import SystemMessage

        chat_chain._chat_history.messages = [SystemMessage(content="System prompt")]

        history = chat_chain.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "system"

    def test_get_history_empty_returns_empty_list(self, chat_chain: ChatChain) -> None:
        """Test that empty history returns empty list."""
        chat_chain._chat_history.messages = []
        history = chat_chain.get_history()
        assert history == []

    def test_get_history_multiple_messages(self, chat_chain: ChatChain) -> None:
        """Test get_history with multiple messages."""
        from langchain_core.messages import HumanMessage, AIMessage

        chat_chain._chat_history.messages = [
            HumanMessage(content="First message"),
            AIMessage(content="First response"),
            HumanMessage(content="Second message"),
        ]

        history = chat_chain.get_history()
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"

    def test_authentication_error_mapping(self, mock_config: ChatConfig) -> None:
        """Test that auth errors are mapped correctly."""
        with (
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.InMemoryChatMessageHistory") as mock_history_cls,
        ):
            mock_groq.return_value = Mock()
            mock_history_cls.return_value = Mock()

            chain = ChatChain(mock_config)
            chain._chain_with_history = Mock()
            chain._chain_with_history.invoke.side_effect = Exception("API key invalid")

            with pytest.raises(AuthenticationError):
                chain.predict("test")

    def test_rate_limit_error_mapping(self, mock_config: ChatConfig) -> None:
        """Test that rate limit errors are mapped correctly."""
        with (
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.InMemoryChatMessageHistory") as mock_history_cls,
        ):
            mock_groq.return_value = Mock()
            mock_history_cls.return_value = Mock()

            chain = ChatChain(mock_config)
            chain._chain_with_history = Mock()
            chain._chain_with_history.invoke.side_effect = Exception(
                "Rate limit exceeded 429"
            )

            with pytest.raises(RateLimitError):
                chain.predict("test")

    def test_network_error_mapping(self, mock_config: ChatConfig) -> None:
        """Test that network errors are mapped correctly."""
        with (
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.InMemoryChatMessageHistory") as mock_history_cls,
        ):
            mock_groq.return_value = Mock()
            mock_history_cls.return_value = Mock()

            chain = ChatChain(mock_config)
            chain._chain_with_history = Mock()
            chain._chain_with_history.invoke.side_effect = Exception(
                "Connection timeout"
            )

            with pytest.raises(NetworkError):
                chain.predict("test")

    def test_network_error_mapping_timeout(self, mock_config: ChatConfig) -> None:
        """Test that timeout errors are mapped to NetworkError."""
        with (
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.InMemoryChatMessageHistory") as mock_history_cls,
        ):
            mock_groq.return_value = Mock()
            mock_history_cls.return_value = Mock()

            chain = ChatChain(mock_config)
            chain._chain_with_history = Mock()
            chain._chain_with_history.invoke.side_effect = Exception("Request timeout")

            with pytest.raises(NetworkError):
                chain.predict("test")

    def test_network_error_mapping_network(self, mock_config: ChatConfig) -> None:
        """Test that network errors are mapped correctly."""
        with (
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.InMemoryChatMessageHistory") as mock_history_cls,
        ):
            mock_groq.return_value = Mock()
            mock_history_cls.return_value = Mock()

            chain = ChatChain(mock_config)
            chain._chain_with_history = Mock()
            chain._chain_with_history.invoke.side_effect = Exception("Network error")

            with pytest.raises(NetworkError):
                chain.predict("test")

    def test_request_size_error_mapping(self, mock_config: ChatConfig) -> None:
        """Test that request size errors are mapped correctly."""
        with (
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.InMemoryChatMessageHistory") as mock_history_cls,
        ):
            mock_groq.return_value = Mock()
            mock_history_cls.return_value = Mock()

            chain = ChatChain(mock_config)
            chain._chain_with_history = Mock()
            chain._chain_with_history.invoke.side_effect = Exception(
                "Request too large for model"
            )

            with pytest.raises(RequestSizeError):
                chain.predict("test")


class TestMultiTurnConversation:
    """Tests for multi-turn conversation scenarios."""

    def test_chat_chain_multi_turn_conversation(self, chat_chain: ChatChain) -> None:
        """Test multi-turn conversation preserves history."""
        from langchain_core.messages import HumanMessage, AIMessage

        mock_response1 = Mock()
        mock_response1.content = "Response 1"
        mock_response2 = Mock()
        mock_response2.content = "Response 2"

        chat_chain._chain_with_history.invoke.side_effect = [
            mock_response1,
            mock_response2,
        ]

        result1 = chat_chain.predict("First message")
        result2 = chat_chain.predict("Second message")

        assert result1 == "Response 1"
        assert result2 == "Response 2"

    def test_make_chain_multi_turn_history(self) -> None:
        """Test make_chain with multi-turn conversation history."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.ChatPromptTemplate") as mock_prompt,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            mock_response = Mock()
            mock_response.content = "Contextual response"

            mock_prompt_instance = Mock()
            mock_prompt_instance.__or__ = Mock(
                return_value=Mock(invoke=Mock(return_value=mock_response))
            )
            mock_prompt.from_messages.return_value = mock_prompt_instance

            chain = make_chain("llama-3.1-70b-versatile")

            messages: list[Message] = [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "It's sunny today."},
                {"role": "user", "content": "What about tomorrow?"},
            ]
            result = chain(messages)
            assert result == "Contextual response"

    def test_make_chain_empty_content_handling(self) -> None:
        """Test make_chain handles empty content in messages."""
        with (
            patch("mychatbot.chain._get_api_key") as mock_get_key,
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.ChatPromptTemplate") as mock_prompt,
        ):
            mock_get_key.return_value = "test-key"
            mock_groq.return_value = Mock()

            mock_response = Mock()
            mock_response.content = "Response"

            mock_prompt_instance = Mock()
            mock_prompt_instance.__or__ = Mock(
                return_value=Mock(invoke=Mock(return_value=mock_response))
            )
            mock_prompt.from_messages.return_value = mock_prompt_instance

            chain = make_chain("llama-3.1-70b-versatile")

            messages = [{"role": "user", "content": ""}]
            result = chain(messages)
            assert result == "Response"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_chat_chain_with_none_config_system_prompt(self) -> None:
        """Test ChatChain initialization with None system prompt."""
        config = ChatConfig(
            groq_api_key="test_key",
            model_name="llama-3.1-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
            system_prompt=None,
            memory_window=10,
        )

        with (
            patch("mychatbot.chain.ChatGroq") as mock_groq,
            patch("mychatbot.chain.InMemoryChatMessageHistory") as mock_history_cls,
        ):
            mock_groq.return_value = Mock()
            mock_history_cls.return_value = Mock()

            chain = ChatChain(config)
            assert chain is not None

    def test_chat_config_temperature_boundary_zero(self) -> None:
        """Test ChatConfig with temperature at boundary 0.0."""
        config = ChatConfig(
            groq_api_key="test_key",
            model_name="llama-3.1-70b-versatile",
            temperature=0.0,
            max_tokens=1024,
            system_prompt="Test",
            memory_window=10,
        )
        assert config.temperature == 0.0

    def test_chat_config_temperature_boundary_max(self) -> None:
        """Test ChatConfig with temperature at boundary 2.0."""
        config = ChatConfig(
            groq_api_key="test_key",
            model_name="llama-3.1-70b-versatile",
            temperature=2.0,
            max_tokens=1024,
            system_prompt="Test",
            memory_window=10,
        )
        assert config.temperature == 2.0

    def test_chat_config_max_tokens_boundary(self) -> None:
        """Test ChatConfig with max_tokens at boundary."""
        config = ChatConfig(
            groq_api_key="test_key",
            model_name="llama-3.1-70b-versatile",
            temperature=0.7,
            max_tokens=8192,
            system_prompt="Test",
            memory_window=10,
        )
        assert config.max_tokens == 8192


class TestExceptions:
    """Tests for exception classes."""

    def test_chat_chain_error_has_user_message(self) -> None:
        """Test ChatChainError has user_message attribute."""
        error = ChatChainError("Test error")
        assert hasattr(error, "user_message")
        assert isinstance(error.user_message, str)

    def test_authentication_error_user_message(self) -> None:
        """Test AuthenticationError has correct user_message."""
        error = AuthenticationError("Auth failed")
        assert "Authentication" in error.user_message

    def test_rate_limit_error_user_message(self) -> None:
        """Test RateLimitError has correct user_message."""
        error = RateLimitError("Rate limited")
        assert (
            "wait" in error.user_message.lower() or "busy" in error.user_message.lower()
        )

    def test_network_error_user_message(self) -> None:
        """Test NetworkError has correct user_message."""
        error = NetworkError("Network failed")
        assert (
            "internet" in error.user_message.lower()
            or "connection" in error.user_message.lower()
        )

    def test_request_size_error_user_message(self) -> None:
        """Test RequestSizeError has correct user_message."""
        error = RequestSizeError("Request too large")
        assert (
            "shorten" in error.user_message.lower()
            or "long" in error.user_message.lower()
        )

    def test_chat_chain_error_is_exception(self) -> None:
        """Test ChatChainError inherits from Exception."""
        assert issubclass(ChatChainError, Exception)
        assert issubclass(AuthenticationError, ChatChainError)
        assert issubclass(RateLimitError, ChatChainError)
        assert issubclass(NetworkError, ChatChainError)
        assert issubclass(RequestSizeError, ChatChainError)
