"""Tests for chain module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from chain import DEFAULT_MODEL, DEFAULT_SYSTEM_MESSAGE, get_chat_history


class TestGetChatHistory:
    """Tests for get_chat_history function."""

    def test_empty_messages(self):
        """Test with empty message list."""
        result = get_chat_history([])
        assert result == []

    def test_human_message(self):
        """Test conversion of human message."""
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Hello")]
        result = get_chat_history(messages)

        assert result == [{"role": "user", "content": "Hello"}]

    def test_ai_message(self):
        """Test conversion of AI message."""
        from langchain_core.messages import AIMessage

        messages = [AIMessage(content="Hi there")]
        result = get_chat_history(messages)

        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_multiple_messages(self):
        """Test conversion of multiple messages."""
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            HumanMessage(content="How are you?"),
        ]
        result = get_chat_history(messages)

        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]


class TestCreateChain:
    """Tests for create_chain function."""

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    def test_create_chain_requires_api_key(self):
        """Test that create_chain raises error without API key."""
        with patch("chain.load_dotenv"):
            from chain import create_chain

            os.environ.pop("GROQ_API_KEY", None)
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                create_chain()

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    def test_create_chain_defaults(self):
        """Test create_chain with default parameters."""
        with patch("chain.load_dotenv"):
            with patch("chain.ChatGroq") as mock_chat:
                with patch("chain.ChatPromptTemplate") as mock_prompt:
                    from chain import create_chain

                    chat_model, prompt = create_chain()

                    mock_chat.assert_called_once()
                    call_kwargs = mock_chat.call_args.kwargs
                    assert call_kwargs["model"] == DEFAULT_MODEL
                    assert call_kwargs["temperature"] == 0.7

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    def test_create_chain_custom_parameters(self):
        """Test create_chain with custom parameters."""
        with patch("chain.load_dotenv"):
            with patch("chain.ChatGroq") as mock_chat:
                with patch("chain.ChatPromptTemplate") as mock_prompt:
                    from chain import create_chain

                    chat_model, prompt = create_chain(
                        model="llama-3.1-70b-versatile",
                        system_message="Custom system prompt",
                        temperature=0.5,
                    )

                    call_kwargs = mock_chat.call_args.kwargs
                    assert call_kwargs["model"] == "llama-3.1-70b-versatile"
                    assert call_kwargs["temperature"] == 0.5


class TestConstants:
    """Tests for module constants."""

    def test_default_model_is_set(self):
        """Test that DEFAULT_MODEL is defined."""
        assert DEFAULT_MODEL == "mixtral-8x7b-32768"

    def test_default_system_message_is_set(self):
        """Test that DEFAULT_SYSTEM_MESSAGE is defined."""
        assert isinstance(DEFAULT_SYSTEM_MESSAGE, str)
        assert len(DEFAULT_SYSTEM_MESSAGE) > 0
