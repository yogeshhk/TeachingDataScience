"""Tests for mychatbot.config module."""

import os
from unittest.mock import patch

import pytest

from mychatbot.config import Config, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, get_config


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_model_when_env_not_set(self) -> None:
        """Test that default model is used when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.model_name == DEFAULT_MODEL

    def test_default_system_prompt_when_env_not_set(self) -> None:
        """Test that default system prompt is used when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.system_prompt == DEFAULT_SYSTEM_PROMPT

    def test_env_var_overrides_model(self) -> None:
        """Test that mychatbot_MODEL env var overrides default."""
        with patch.dict(os.environ, {"mychatbot_MODEL": "llama-3.1-8b-instant"}):
            config = Config()
            assert config.model_name == "llama-3.1-8b-instant"

    def test_env_var_overrides_system_prompt(self) -> None:
        """Test that mychatbot_SYSTEM_PROMPT env var overrides default."""
        with patch.dict(
            os.environ, {"mychatbot_SYSTEM_PROMPT": "You are a custom bot."}
        ):
            config = Config()
            assert config.system_prompt == "You are a custom bot."

    def test_both_env_vars_together(self) -> None:
        """Test that both env vars work together."""
        with patch.dict(
            os.environ,
            {
                "mychatbot_MODEL": "mixtral-8x7b-32768",
                "mychatbot_SYSTEM_PROMPT": "You are a pirate assistant.",
            },
        ):
            config = Config()
            assert config.model_name == "mixtral-8x7b-32768"
            assert config.system_prompt == "You are a pirate assistant."


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_returns_config_instance(self) -> None:
        """Test that get_config returns a Config instance."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_config()
            assert isinstance(config, Config)
            assert config.model_name == DEFAULT_MODEL
            assert config.system_prompt == DEFAULT_SYSTEM_PROMPT
