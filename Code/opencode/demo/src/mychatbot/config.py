"""Configuration loading for mychatbot."""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    model_name: str = field(
        default_factory=lambda: os.getenv("mychatbot_MODEL") or DEFAULT_MODEL
    )
    system_prompt: str = field(
        default_factory=lambda: (
            os.getenv("mychatbot_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
        )
    )

    def __post_init__(self) -> None:
        """Load .env file after initialization."""
        load_dotenv()
        if not os.getenv("mychatbot_MODEL"):
            object.__setattr__(
                self, "model_name", os.getenv("mychatbot_MODEL") or DEFAULT_MODEL
            )
        if not os.getenv("mychatbot_SYSTEM_PROMPT"):
            object.__setattr__(
                self,
                "system_prompt",
                os.getenv("mychatbot_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
            )


def get_config() -> Config:
    """Get application configuration.

    Returns:
        Config instance with model and system prompt from env vars.
    """
    return Config()
