"""Shared types for mychatbot."""

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class Message(TypedDict):
    """Chat message structure."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatConfig(BaseModel):
    """Configuration for the chatbot."""

    groq_api_key: str = Field(..., min_length=1, description="Groq API key")
    model_name: str = Field(
        default="llama-3.1-70b-versatile", description="LLM model name"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    memory_window: int = Field(default=10, ge=1, description="Number of messages to retain")
