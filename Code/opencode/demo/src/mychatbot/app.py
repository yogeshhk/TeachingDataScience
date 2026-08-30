"""Streamlit web interface for mychatbot."""

import logging
import os

import streamlit as st

from mychatbot.chain import ChatChainError, make_chain
from mychatbot.config import Config, get_config

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]


def get_api_key() -> str:
    """Get GROQ_API_KEY from environment or raise error.

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


def init_session_state() -> None:
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "chain" not in st.session_state:
        st.session_state["chain"] = None


def get_chain(model_name: str, system_prompt: str):
    """Get or create the chain callable.

    Args:
        model_name: The model to use.
        system_prompt: The system prompt to use.

    Returns:
        The chain callable.
    """
    if st.session_state["chain"] is None:
        try:
            st.session_state["chain"] = make_chain(
                model_name=model_name,
                system_prompt=system_prompt,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
    return st.session_state["chain"]


def handle_user_input(user_input: str, chain) -> None:
    """Handle user input and get response.

    Args:
        user_input: The user's message.
        chain: The chain callable.
    """
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chain(st.session_state["messages"])
                st.markdown(response)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )
            except ChatChainError as e:
                st.error(e.user_message)


def clear_conversation() -> None:
    """Clear conversation history."""
    st.session_state["messages"] = []
    st.session_state["chain"] = None


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="mychatbot",
        layout="centered",
    )

    init_session_state()

    config = get_config()

    with st.sidebar:
        st.header("Settings")

        model_name = st.selectbox(
            "Model",
            options=AVAILABLE_MODELS,
            index=0,
        )

        system_prompt = st.text_area(
            "System Prompt",
            value=config.system_prompt,
            height=100,
        )

        if st.button("Clear conversation"):
            clear_conversation()
            st.rerun()

    st.title("mychatbot")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type your message..."):
        chain = get_chain(model_name, system_prompt)
        handle_user_input(prompt, chain)
        st.rerun()


if __name__ == "__main__":
    main()
