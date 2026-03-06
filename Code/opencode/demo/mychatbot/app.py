"""Streamlit web chatbot interface."""

import os

import streamlit as st
from dotenv import load_dotenv

from chain import DEFAULT_MODEL, create_chain, get_chat_history

load_dotenv()


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        try:
            chat_model, prompt = create_chain()
            st.session_state.chain = prompt | chat_model
        except ValueError as e:
            st.session_state.chain = None
            st.session_state.error = str(e)


def display_messages() -> None:
    """Display chat messages from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(user_input: str) -> None:
    """Handle user input and generate AI response.

    Args:
        user_input: User's message.
    """
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.chain is None:
        error_msg = st.session_state.get("error", "Unknown error")
        response = f"Error: {error_msg}"
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = get_chat_history(st.session_state.messages)
                response = st.session_state.chain.invoke(
                    {
                        "input": user_input,
                        "history": history,
                    }
                )
                response = response.content

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)


def main() -> None:
    """Main entry point for Streamlit app."""
    st.set_page_config(
        page_title="MyChatbot",
        page_icon="💬",
    )

    initialize_session_state()

    st.title("💬 MyChatbot")
    st.caption(f"Powered by Groq ({DEFAULT_MODEL})")

    display_messages()

    if prompt := st.chat_input("Type your message..."):
        handle_user_input(prompt)


if __name__ == "__main__":
    main()
