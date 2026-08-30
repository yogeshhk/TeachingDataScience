"""CLI interface for mychatbot."""

import logging
import os
import sys
from typing import NoReturn

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from mychatbot.chain import ChatChainError, make_chain
from mychatbot.config import Config, get_config

console = Console()

_log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.WARNING))
logger = logging.getLogger(__name__)


def print_welcome() -> None:
    """Print welcome message using Rich Panel."""
    welcome_text = (
        "[bold]Welcome to mychatbot CLI![/bold]\n\n"
        "Commands:\n"
        "  [cyan]/exit[/], [cyan]/quit[/] - Exit the chat\n"
        "  [cyan]/clear[/] - Clear conversation history\n\n"
        "Type your message and press Enter to chat."
    )
    panel = Panel(welcome_text, title="mychatbot", border_style="blue")
    console.print(panel)


def print_goodbye() -> None:
    """Print goodbye message."""
    console.print("\n[bold green]Goodbye![/bold green]\n")


def print_error(message: str) -> None:
    """Print error message to user."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def should_exit(user_input: str) -> bool:
    """Check if user wants to exit.

    Args:
        user_input: The user's input.

    Returns:
        True if user wants to exit.
    """
    return user_input.lower().strip() in ("/quit", "/exit", "q")


def should_clear(user_input: str) -> bool:
    """Check if user wants to clear history.

    Args:
        user_input: The user's input.

    Returns:
        True if user wants to clear history.
    """
    return user_input.strip() == "/clear"


def get_input() -> str:
    """Get user input from command line.

    Returns:
        The user's input string.
    """
    return console.input("[bold cyan][You][/bold cyan] ")


def print_user_message(message: str) -> None:
    """Print user's message with styling.

    Args:
        message: The user's message.
    """
    console.print(f"[bold cyan][You]:[/bold cyan] {message}")


def print_assistant_message(response: str) -> None:
    """Print assistant's response with markdown rendering and green prefix.

    Args:
        response: The assistant's response.
    """
    console.print("\n[bold green][mychatbot]:[/bold green]")
    md = Markdown(response)
    console.print(md)


def print_token_count(token_count: int | None) -> None:
    """Print token count if available.

    Args:
        token_count: Number of tokens in the response, or None if unavailable.
    """
    if token_count is not None:
        console.print(f"\n[dim]Tokens: {token_count}[/dim]")


def print_turn_count(turn_count: int) -> None:
    """Print turn count.

    Args:
        turn_count: Number of conversation turns.
    """
    console.print(f"[dim]Turn: {turn_count}[/dim]\n")


def run_cli(
    model_name: str | None = None,
    system_prompt: str | None = None,
) -> None:
    """Run the CLI chatbot.

    Args:
        model_name: LLM model name (overrides env var).
        system_prompt: Custom system prompt (overrides env var).
    """
    if model_name is None or system_prompt is None:
        config = get_config()
        model_name = model_name or config.model_name
        system_prompt = system_prompt or config.system_prompt

    try:
        chain = make_chain(
            model_name=model_name,
            system_prompt=system_prompt,
        )
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)

    conversation_history: list[dict[str, str]] = []
    turn_count = 0

    print_welcome()

    while True:
        try:
            user_input = get_input()
        except (EOFError, KeyboardInterrupt):
            print_goodbye()
            break

        if not user_input.strip():
            continue

        if should_exit(user_input):
            print_goodbye()
            break

        if should_clear(user_input):
            conversation_history = []
            turn_count = 0
            console.print("[yellow]Conversation history cleared.[/yellow]\n")
            continue

        print_user_message(user_input)

        conversation_history.append({"role": "user", "content": user_input})

        try:
            response = chain(conversation_history)
            conversation_history.append({"role": "assistant", "content": response})
            print_assistant_message(response)
            turn_count += 1
            print_turn_count(turn_count)

        except ChatChainError as e:
            print_error(e.user_message)
            conversation_history.pop()


def main() -> NoReturn:
    """Main entry point for CLI."""
    run_cli()
    sys.exit(0)


if __name__ == "__main__":
    main()
