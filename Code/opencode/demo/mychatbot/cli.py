"""CLI chatbot using Rich for terminal UI."""

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from chain import DEFAULT_MODEL, invoke_chain
from chain import create_chain as create_chat_chain
from chain import get_chat_history

load_dotenv()

console = Console()


def display_welcome() -> None:
    """Display welcome message and instructions."""
    console.print("\n[bold blue]Welcome to MyChatbot![/bold blue]")
    console.print(f"Using model: [green]{DEFAULT_MODEL}[/green]\n")
    console.print(
        "Type [bold]exit[/bold] or [bold]quit[/bold] to end the conversation.\n"
    )


def display_response(response: str) -> None:
    """Display AI response as formatted markdown.

    Args:
        response: AI response string.
    """
    console.print("\n[bold]Assistant:[/bold]")
    md = Markdown(response)
    console.print(md)
    console.print()


def run_cli() -> None:
    """Run the CLI chatbot loop."""
    if not os.getenv("GROQ_API_KEY"):
        console.print(
            "[bold red]Error:[/bold red] GROQ_API_KEY not set. "
            "Please create a .env file with your API key."
        )
        console.print("See .env.example for template.\n")
        return

    display_welcome()

    try:
        chat_model, prompt = create_chat_chain()
        chain = prompt | chat_model
        history: list[dict[str, str]] = []

        while True:
            user_input = Prompt.ask(
                "[bold]You[/bold]",
                default="",
            )

            if not user_input.strip():
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[italic]Goodbye![/italic]\n")
                break

            with console.status("[bold green]Thinking...[/bold green]"):
                response = chain.invoke(
                    {
                        "input": user_input,
                        "history": history,
                    }
                )

            history.extend(
                [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response.content},  # type: ignore[dict-item]
                ]
            )

            display_response(response.content)  # type: ignore[arg-type]

    except KeyboardInterrupt:
        console.print("\n[italic]Goodbye![/italic]\n")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}\n")


def main() -> None:
    """Main entry point for CLI chatbot."""
    run_cli()


if __name__ == "__main__":
    main()
