"""Tests for mychatbot.cli module."""

import os
from unittest.mock import Mock, patch

import pytest

from mychatbot.cli import (
    get_input,
    main,
    print_assistant_message,
    print_error,
    print_goodbye,
    print_token_count,
    print_turn_count,
    print_user_message,
    print_welcome,
    run_cli,
    should_clear,
    should_exit,
)
from mychatbot.config import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, get_config


class TestGetEnvConfig:
    """Tests for get_env_config function."""

    def test_returns_defaults_when_env_not_set(self) -> None:
        """Test that defaults are returned when env vars not set."""
        with patch.dict("os.environ", {}, clear=True):
            config = get_config()
            assert config.model_name == DEFAULT_MODEL
            assert config.system_prompt == DEFAULT_SYSTEM_PROMPT

    def test_returns_env_model_when_set(self) -> None:
        """Test that env model is returned when set."""
        with patch.dict("os.environ", {"mychatbot_MODEL": "custom-model"}):
            config = get_config()
            assert config.model_name == "custom-model"
            assert config.system_prompt == DEFAULT_SYSTEM_PROMPT

    def test_returns_env_system_prompt_when_set(self) -> None:
        """Test that env system prompt is returned when set."""
        with patch.dict("os.environ", {"mychatbot_SYSTEM_PROMPT": "Custom prompt"}):
            config = get_config()
            assert config.model_name == DEFAULT_MODEL
            assert config.system_prompt == "Custom prompt"

    def test_returns_both_env_values_when_set(self) -> None:
        """Test that both env values are returned when set."""
        with patch.dict(
            os.environ,
            {
                "mychatbot_MODEL": "llama-3.1-70b-versatile",
                "mychatbot_SYSTEM_PROMPT": "You are a coding assistant.",
            },
            clear=True,
        ):
            config = get_config()
            assert config.model_name == "llama-3.1-70b-versatile"
            assert config.system_prompt == "You are a coding assistant."

    def test_empty_env_model_uses_default(self) -> None:
        """Test that empty env model uses default."""
        with patch.dict("os.environ", {"mychatbot_MODEL": ""}):
            config = get_config()
            assert config.model_name == DEFAULT_MODEL


class TestShouldExit:
    """Tests for should_exit function."""

    def test_should_exit_with_quit(self) -> None:
        """Test that /quit triggers exit."""
        assert should_exit("/quit") is True
        assert should_exit("/QUIT") is True

    def test_should_exit_with_exit(self) -> None:
        """Test that /exit triggers exit."""
        assert should_exit("/exit") is True
        assert should_exit("/EXIT") is True

    def test_should_exit_with_q(self) -> None:
        """Test that q triggers exit."""
        assert should_exit("q") is True
        assert should_exit("Q") is True

    def test_should_not_exit_regular_input(self) -> None:
        """Test that regular input doesn't trigger exit."""
        assert should_exit("Hello") is False
        assert should_exit("What is Python?") is False

    def test_trim_whitespace(self) -> None:
        """Test that whitespace is trimmed."""
        assert should_exit("  q  ") is True

    def test_empty_string(self) -> None:
        """Test that empty string doesn't exit."""
        assert should_exit("") is False

    def test_similar_commands_not_exit(self) -> None:
        """Test that similar commands don't trigger exit."""
        assert should_exit("/quitnow") is False
        assert should_exit("/exitall") is False
        assert should_exit("quit") is False
        assert should_exit("exit") is False


class TestShouldClear:
    """Tests for should_clear function."""

    def test_should_clear_with_clear(self) -> None:
        """Test that /clear triggers clear."""
        assert should_clear("/clear") is True
        assert should_clear("  /clear  ") is True

    def test_should_clear_with_whitespace(self) -> None:
        """Test that /clear with whitespace is recognized."""
        assert should_clear("  /clear  ") is True

    def test_should_not_clear_regular_input(self) -> None:
        """Test that regular input doesn't trigger clear."""
        assert should_clear("clear") is False
        assert should_clear("clear history") is False
        assert should_clear("/clearsomething") is False
        assert should_clear("") is False

    def test_case_sensitive(self) -> None:
        """Test that clear is case sensitive."""
        assert should_clear("/Clear") is False


class TestPrintWelcome:
    """Tests for print_welcome function."""

    @patch("mychatbot.cli.console")
    def test_print_welcome_calls_console(self, mock_console: Mock) -> None:
        """Test that print_welcome prints to console."""
        print_welcome()
        mock_console.print.assert_called_once()

    @patch("mychatbot.cli.console")
    def test_print_welcome_contains_welcome_text(self, mock_console: Mock) -> None:
        """Test that welcome message contains expected text."""
        print_welcome()
        call_args = mock_console.print.call_args[0][0]
        assert "mychatbot" in call_args.title


class TestPrintGoodbye:
    """Tests for print_goodbye function."""

    @patch("mychatbot.cli.console")
    def test_print_goodbye_calls_console(self, mock_console: Mock) -> None:
        """Test that print_goodbye prints to console."""
        print_goodbye()
        mock_console.print.assert_called_once()


class TestPrintError:
    """Tests for print_error function."""

    @patch("mychatbot.cli.console")
    def test_print_error_calls_console(self, mock_console: Mock) -> None:
        """Test that print_error prints to console."""
        print_error("Test error")
        mock_console.print.assert_called_once()

    @patch("mychatbot.cli.console")
    def test_print_error_contains_message(self, mock_console: Mock) -> None:
        """Test that error message is printed."""
        print_error("Something went wrong")
        call_args = mock_console.print.call_args[0][0]
        assert "Something went wrong" in call_args


class TestGetInput:
    """Tests for get_input function."""

    @patch("mychatbot.cli.console")
    def test_get_input_returns_user_input(self, mock_console: Mock) -> None:
        """Test that get_input returns user input."""
        mock_console.input.return_value = "Hello world"
        result = get_input()
        assert result == "Hello world"

    @patch("mychatbot.cli.console")
    def test_get_input_returns_empty_string(self, mock_console: Mock) -> None:
        """Test that get_input returns empty string."""
        mock_console.input.return_value = ""
        result = get_input()
        assert result == ""


class TestPrintUserMessage:
    """Tests for print_user_message function."""

    @patch("mychatbot.cli.console")
    def test_print_user_message_calls_console(self, mock_console: Mock) -> None:
        """Test that print_user_message prints to console."""
        print_user_message("Hello")
        mock_console.print.assert_called_once()

    @patch("mychatbot.cli.console")
    def test_print_user_message_contains_message(self, mock_console: Mock) -> None:
        """Test that user message is in output."""
        print_user_message("My message")
        call_args = mock_console.print.call_args[0][0]
        assert "My message" in call_args


class TestPrintAssistantMessage:
    """Tests for print_assistant_message function."""

    @patch("mychatbot.cli.Markdown")
    @patch("mychatbot.cli.console")
    def test_print_assistant_message_calls_console(
        self, mock_console: Mock, mock_markdown: Mock
    ) -> None:
        """Test that print_assistant_message prints to console."""
        mock_markdown.return_value = Mock()
        print_assistant_message("Response")
        assert mock_console.print.call_count == 2

    @patch("mychatbot.cli.Markdown")
    @patch("mychatbot.cli.console")
    def test_print_assistant_message_renders_markdown(
        self, mock_console: Mock, mock_markdown: Mock
    ) -> None:
        """Test that markdown is rendered."""
        mock_markdown_instance = Mock()
        mock_markdown.return_value = mock_markdown_instance
        print_assistant_message("# Hello")
        mock_markdown.assert_called_once_with("# Hello")

    @patch("mychatbot.cli.Markdown")
    @patch("mychatbot.cli.console")
    def test_print_assistant_message_empty_response(
        self, mock_console: Mock, mock_markdown: Mock
    ) -> None:
        """Test that empty response is handled."""
        mock_markdown.return_value = Mock()
        print_assistant_message("")
        mock_markdown.assert_called_once_with("")


class TestPrintTokenCount:
    """Tests for print_token_count function."""

    @patch("mychatbot.cli.console")
    def test_print_token_count_with_value(self, mock_console: Mock) -> None:
        """Test that token count is printed when provided."""
        print_token_count(100)
        mock_console.print.assert_called_once()

    @patch("mychatbot.cli.console")
    def test_print_token_count_with_none(self, mock_console: Mock) -> None:
        """Test that nothing is printed when token count is None."""
        print_token_count(None)
        mock_console.print.assert_not_called()

    @patch("mychatbot.cli.console")
    def test_print_token_count_with_zero(self, mock_console: Mock) -> None:
        """Test that zero tokens are printed."""
        print_token_count(0)
        mock_console.print.assert_called_once()


class TestPrintTurnCount:
    """Tests for print_turn_count function."""

    @patch("mychatbot.cli.console")
    def test_print_turn_count_calls_console(self, mock_console: Mock) -> None:
        """Test that turn count is printed."""
        print_turn_count(5)
        mock_console.print.assert_called_once()

    @patch("mychatbot.cli.console")
    def test_print_turn_count_with_zero(self, mock_console: Mock) -> None:
        """Test that zero turns are printed."""
        print_turn_count(0)
        mock_console.print.assert_called_once()


class TestRunCli:
    """Tests for run_cli function."""

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_run_cli_exits_on_keyboard_interrupt(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that run_cli handles KeyboardInterrupt."""
        mock_chain = Mock()
        mock_make_chain.return_value = mock_chain
        mock_get_input.side_effect = KeyboardInterrupt()

        with patch("mychatbot.cli.print_welcome"):
            run_cli()

        mock_make_chain.assert_called_once()

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_run_cli_exits_on_eof(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that run_cli handles EOFError."""
        mock_chain = Mock()
        mock_make_chain.return_value = mock_chain
        mock_get_input.side_effect = EOFError()

        with patch("mychatbot.cli.print_welcome"):
            run_cli()

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_run_cli_handles_empty_input(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that run_cli handles empty input."""
        mock_chain = Mock()
        mock_chain.return_value = "Hello response"
        mock_make_chain.return_value = mock_chain
        mock_get_input.side_effect = ["", "Hello", "/quit"]

        with patch("mychatbot.cli.print_welcome"):
            with patch("mychatbot.cli.print_goodbye"):
                with patch("mychatbot.cli.print_assistant_message"):
                    run_cli()

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_run_cli_handles_exit_command(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that run_cli exits on /exit command."""
        mock_chain = Mock()
        mock_make_chain.return_value = mock_chain
        mock_get_input.return_value = "/exit"

        with patch("mychatbot.cli.print_welcome"):
            with patch("mychatbot.cli.print_goodbye"):
                run_cli()

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_run_cli_handles_quit_command(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that run_cli exits on /quit command."""
        mock_chain = Mock()
        mock_make_chain.return_value = mock_chain
        mock_get_input.return_value = "/quit"

        with patch("mychatbot.cli.print_welcome"):
            with patch("mychatbot.cli.print_goodbye"):
                run_cli()

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_run_cli_handles_clear_command(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that run_cli clears history on /clear command."""
        mock_chain = Mock()
        mock_make_chain.return_value = mock_chain
        mock_get_input.side_effect = ["/clear", "/quit"]

        with patch("mychatbot.cli.print_welcome"):
            with patch("mychatbot.cli.print_goodbye"):
                with patch("mychatbot.cli.console") as mock_console:
                    run_cli()
                    assert mock_console.print.called

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_run_cli_sends_message_and_gets_response(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that run_cli sends message and gets response."""
        mock_chain = Mock()
        mock_chain.return_value = "Hello, how can I help?"
        mock_make_chain.return_value = mock_chain
        mock_get_input.side_effect = ["Hello", "/quit"]

        with patch("mychatbot.cli.print_welcome"):
            with patch("mychatbot.cli.print_goodbye"):
                with patch("mychatbot.cli.print_user_message"):
                    with patch("mychatbot.cli.print_assistant_message"):
                        run_cli()

        mock_chain.assert_called_once()

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_run_cli_handles_chat_chain_error(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that run_cli handles ChatChainError."""
        from mychatbot.chain import ChatChainError

        mock_chain = Mock()
        mock_chain.side_effect = ChatChainError("API error")
        mock_make_chain.return_value = mock_chain
        mock_get_input.side_effect = ["Hello", "/quit"]

        with patch("mychatbot.cli.print_welcome"):
            with patch("mychatbot.cli.print_goodbye"):
                with patch("mychatbot.cli.print_user_message"):
                    with patch("mychatbot.cli.print_error"):
                        run_cli()

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_run_cli_uses_env_config_when_no_args(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that run_cli uses env config when no args provided."""
        mock_chain = Mock()
        mock_make_chain.return_value = mock_chain
        mock_get_input.side_effect = ["/quit"]

        with patch("mychatbot.cli.print_welcome"):
            with patch("mychatbot.cli.print_goodbye"):
                with patch("mychatbot.cli.get_config") as mock_env:
                    mock_env.return_value = Mock(
                        model_name="env-model", system_prompt="env-prompt"
                    )
                    run_cli()

        mock_make_chain.assert_called_once_with(
            model_name="env-model",
            system_prompt="env-prompt",
        )

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_run_cli_args_override_env(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that args override env config."""
        mock_chain = Mock()
        mock_make_chain.return_value = mock_chain
        mock_get_input.side_effect = ["/quit"]

        with patch("mychatbot.cli.print_welcome"):
            with patch("mychatbot.cli.print_goodbye"):
                with patch("mychatbot.cli.get_config") as mock_env:
                    mock_env.return_value = Mock(
                        model_name="env-model", system_prompt="env-prompt"
                    )
                    run_cli(
                        model_name="override-model",
                        system_prompt="override-prompt",
                    )

        mock_make_chain.assert_called_once_with(
            model_name="override-model",
            system_prompt="override-prompt",
        )

    @patch("mychatbot.cli.make_chain")
    def test_run_cli_key(self, mock_make_chain: Mock) -> None:
        """Test that_fails_without_api run_cli fails when API key is missing."""
        mock_make_chain.side_effect = ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it to your Groq API key."
        )

        with patch("sys.argv", ["mychatbot-cli"]):
            with pytest.raises(SystemExit) as exc_info:
                run_cli()
            assert exc_info.value.code == 1


class TestMain:
    """Tests for main function."""

    @patch("mychatbot.cli.run_cli")
    @patch("mychatbot.cli.sys")
    def test_main_calls_run_cli(self, mock_sys: Mock, mock_run_cli: Mock) -> None:
        """Test that main calls run_cli."""
        main()
        mock_run_cli.assert_called_once()
        mock_sys.exit.assert_called_once_with(0)

    @patch("mychatbot.cli.run_cli")
    @patch("mychatbot.cli.sys")
    def test_main_exits_with_zero(self, mock_sys: Mock, mock_run_cli: Mock) -> None:
        """Test that main exits with code 0."""
        main()
        mock_sys.exit.assert_called_once_with(0)


class TestEdgeCases:
    """Tests for edge cases."""

    @patch("mychatbot.cli.console")
    def test_print_assistant_with_special_characters(self, mock_console: Mock) -> None:
        """Test that special characters are handled."""
        with patch("mychatbot.cli.Markdown") as mock_md:
            mock_md.return_value = Mock()
            print_assistant_message("Hello <world> & 'quotes'")

    @patch("mychatbot.cli.console")
    def test_print_assistant_with_unicode(self, mock_console: Mock) -> None:
        """Test that unicode is handled."""
        with patch("mychatbot.cli.Markdown") as mock_md:
            mock_md.return_value = Mock()
            print_assistant_message("Hello 你好 🔥")

    @patch("mychatbot.cli.console")
    def test_print_assistant_with_markdown(self, mock_console: Mock) -> None:
        """Test that markdown is rendered."""
        with patch("mychatbot.cli.Markdown") as mock_md:
            mock_md.return_value = Mock()
            print_assistant_message("# Header\n**bold** and *italic*")

    def test_should_clear_with_leading_slash(self) -> None:
        """Test clear with leading whitespace containing slash."""
        assert should_clear("  /clear") is True

    def test_should_exit_with_leading_whitespace(self) -> None:
        """Test exit with leading whitespace."""
        assert should_exit("  /exit") is True


class TestMultiTurnConversation:
    """Tests for multi-turn conversation scenarios."""

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_multi_turn_conversation(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test multi-turn conversation maintains history."""
        mock_chain = Mock()
        mock_chain.side_effect = [
            "Response 1",
            "Response 2",
            "Response 3",
        ]
        mock_make_chain.return_value = mock_chain
        mock_get_input.side_effect = [
            "First message",
            "Second message",
            "Third message",
            "/quit",
        ]

        with patch("mychatbot.cli.print_welcome"):
            with patch("mychatbot.cli.print_goodbye"):
                with patch("mychatbot.cli.print_user_message"):
                    with patch("mychatbot.cli.print_assistant_message"):
                        run_cli()

        assert mock_chain.call_count == 3

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_history_cleared_after_clear_command(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that history is cleared after /clear command."""
        captured_args: list = []

        def capture_side_effect(*args, **kwargs):
            captured_args.append(list(args[0]) if args else [])
            return "Response" if len(captured_args) == 1 else "Response 2"

        mock_chain = Mock(side_effect=capture_side_effect)
        mock_make_chain.return_value = mock_chain
        mock_get_input.side_effect = [
            "First message",
            "/clear",
            "After clear",
            "/quit",
        ]

        with patch("mychatbot.cli.print_welcome"):
            with patch("mychatbot.cli.print_goodbye"):
                with patch("mychatbot.cli.print_user_message"):
                    with patch("mychatbot.cli.print_assistant_message"):
                        run_cli()

        assert mock_chain.call_count == 2
        assert len(captured_args[0]) == 1
        assert len(captured_args[1]) == 1

    @patch("mychatbot.cli.get_input")
    @patch("mychatbot.cli.make_chain")
    def test_conversation_turn_count_increments(
        self, mock_make_chain: Mock, mock_get_input: Mock
    ) -> None:
        """Test that turn count increments correctly."""
        mock_chain = Mock()
        mock_chain.return_value = "Response"
        mock_make_chain.return_value = mock_chain
        mock_get_input.side_effect = [
            "Message 1",
            "Message 2",
            "/quit",
        ]

        with patch("mychatbot.cli.print_welcome"):
            with patch("mychatbot.cli.print_goodbye"):
                with patch("mychatbot.cli.print_user_message"):
                    with patch("mychatbot.cli.print_assistant_message"):
                        with patch("mychatbot.cli.print_turn_count") as mock_print_turn:
                            run_cli()
                            assert mock_print_turn.call_count == 2
                            mock_print_turn.assert_called_with(2)


class TestDefaults:
    """Tests for default values."""

    def test_default_model_value(self) -> None:
        """Test that DEFAULT_MODEL is set."""
        assert DEFAULT_MODEL == "llama-3.3-70b-versatile"

    def test_default_system_prompt_value(self) -> None:
        """Test that DEFAULT_SYSTEM_PROMPT is set."""
        assert DEFAULT_SYSTEM_PROMPT == "You are a helpful assistant."

    def test_default_system_prompt_is_string(self) -> None:
        """Test that DEFAULT_SYSTEM_PROMPT is a non-empty string."""
        assert isinstance(DEFAULT_SYSTEM_PROMPT, str)
        assert len(DEFAULT_SYSTEM_PROMPT) > 0
