"""Tests for mychatbot.app module."""

from unittest.mock import MagicMock, Mock, patch

from mychatbot import app as mychatbot_app


class TestStreamlitApp:
    """Tests for Streamlit app functions."""

    def test_init_session_state(self) -> None:
        """Test that init_session_state initializes messages and chain."""
        mock_st = MagicMock()
        mock_st.session_state = {}

        with patch("mychatbot.app.st", mock_st):
            mychatbot_app.init_session_state()

        assert "messages" in mock_st.session_state
        assert "chain" in mock_st.session_state
