"""
Tests for google-adk tool functions.
The `adk` package is mocked so tests run without a Google API key.
Tool helper functions (get_stock_price, web_search, etc.) are tested in isolation.
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the adk package before any import that needs it
adk_mock = MagicMock()
adk_models_mock = MagicMock()
sys.modules.setdefault("adk", adk_mock)
sys.modules.setdefault("adk.models", adk_models_mock)


class TestWebSearch:
    def test_returns_string(self):
        from multi_agent import web_search
        result = web_search("NVIDIA AI chips")
        assert isinstance(result, str)

    def test_result_contains_query(self):
        from multi_agent import web_search
        result = web_search("market trends 2025")
        assert "market trends 2025" in result

    def test_non_empty_result(self):
        from multi_agent import web_search
        result = web_search("anything")
        assert len(result) > 0


class TestGetStockData:
    def test_returns_dict(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 875.50}
        mock_ticker.recommendations.tail.return_value.to_dict.return_value = {}
        with patch("yfinance.Ticker", return_value=mock_ticker):
            from multi_agent import get_stock_data
            result = get_stock_data("NVDA")
            assert isinstance(result, dict)
            assert "price" in result

    def test_price_value(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 123.45}
        mock_ticker.recommendations.tail.return_value.to_dict.return_value = {}
        with patch("yfinance.Ticker", return_value=mock_ticker):
            from multi_agent import get_stock_data
            result = get_stock_data("AAPL")
            assert result["price"] == 123.45


class TestSingleAgentTools:
    def test_get_stock_price_returns_dict(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 500.0}
        with patch("yfinance.Ticker", return_value=mock_ticker):
            from single_agent import get_stock_price
            result = get_stock_price("MSFT")
            assert isinstance(result, dict)
            assert "symbol" in result
            assert result["symbol"] == "MSFT"

    def test_get_stock_price_missing_price(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {}  # no currentPrice key
        with patch("yfinance.Ticker", return_value=mock_ticker):
            from single_agent import get_stock_price
            result = get_stock_price("UNKNOWN")
            assert result["price"] is None


class TestMultipleToolsAgent:
    def test_get_stock_price_returns_float(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 800.0}
        with patch("yfinance.Ticker", return_value=mock_ticker):
            from agent_with_multipletools import get_stock_price
            result = get_stock_price("NVDA")
            assert isinstance(result, float)

    def test_get_company_info_returns_dict(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {"longName": "NVIDIA Corp", "sector": "Technology",
                            "longBusinessSummary": "Makes GPUs"}
        with patch("yfinance.Ticker", return_value=mock_ticker):
            from agent_with_multipletools import get_company_info
            result = get_company_info("NVDA")
            assert isinstance(result, dict)
            assert "name" in result
            assert "sector" in result

    def test_get_analyst_recommendations_returns_string(self):
        mock_ticker = MagicMock()
        mock_ticker.recommendations.to_string.return_value = "Buy: 15, Hold: 5, Sell: 2"
        with patch("yfinance.Ticker", return_value=mock_ticker):
            from agent_with_multipletools import get_analyst_recommendations
            result = get_analyst_recommendations("NVDA")
            assert isinstance(result, str)
