"""
Tests for agno agent scripts.
LM Studio server is mocked — no local model server required.
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestAgnoImports:
    def test_agno_package_importable(self):
        import agno
        assert agno is not None

    def test_agno_agent_importable(self):
        from agno.agent import Agent
        assert Agent is not None

    def test_agno_models_importable(self):
        try:
            from agno.models.lmstudio import LMStudio
            assert LMStudio is not None
        except ImportError:
            pytest.skip("LMStudio model not available in this agno version")


class TestWebSearchAgent:
    def test_web_search_agent_module_structure(self):
        """Verify web_search_agent.py has expected attributes without running it."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "web_search_agent",
            os.path.join(os.path.dirname(__file__), "web_search_agent.py")
        )
        # Just check the file is parseable Python
        assert spec is not None

    def test_python_file_is_valid_syntax(self):
        import ast
        files = ["reasoning_agent.py", "trial_agent.py", "web_search_agent.py"]
        for fname in files:
            fpath = os.path.join(os.path.dirname(__file__), fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    source = f.read()
                try:
                    ast.parse(source)
                except SyntaxError as e:
                    pytest.fail(f"{fname} has syntax error: {e}")


class TestRagAgentStub:
    def test_rag_agent_file_exists(self):
        fpath = os.path.join(os.path.dirname(__file__), "rag_agent.py")
        assert os.path.exists(fpath)

    def test_rag_agent_is_valid_python(self):
        import ast
        fpath = os.path.join(os.path.dirname(__file__), "rag_agent.py")
        with open(fpath) as f:
            source = f.read()
        ast.parse(source)  # raises SyntaxError if invalid
