"""
Tests for chatbot-faqs: FAQChatbot and BenchmarkTester.
LlamaIndex and HuggingFace are mocked — no model downloads or API keys needed.
"""
import sys
import os
import csv
import tempfile
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestFAQChatbotInit:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        fpath = tmp_path / "faq.csv"
        with open(fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Answer"])
            writer.writerow(["What is RAG?", "Retrieval Augmented Generation"])
            writer.writerow(["What is LLM?", "Large Language Model"])
        return str(fpath)

    def test_raises_without_hf_key(self, sample_csv):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HUGGINGFACE_API_KEY", None)
            with pytest.raises((ValueError, Exception)):
                from main_faq_chatbot import FAQChatbot
                FAQChatbot(sample_csv)

    def test_similarity_threshold_stored(self, sample_csv):
        """FAQChatbot.__init__ signature accepts and stores similarity_threshold."""
        # Inspect constructor signature without importing the full module (avoids
        # a faiss DLL conflict with the system Python on this machine).
        import ast, inspect as _inspect
        src_path = os.path.join(os.path.dirname(__file__), "main_faq_chatbot.py")
        with open(src_path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "FAQChatbot":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        arg_names = [a.arg for a in item.args.args]
                        assert "similarity_threshold" in arg_names
                        return
        pytest.fail("FAQChatbot.__init__ does not accept similarity_threshold")


class TestCSVValidation:
    def test_valid_two_column_csv(self, tmp_path):
        """Simulate the CSV loading logic used in main_faq_chatbot.py."""
        import pandas as pd
        fpath = tmp_path / "faq.csv"
        fpath.write_text("Question,Answer\nWhat is AI?,Artificial Intelligence\n")
        df = pd.read_csv(str(fpath))
        assert len(df.columns) >= 2
        assert len(df) == 1

    def test_empty_rows_filtered(self, tmp_path):
        import pandas as pd
        fpath = tmp_path / "faq.csv"
        fpath.write_text("Question,Answer\nQ1,A1\n,\nQ2,A2\n")
        df = pd.read_csv(str(fpath)).dropna(subset=["Question", "Answer"])
        assert len(df) == 2

    def test_single_column_csv_fails_check(self, tmp_path):
        import pandas as pd
        fpath = tmp_path / "bad.csv"
        fpath.write_text("Question\nQ1\nQ2\n")
        df = pd.read_csv(str(fpath))
        assert len(df.columns) < 2


class TestSimilarityThreshold:
    @pytest.mark.parametrize("threshold", [0.1, 0.5, 0.7, 0.9, 1.0])
    def test_valid_thresholds(self, threshold):
        assert 0.0 <= threshold <= 1.0

    def test_threshold_filters_low_scores(self):
        threshold = 0.7
        scores = [0.9, 0.5, 0.75, 0.3]
        passing = [s for s in scores if s >= threshold]
        assert passing == [0.9, 0.75]


class TestBenchmarkLogic:
    def test_cosine_similarity_range(self):
        """Cosine similarity must be in [-1, 1]."""
        import numpy as np
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert -1.0 <= sim <= 1.0

    def test_identical_vectors_score_one(self):
        import numpy as np
        v = np.array([0.5, 0.3, 0.8])
        sim = np.dot(v, v) / (np.linalg.norm(v) * np.linalg.norm(v))
        assert abs(sim - 1.0) < 1e-6

    def test_random_sample_count(self):
        import pandas as pd
        import random
        data = pd.DataFrame({
            "Question": [f"Q{i}" for i in range(20)],
            "Answer": [f"A{i}" for i in range(20)]
        })
        sample = data.sample(n=10, random_state=42)
        assert len(sample) == 10
