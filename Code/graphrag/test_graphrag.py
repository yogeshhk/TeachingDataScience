"""
Tests for graphrag helper functions.
The distance() function is extracted from graphrag_langchain_spicejet.py for isolated testing.
"""
import pytest


def distance(x):
    """Categorize flight distance — copied from graphrag_langchain_spicejet.py."""
    if x < 500:
        return "near, short distance"
    elif x < 1000:
        return " double distance but not very close"
    else:
        return 'far away'


class TestDistanceFunction:
    def test_below_500_is_near(self):
        assert distance(100) == "near, short distance"
        assert distance(499) == "near, short distance"

    def test_boundary_500_is_medium(self):
        assert distance(500) == " double distance but not very close"

    def test_between_500_and_1000(self):
        assert distance(750) == " double distance but not very close"
        assert distance(999) == " double distance but not very close"

    def test_boundary_1000_is_far(self):
        assert distance(1000) == 'far away'

    def test_above_1000_is_far(self):
        assert distance(5000) == 'far away'

    def test_zero_is_near(self):
        assert distance(0) == "near, short distance"


class TestGraphragImports:
    def test_networkx_importable(self):
        import networkx as nx
        g = nx.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", relation="near, short distance")
        assert g.number_of_nodes() == 2
        assert g.number_of_edges() == 1

    def test_langchain_openai_importable(self):
        from langchain_openai import ChatOpenAI
        assert ChatOpenAI is not None

    def test_pandas_importable(self):
        import pandas as pd
        df = pd.DataFrame({"origin": ["HYD"], "destination": ["BOM"], "distance": [600]})
        df["distance_label"] = df["distance"].apply(distance)
        assert df["distance_label"].iloc[0] == " double distance but not very close"
