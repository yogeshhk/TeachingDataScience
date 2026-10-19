"""Tests for workflow_compare.py. No network: the fetcher and the model are fakes."""
from langchain_core.messages import AIMessage

import workflow_compare as wc


class FakeModel:
    def __init__(self, text):
        self.text = text
        self.prompts = []

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return AIMessage(content=self.text)


DATA = {"Berlin": {"mean_c": 11.4}, "Munich": {"mean_c": 10.2}, "Nowhere": {"error": "Location not found"}}


def fetch(city, day):
    return DATA[city]


def run(model, a="Munich", b="Berlin"):
    return wc.build_graph(model, fetch=fetch).invoke({"city_a": a, "city_b": b, "day": "2026-10-20"})


def test_code_does_the_comparison():
    out = run(FakeModel("Berlin is warmer by 1.2 C."))
    assert out["warmer"] == "Berlin" and out["diff_c"] == 1.2
    assert out["facts"] == "Berlin is warmer than Munich by 1.2 C (mean Berlin 11.4 C, mean Munich 10.2 C)."


def test_prompt_carries_the_numbers():
    model = FakeModel("Berlin is warmer by 1.2 C.")
    run(model)
    assert "1.2 C" in model.prompts[0] and "11.4" in model.prompts[0]


def test_good_sentence_is_kept():
    out = run(FakeModel("Pack light for Berlin: it is 1.2 C warmer than Munich."))
    assert out["answer"].startswith("Pack light") and out["used_fallback"] is False


def test_sentence_without_the_number_falls_back_to_template():
    out = run(FakeModel("Head to Berlin, it will be a bit hotter."))
    assert out["used_fallback"] is True and "1.2 C" in out["answer"]


def test_sentence_naming_wrong_city_falls_back():
    out = run(FakeModel("Munich is warmer by 1.2 C."))   # the real small-model mistake
    assert out["used_fallback"] is True and out["answer"].startswith("Berlin is warmer")


def test_order_of_cities_does_not_change_the_verdict():
    out = run(FakeModel("x"), a="Berlin", b="Munich")
    assert out["warmer"] == "Berlin" and out["diff_c"] == 1.2


def test_equal_temperatures():
    data = {"A": {"mean_c": 5.0}, "B": {"mean_c": 5.0}}
    graph = wc.build_graph(FakeModel("They are the same."), fetch=lambda c, d: data[c])
    out = graph.invoke({"city_a": "A", "city_b": "B", "day": "d"})
    assert out["diff_c"] == 0 and out["answer"] == "They are the same."


def test_tool_error_stops_without_calling_the_model():
    model = FakeModel("never used")
    out = run(model, a="Nowhere")
    assert model.prompts == [] and "Location not found" in out["answer"]
