"""Tests for react_agent.py. No network, no API keys: the model and HTTP are mocked."""
import json

import pytest
from langchain_core.messages import AIMessage

import react_agent as ra


class FakeModel:
    """Scripted chat model: replays a list of AIMessages, one per invoke."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, config=None):
        msg = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        # fresh id each time: add_messages merges messages that share an id
        return msg.model_copy(update={"id": None})


def tool_call_msg():
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "get_weather_forecast",
            "args": {"location": "Berlin", "date": "2026-10-20"},
            "id": "call_1",
        }],
    )


@pytest.fixture
def fake_weather(monkeypatch):
    monkeypatch.setattr(ra, "_geocode", lambda place: (52.5, 13.4))
    monkeypatch.setattr(
        ra, "_fetch_hourly",
        lambda lat, lon, day: {f"{day}T00:00": 8.0, f"{day}T12:00": 14.0},
    )


def test_tool_summarises_day(fake_weather):
    out = json.loads(
        ra.get_weather_forecast.invoke({"location": "Berlin", "date": "2026-10-20"})
    )
    assert out["min_c"] == 8.0 and out["max_c"] == 14.0 and out["mean_c"] == 11.0


def test_tool_location_not_found(monkeypatch):
    monkeypatch.setattr(ra, "_geocode", lambda place: None)
    out = json.loads(
        ra.get_weather_forecast.invoke({"location": "Nowhere", "date": "2026-10-20"})
    )
    assert out == {"error": "Location not found"}


def test_react_loop_calls_tool_then_answers(fake_weather):
    model = FakeModel([tool_call_msg(), AIMessage(content="It is 11 C in Berlin.")])
    graph = ra.build_graph(model)
    result = graph.invoke({"messages": [("user", "Weather in Berlin?")], "number_of_steps": 0})
    kinds = [m.type for m in result["messages"]]
    assert kinds == ["human", "ai", "tool", "ai"]
    assert result["messages"][-1].content == "It is 11 C in Berlin."
    assert result["number_of_steps"] == 2


def test_cycle_limit_stops_endless_tool_calls(fake_weather):
    model = FakeModel([tool_call_msg()])  # model asks for a tool forever
    graph = ra.build_graph(model, max_steps=3)
    result = graph.invoke({"messages": [("user", "loop")], "number_of_steps": 0})
    assert result["number_of_steps"] == 3


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        ra.get_llm("openai")
