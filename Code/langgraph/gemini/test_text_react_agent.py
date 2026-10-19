"""Tests for text_react_agent.py. No network: the model is scripted and the tool is a stub."""
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import text_react_agent as tra


class ScriptedModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.seen = []

    def invoke(self, messages, config=None):
        self.seen.append(list(messages))
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return AIMessage(content=text)


@tool("get_weather_forecast")
def stub_weather(location: str, date: str) -> str:
    """Stub weather tool."""
    return json.dumps({"location": location, "date": date, "min_c": 8.0, "max_c": 14.0, "mean_c": 11.0})


TOOLS = {"get_weather_forecast": stub_weather}
ACTION = 'Action: get_weather_forecast\nAction Input: {"location": "Berlin", "date": "2026-10-20"}'


def run(replies, max_steps=None):
    model = ScriptedModel(replies)
    graph = tra.build_graph(model, tools=TOOLS, max_steps=max_steps)
    out = graph.invoke(tra.initial_state("Weather in Berlin?", today="2026-10-19"))
    return model, out


def test_parse_action_beats_final_answer():
    kind, name, raw = tra.parse_reply(ACTION + "\nFinal Answer: It is 20 C.")
    assert kind == "action" and name == "get_weather_forecast"
    assert "Final" not in raw and json.loads(raw)["location"] == "Berlin"


def test_parse_cuts_invented_observation():
    kind, name, raw = tra.parse_reply(ACTION + '\nObservation: {"mean_c": 99}')
    assert kind == "action" and "99" not in raw


def test_single_tool_call_then_answer():
    model, out = run([ACTION, "Final Answer: Berlin is 11.0 C on average."])
    assert out["final_answer"] == "Berlin is 11.0 C on average."
    assert out["number_of_steps"] == 2
    # the observation stays in the history the model sees on its second call
    assert any("Observation:" in m.content and "11.0" in m.content for m in model.seen[1])


def test_action_plus_premature_final_answer_is_not_accepted():
    model, out = run([ACTION + "\nFinal Answer: Munich is warmer.", "Final Answer: Berlin is 11.0 C."])
    assert out["final_answer"] == "Berlin is 11.0 C."
    assert model.calls == 2
    # the guessed answer was not fed back into the history
    assert all("Munich is warmer" not in m.content for m in model.seen[1])


def test_step_limit_stops_endless_actions():
    model, out = run([ACTION], max_steps=3)
    assert out["number_of_steps"] == 3
    assert "stopped after 3 steps" in out["final_answer"]


def test_bad_json_returns_observation_error():
    bad = "Action: get_weather_forecast\nAction Input: not json at all"
    model, out = run([bad, "Final Answer: sorry."])
    obs = [m.content for m in model.seen[1] if str(m.content).startswith("Observation:")]
    assert obs and "not valid JSON" in obs[0]
    assert out["final_answer"] == "sorry."


def test_unknown_tool_returns_observation_error():
    model, out = run(["Action: get_stock_price\nAction Input: {}", "Final Answer: no such tool."])
    obs = [m.content for m in model.seen[1] if str(m.content).startswith("Observation:")]
    assert obs and "Unknown tool" in obs[0] and "get_weather_forecast" in obs[0]


def test_wrong_arguments_become_observation():
    model, out = run(['Action: get_weather_forecast\nAction Input: {"city": "X"}', "Final Answer: ok"])
    obs = [m.content for m in model.seen[1] if str(m.content).startswith("Observation:")]
    assert obs and "error" in obs[0]


def test_plain_text_reply_is_the_answer():
    model, out = run(["I cannot help with that."])
    assert out["final_answer"] == "I cannot help with that." and model.calls == 1


def test_system_prompt_has_date():
    assert "2026-10-19" in tra.initial_state("q", today="2026-10-19")["messages"][0].content
