"""ReAct agent from scratch with LangGraph, on a small local model by default.

Pattern: LLM node -> (tool calls?) -> tool node -> LLM node ... -> final answer.
The model comes from ../llm_config.py, the one place to change it. Default: a small model in LM Studio
(for example gemma-2-2b-it), so no API key and no per-token charge:

    python react_agent.py --question "Weather in Pune on 2026-10-20?"
    python react_agent.py --provider groq   --question "Is Munich warmer than Berlin tomorrow?"
    python react_agent.py --model qwen/qwen3-1.7b --question "Weather in Berlin tomorrow?"

Other back ends need keys in the environment (never put keys in code): GROQ_API_KEY, GEMINI_API_KEY.
The cycle limit is MAX_STEPS below. The folder is still called gemini because the seminar deck first
showed this loop with Gemini.

Ref: "ReAct agent from scratch with Gemini and LangGraph", Gemini API docs.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find llm_config.py in the folder above
from llm_config import PROVIDER, PROVIDERS, get_llm  # noqa: E402
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

MAX_STEPS = 6   # cycle limit: stop endless tool loops


# ---------------------------------------------------------------- state
class AgentState(BaseModel):
    """The context object that flows through every node.

    A Pydantic model: typed, validated, with defaults. Nodes read it with
    attribute access (state.messages) and return only the fields they change.
    The add_messages reducer appends new messages instead of overwriting.
    """
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    number_of_steps: int = 0


# ---------------------------------------------------------------- tools
class SearchInput(BaseModel):
    location: str = Field(description="The city, e.g., San Francisco")
    date: str = Field(description="The forecasting date, format yyyy-mm-dd")


def _geocode(place):
    """Return (latitude, longitude) for a place name, or None."""
    from geopy.geocoders import Nominatim

    hit = Nominatim(user_agent="weather-app").geocode(place)
    return (hit.latitude, hit.longitude) if hit else None


def _fetch_hourly(lat, lon, day):
    """Hourly temperatures (deg C) for one day from the Open-Meteo API."""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "start_date": day,
            "end_date": day,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return dict(zip(data["hourly"]["time"], data["hourly"]["temperature_2m"]))


@tool("get_weather_forecast", args_schema=SearchInput)
def get_weather_forecast(location: str, date: str) -> str:
    """Retrieves the weather using the Open-Meteo API.

    Takes a city and a date (yyyy-mm-dd). Returns JSON with min, max and
    mean temperature in deg C for that day.
    """
    try:
        coords = _geocode(location)
        if coords is None:
            return json.dumps({"error": "Location not found"})
        hourly = _fetch_hourly(coords[0], coords[1], date)
        temps = [t for t in hourly.values() if t is not None]
        if not temps:
            return json.dumps({"error": "No data for that date"})
        return json.dumps({
            "location": location,
            "date": date,
            "min_c": min(temps),
            "max_c": max(temps),
            "mean_c": round(sum(temps) / len(temps), 1),
        })
    except Exception as exc:  # network, rate limit, bad date
        return json.dumps({"error": str(exc)})


TOOLS = [get_weather_forecast]


# ---------------------------------------------------------------- graph
def build_graph(model, tools=None, max_steps=None):
    """Compile the ReAct graph for any chat model that supports bind_tools."""
    tools = tools or TOOLS
    max_steps = max_steps or MAX_STEPS
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    def call_model(state: AgentState):
        response = model_with_tools.invoke(state.messages)
        return {"messages": [response], "number_of_steps": state.number_of_steps + 1}

    def call_tool(state: AgentState):
        outputs = []
        for tool_call in state.messages[-1].tool_calls:
            result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=result if isinstance(result, str) else json.dumps(result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    def should_continue(state: AgentState):
        if state.number_of_steps >= max_steps:
            return "end"  # cycle limit, never loop forever
        if not state.messages[-1].tool_calls:
            return "end"
        return "continue"

    workflow = StateGraph(AgentState)
    workflow.add_node("llm", call_model)
    workflow.add_node("tools", call_tool)
    workflow.add_edge(START, "llm")
    workflow.add_conditional_edges(
        "llm", should_continue, {"continue": "tools", "end": END}
    )
    workflow.add_edge("tools", "llm")
    return workflow.compile()


# ---------------------------------------------------------------- run
def run(question, provider=None, model=None):
    graph = build_graph(get_llm(provider, model=model))
    state = {"messages": [("user", question)], "number_of_steps": 0}
    last = None
    for state in graph.stream(state, stream_mode="values"):
        last = state["messages"][-1]
        last.pretty_print()
    return last


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider", default=PROVIDER, choices=list(PROVIDERS))
    parser.add_argument("--model", default=None, help="model id; default comes from llm_config.py")
    parser.add_argument(
        "--question",
        default=f"What is the weather in Berlin on {date.today().isoformat()}?",
    )
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    run(args.question, args.provider, args.model)


if __name__ == "__main__":
    main()
