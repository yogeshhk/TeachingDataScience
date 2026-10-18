"""ReAct agent from scratch with LangGraph (provider switch: Groq or Gemini).

Pattern: LLM node -> (tool calls?) -> tool node -> LLM node ... -> final answer.
Same graph, two back ends. Pick one with --provider or the LLM_PROVIDER env var:

    python react_agent.py --provider groq   --question "Weather in Pune on 2026-10-20?"
    python react_agent.py --provider gemini --question "Is Munich warmer than Berlin tomorrow?"

API keys are read from the environment (never put keys in code):
    GROQ_API_KEY, GEMINI_API_KEY

Models and the cycle limit are the constants below; edit them here.

Ref: "ReAct agent from scratch with Gemini and LangGraph", Gemini API docs.
"""
import argparse
import json
import os
from datetime import date
from typing import Annotated

import requests
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

GROQ_MODEL = "openai/gpt-oss-20b"   # open-weight model served by Groq
GEMINI_MODEL = "gemini-3.8-flash"   # name taken from the Gemini API docs
MAX_STEPS = 6                       # cycle limit: stop endless tool loops


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


# ---------------------------------------------------------------- model
def get_llm(provider):
    """Build the chat model for the chosen provider (imports are lazy)."""
    provider = provider.lower()
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            max_retries=2,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=1.0,
            max_retries=2,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    raise ValueError(f"Unknown provider: {provider!r} (use 'groq' or 'gemini')")


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
def run(question, provider):
    graph = build_graph(get_llm(provider))
    state = {"messages": [("user", question)], "number_of_steps": 0}
    last = None
    for state in graph.stream(state, stream_mode="values"):
        last = state["messages"][-1]
        last.pretty_print()
    return last


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--provider",
        default=os.getenv("LLM_PROVIDER", "groq"),
        choices=["groq", "gemini"],
    )
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
    run(args.question, args.provider)


if __name__ == "__main__":
    main()
