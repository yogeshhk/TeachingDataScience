"""Text ReAct: tools without tool calling, for small models that ignore bind_tools.

The model never sees a tool schema. A system prompt teaches a plain-text protocol:

    Action: get_weather_forecast
    Action Input: {"location": "Berlin", "date": "2026-10-20"}

The graph parses that text, runs the tool, and appends "Observation: ..." to the history. Rules that a
2B model needs (measured: it writes an Action AND a guessed Final Answer in one reply):
    * an Action wins: any Final Answer in the same reply is ignored
    * the reply is cut at "Observation:" (the model must not invent tool results)
    * tool errors, bad JSON and unknown tool names come back as Observation text, never as exceptions
    * a step limit stops endless loops

    python text_react_agent.py --question "What is the weather in Berlin on 2026-10-20?"
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find llm_config.py in the folder above
from llm_config import PROVIDER, PROVIDERS, get_llm  # noqa: E402
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import react_agent as ra  # reuse the weather tool and its helpers  # noqa: E402

MAX_STEPS = 6

SYSTEM_PROMPT = """You answer weather questions. You can use one tool.

Tool: get_weather_forecast
  Input: JSON with "location" (a city) and "date" (yyyy-mm-dd)
  Output: JSON with min_c, max_c, mean_c in degrees Celsius for that day

Reply in exactly one of two ways.

To use the tool, write only these two lines and then stop:
Action: get_weather_forecast
Action Input: {{"location": "<city>", "date": "<yyyy-mm-dd>"}}

When you already have every Observation you need, write only:
Final Answer: <your answer in one or two sentences>

Never write an Observation yourself and never write a Final Answer in the same reply as an Action.
Use one Action per reply. Today is {today}."""


# ---------------------------------------------------------------- state
class TextAgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    number_of_steps: int = 0
    final_answer: str = ""
    pending: dict = Field(default_factory=dict)   # parsed Action waiting to run: {"name", "input"}


# ---------------------------------------------------------------- parsing
_ACTION = re.compile(r"Action\s*:\s*([A-Za-z_][\w\-]*)", re.I)
_INPUT = re.compile(r"Action\s*Input\s*:\s*(.*)", re.I | re.S)
_FINAL = re.compile(r"Final\s*Answer\s*:\s*(.*)", re.I | re.S)


def parse_reply(text):
    """Return (kind, a, b): ("action", name, raw_input) | ("final", answer, "") | ("none", text, "").

    An Action always wins over a Final Answer in the same reply. Anything from "Observation:" on is dropped.
    """
    text = re.split(r"\n\s*Observation\s*:", text, maxsplit=1, flags=re.I)[0]
    act = _ACTION.search(text)
    if act:
        rest = text[act.end():]
        inp = _INPUT.search(rest)
        raw = inp.group(1) if inp else ""
        raw = re.split(r"\n\s*Final\s*Answer\s*:", raw, maxsplit=1, flags=re.I)[0].strip()
        return "action", act.group(1), raw
    fin = _FINAL.search(text)
    if fin:
        return "final", fin.group(1).strip(), ""
    return "none", text.strip(), ""


def _json_object(raw):
    """Parse the Action Input; fall back to the first {...} block (models add chatter or code fences)."""
    try:
        return json.loads(raw)
    except ValueError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
        raise


# ---------------------------------------------------------------- graph
def build_graph(model, tools=None, max_steps=None):
    """Compile the text-protocol ReAct graph. `model` only needs .invoke(messages)."""
    tools = tools or {t.name: t for t in ra.TOOLS}
    max_steps = max_steps or MAX_STEPS

    def call_model(state: TextAgentState):
        reply = model.invoke(state.messages)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        kind, a, b = parse_reply(text)
        steps = state.number_of_steps + 1
        if kind == "action":
            # keep only the Action in the history, so a guessed Final Answer is never fed back
            kept = f"Action: {a}\nAction Input: {b}"
            return {"messages": [AIMessage(content=kept)], "number_of_steps": steps,
                    "pending": {"name": a, "input": b}}
        answer = a if kind == "final" else a or "(empty reply)"
        return {"messages": [AIMessage(content=text)], "number_of_steps": steps,
                "final_answer": answer, "pending": {}}

    def call_tool(state: TextAgentState):
        name, raw = state.pending["name"], state.pending["input"]
        if name not in tools:
            obs = json.dumps({"error": f"Unknown tool {name!r}. Available: {', '.join(tools)}"})
        else:
            try:
                obs = tools[name].invoke(_json_object(raw))
                obs = obs if isinstance(obs, str) else json.dumps(obs)
            except ValueError as exc:
                obs = json.dumps({"error": f"Action Input is not valid JSON: {exc}"})
            except Exception as exc:  # bad arguments, network, anything the tool raises
                obs = json.dumps({"error": str(exc)})
        return {"messages": [HumanMessage(content=f"Observation: {obs}")], "pending": {}}

    def should_continue(state: TextAgentState):
        if state.pending and state.number_of_steps < max_steps:
            return "continue"
        return "end"

    def finish(state: TextAgentState):
        if state.final_answer:
            return {}
        return {"final_answer": f"(stopped after {state.number_of_steps} steps without a Final Answer)"}

    workflow = StateGraph(TextAgentState)
    workflow.add_node("llm", call_model)
    workflow.add_node("tools", call_tool)
    workflow.add_node("finish", finish)
    workflow.add_edge(START, "llm")
    workflow.add_conditional_edges("llm", should_continue, {"continue": "tools", "end": "finish"})
    workflow.add_edge("tools", "llm")
    workflow.add_edge("finish", END)
    return workflow.compile()


# ---------------------------------------------------------------- run
def initial_state(question, today=None):
    system = SYSTEM_PROMPT.format(today=today or date.today().isoformat())
    return {"messages": [SystemMessage(content=system), HumanMessage(content=question)],
            "number_of_steps": 0}


def run(question, provider=None, model=None, verbose=True):
    graph = build_graph(get_llm(provider, model=model))
    last = None
    for last in graph.stream(initial_state(question), stream_mode="values"):
        if verbose and last["messages"]:
            last["messages"][-1].pretty_print()
    return last


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider", default=PROVIDER, choices=list(PROVIDERS))
    parser.add_argument("--model", default=None, help="model id; default comes from llm_config.py")
    parser.add_argument("--question",
                        default=f"What is the weather in Berlin on {date.today().isoformat()}?")
    args = parser.parse_args()
    final = run(args.question, args.provider, args.model)
    print("\nFINAL:", final["final_answer"])


if __name__ == "__main__":
    main()
