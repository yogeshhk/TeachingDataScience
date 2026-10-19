"""A fixed LangGraph WORKFLOW (not an agent): is city A warmer than city B on a day?

    START -> fetch_a -> fetch_b -> compare -> phrase -> END

Code does the facts: it fetches both forecasts, compares the means and picks the warmer city. The small
model only turns the computed result into a sentence, and the sentence is checked: it must contain the
computed numbers, otherwise a template sentence is used. No tool calling, no loop, no step limit needed.
(Measured with gemma-2-2b-it: asked to COMPARE two numbers it was wrong 5 of 5 tries.)

    python workflow_compare.py --city-a Munich --city-b Berlin --date 2026-10-20
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find llm_config.py in the folder above
from llm_config import PROVIDER, PROVIDERS, get_llm  # noqa: E402
from langgraph.graph import END, START, StateGraph

import react_agent as ra  # reuse the weather tool  # noqa: E402


class CompareState(TypedDict, total=False):
    city_a: str
    city_b: str
    day: str
    forecast_a: dict
    forecast_b: dict
    warmer: str        # city name, or "" when equal
    other: str
    diff_c: float
    facts: str         # the computed result, as text
    answer: str
    used_fallback: bool
    error: str


def fetch_with_tool(city, day):
    """Default fetcher: the get_weather_forecast tool from react_agent.py, parsed from its JSON."""
    return json.loads(ra.get_weather_forecast.invoke({"location": city, "date": day}))


def make_facts(a, b, fa, fb):
    """Code decides: compare the two mean temperatures. Returns (warmer, other, diff, facts_text)."""
    ma, mb = fa["mean_c"], fb["mean_c"]
    diff = round(abs(ma - mb), 1)
    if diff == 0:
        return "", "", 0.0, f"{a} and {b} have the same mean temperature, {ma} C."
    warmer, other = (a, b) if ma > mb else (b, a)
    mw, mo = (ma, mb) if ma > mb else (mb, ma)
    return warmer, other, diff, (
        f"{warmer} is warmer than {other} by {diff} C "
        f"(mean {warmer} {mw} C, mean {other} {mo} C)."
    )


def template_sentence(state):
    return state["facts"]


def answer_ok(text, state):
    """The sentence must name the warmer city and contain the computed difference."""
    if state["diff_c"] == 0:
        return "same" in text.lower() or "equal" in text.lower()
    return state["warmer"].lower() in text.lower() and f"{state['diff_c']}" in text


PHRASE_PROMPT = """Rewrite a weather result as one friendly sentence for a traveller.
Copy the city names and all numbers exactly. The sentence must contain the difference, written like "6.0 C".

Example
Result: Rome is warmer than Oslo by 6.0 C (mean Rome 14.5 C, mean Oslo 8.5 C).
Sentence: Rome is 6.0 C warmer than Oslo, with a mean of 14.5 C against 8.5 C.

Now your turn
Result: {facts}
Sentence:"""


def build_graph(model, fetch=None):
    """Compile the workflow. `fetch(city, day)` returns a dict with mean_c (or error); `model` needs .invoke."""
    fetch = fetch or fetch_with_tool

    def fetch_a(state: CompareState):
        return {"forecast_a": fetch(state["city_a"], state["day"])}

    def fetch_b(state: CompareState):
        return {"forecast_b": fetch(state["city_b"], state["day"])}

    def compare(state: CompareState):
        for key in ("forecast_a", "forecast_b"):
            if "mean_c" not in state[key]:
                return {"error": state[key].get("error", "no data"), "facts": "", "answer": ""}
        warmer, other, diff, facts = make_facts(
            state["city_a"], state["city_b"], state["forecast_a"], state["forecast_b"])
        return {"warmer": warmer, "other": other, "diff_c": diff, "facts": facts}

    def after_compare(state: CompareState):
        return "stop" if state.get("error") else "phrase"

    def phrase(state: CompareState):
        reply = model.invoke(PHRASE_PROMPT.format(facts=state["facts"], diff=state["diff_c"]))
        text = (reply.content if isinstance(reply.content, str) else str(reply.content)).strip()
        if answer_ok(text, state):
            return {"answer": text, "used_fallback": False}
        return {"answer": template_sentence(state), "used_fallback": True}

    def stop(state: CompareState):
        return {"answer": f"Could not compare: {state['error']}", "used_fallback": True}

    workflow = StateGraph(CompareState)
    workflow.add_node("fetch_a", fetch_a)
    workflow.add_node("fetch_b", fetch_b)
    workflow.add_node("compare", compare)
    workflow.add_node("phrase", phrase)
    workflow.add_node("stop", stop)
    workflow.add_edge(START, "fetch_a")
    workflow.add_edge("fetch_a", "fetch_b")
    workflow.add_edge("fetch_b", "compare")
    workflow.add_conditional_edges("compare", after_compare, {"phrase": "phrase", "stop": "stop"})
    workflow.add_edge("phrase", END)
    workflow.add_edge("stop", END)
    return workflow.compile()


def run(city_a, city_b, day, provider=None, model=None):
    graph = build_graph(get_llm(provider, model=model))
    return graph.invoke({"city_a": city_a, "city_b": city_b, "day": day})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider", default=PROVIDER, choices=list(PROVIDERS))
    parser.add_argument("--model", default=None, help="model id; default comes from llm_config.py")
    parser.add_argument("--city-a", default="Munich")
    parser.add_argument("--city-b", default="Berlin")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    out = run(args.city_a, args.city_b, args.date, args.provider, args.model)
    print("facts :", out.get("facts"))
    print("answer:", out["answer"], "(template fallback)" if out.get("used_fallback") else "")


if __name__ == "__main__":
    main()
