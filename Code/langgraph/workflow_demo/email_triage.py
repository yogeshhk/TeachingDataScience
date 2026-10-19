"""Email triage as a LangGraph workflow: several nodes, one conditional edge.

START -> read_email -> classify_email -> (spam?) -> handle_spam  -> END
                                                 -> draft_reply  -> END

read_email and handle_spam are plain Python functions; classify_email and draft_reply call an LLM.
The model comes from ../llm_config.py (default: a small local model in LM Studio, no key):

    python email_triage.py
    python email_triage.py --email "URGENT! You won 1M dollars. Click now!!!"
    python email_triage.py --provider groq
"""
import argparse
import operator
import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find llm_config.py in the folder above
from llm_config import PROVIDER, PROVIDERS, get_llm  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

MAX_WORDS = 200   # the classifier sees at most this many words: a short, cheap prompt

SAMPLE_EMAILS = {
    "legit": "Hello, could we schedule a meeting on Thursday to discuss the project timeline?",
    "spam": "URGENT! You have won 1,000,000 dollars. Click here now!!!",
}


# ---------------------------------------------------------------- state
class EmailState(BaseModel):
    """The context object that flows through every node."""
    email: str
    words_list: list[str] = Field(default_factory=list)
    is_spam: bool = False
    draft: str = ""
    # reducer: each node adds its own line; operator.add concatenates the lists
    log: Annotated[list[str], operator.add] = Field(default_factory=list)


# ---------------------------------------------------------------- plain-function nodes
def read_email(state: EmailState):
    """No LLM: plain code that splits the email into words, keeping the first MAX_WORDS."""
    return {"words_list": state.email.split()[:MAX_WORDS], "log": ["read: email received"]}


def handle_spam(state: EmailState):
    """No LLM: spam goes to junk."""
    return {"log": ["spam: moved to junk"]}


# ---------------------------------------------------------------- routing (the conditional edge)
def route_email(state: EmailState):
    return "spam" if state.is_spam else "legit"


# ---------------------------------------------------------------- graph
def build_graph(llm):
    """Compile the workflow for any chat model with .invoke(prompt) returning an object with .content."""

    def classify_email(state: EmailState):
        """LLM node: the model judges the words read_email kept; the code parses its one-word answer."""
        text = " ".join(state.words_list)
        prompt = (f"Classify this email as spam or legitimate:\n{text}\n"
                  "Answer with one word: spam or legitimate.")
        answer = llm.invoke(prompt).content.strip().lower()
        # startswith, because "not spam" also contains the word "spam"
        return {"is_spam": answer.startswith("spam"), "log": [f"classify: {answer}"]}

    def draft_reply(state: EmailState):
        """LLM node: write a short professional reply."""
        reply = llm.invoke(f"Draft a short, professional reply to this email:\n{state.email}").content
        return {"draft": reply, "log": ["draft: reply written"]}

    workflow = StateGraph(EmailState)
    workflow.add_node("read_email", read_email)
    workflow.add_node("classify_email", classify_email)
    workflow.add_node("handle_spam", handle_spam)
    workflow.add_node("draft_reply", draft_reply)

    workflow.add_edge(START, "read_email")
    workflow.add_edge("read_email", "classify_email")
    workflow.add_conditional_edges(
        "classify_email", route_email,
        {"spam": "handle_spam", "legit": "draft_reply"})   # label -> next node
    workflow.add_edge("handle_spam", END)
    workflow.add_edge("draft_reply", END)
    return workflow.compile()


# ---------------------------------------------------------------- run
def run(email, provider=None, model=None):
    graph = build_graph(get_llm(provider, model=model))
    for update in graph.stream(EmailState(email=email), stream_mode="updates"):
        print(update)                       # {node name: what that node returned}
    return graph.invoke(EmailState(email=email))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider", default=PROVIDER, choices=list(PROVIDERS))
    parser.add_argument("--model", default=None, help="model id; default comes from llm_config.py")
    parser.add_argument("--email", default=SAMPLE_EMAILS["legit"])
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    result = run(args.email, args.provider, args.model)
    print("\nis_spam:", result["is_spam"], "| words:", len(result["words_list"]))
    print("log:", result["log"])
    if result["draft"]:
        print("draft:", result["draft"])


if __name__ == "__main__":
    main()
