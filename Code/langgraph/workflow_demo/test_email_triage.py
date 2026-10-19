"""Tests for email_triage.py. No network, no API keys: the model is a scripted fake."""
from types import SimpleNamespace

import email_triage as et


class FakeLLM:
    """Replays a list of replies, one per invoke, and records the prompts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.replies[len(self.prompts) - 1])


def run(email, replies):
    llm = FakeLLM(replies)
    return et.build_graph(llm).invoke(et.EmailState(email=email)), llm


def test_legitimate_email_gets_a_draft():
    result, llm = run(et.SAMPLE_EMAILS["legit"], ["legitimate", "Thursday works for me."])
    assert result["is_spam"] is False
    assert result["draft"] == "Thursday works for me."
    assert len(llm.prompts) == 2                      # classify, then draft
    assert result["log"] == ["read: email received", "classify: legitimate", "draft: reply written"]


def test_spam_goes_to_junk_without_a_second_llm_call():
    result, llm = run(et.SAMPLE_EMAILS["spam"], ["spam"])
    assert result["is_spam"] is True
    assert result["draft"] == ""
    assert len(llm.prompts) == 1                      # only classify; handle_spam is plain code
    assert result["log"][-1] == "spam: moved to junk"


def test_not_spam_is_not_treated_as_spam():
    result, _ = run("Hi, quick question.", ["not spam, this is a normal email", "Sure."])
    assert result["is_spam"] is False


def test_plain_node_splits_the_email_into_words():
    result, _ = run("one two three", ["legitimate", "ok"])
    assert result["words_list"] == ["one", "two", "three"]


def test_classifier_reads_the_words_kept_by_read_email():
    long_email = " ".join(f"w{i}" for i in range(et.MAX_WORDS + 50))
    result, llm = run(long_email, ["legitimate", "ok"])
    assert len(result["words_list"]) == et.MAX_WORDS
    assert f"w{et.MAX_WORDS - 1}" in llm.prompts[0]         # last kept word is in the classify prompt
    assert f"w{et.MAX_WORDS}" not in llm.prompts[0]         # a dropped word is not


def test_reducer_appends_log_lines_in_order():
    result, _ = run(et.SAMPLE_EMAILS["spam"], ["spam"])
    assert [line.split(":")[0] for line in result["log"]] == ["read", "classify", "spam"]


def test_stream_updates_are_keyed_by_node_name():
    graph = et.build_graph(FakeLLM(["spam"]))
    nodes = [next(iter(chunk)) for chunk in graph.stream(et.EmailState(email="x"), stream_mode="updates")]
    assert nodes == ["read_email", "classify_email", "handle_spam"]


def test_graph_has_one_conditional_edge_with_two_targets():
    drawn = et.build_graph(FakeLLM(["spam"])).get_graph()
    conditional = [e for e in drawn.edges if e.conditional]
    assert {e.target for e in conditional} == {"handle_spam", "draft_reply"}
