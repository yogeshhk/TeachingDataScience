"""Offline tests for the demo helpers: no LM Studio, no network."""

import json
import socket
import threading

import pytest

import demo_lib as D


def test_pages_refuse_to_run_unless_bound_to_this_machine():
    assert D.bind_problem("localhost") == "" and D.bind_problem("127.0.0.1") == ""
    assert D.bind_problem(None) != "" and D.bind_problem("") != ""
    assert D.bind_problem("0.0.0.0") != ""


def test_question_rules_are_fixed_and_ordered():
    heldout = [{"id": "a", "kind": "cross", "q": "q1"}, {"id": "b", "kind": "status", "q": "q2"},
               {"id": "c", "kind": "status", "q": "q3"}, {"id": "d", "kind": "unanswerable", "q": "q4"},
               {"id": "e", "kind": "lookup", "q": "q5"}]
    got = D.qwiki_questions(heldout)
    assert [q["id"] for q in got] == ["b", "a", "d"]                 # first of each kind, in the fixed kind order, lookup left out


def test_demo_questions_are_the_five_chosen():
    assert len(D.DEMO_QUESTIONS) == 5 and len(set(D.DEMO_QUESTIONS)) == 5
    assert all(q.endswith("?") for q in D.DEMO_QUESTIONS)


def test_rag_score_sums_the_structure_rows_only():
    rows = [{"system": "struct", "kind": "all", "n": "80", "accuracy": "0.688"},
            {"system": "struct", "kind": "all", "n": "30", "accuracy": "0.6"},
            {"system": "agent", "kind": "all", "n": "80", "accuracy": "0.975"},
            {"system": "struct", "kind": "status", "n": "8", "accuracy": "1.0"}]
    s = D.rag_score(rows)
    assert "66%" in s and "110" in s
    assert D.rag_score([{"system": "agent", "kind": "all", "n": "80", "accuracy": "0.975"}]) is None


def test_faq_score_line_is_plain_and_has_no_dash():
    assert "20" in D.FAQ_SCORE and "91%" in D.FAQ_SCORE
    assert chr(0x2014) not in D.FAQ_SCORE and chr(0x2013) not in D.FAQ_SCORE   # no em dash, no en dash


def test_clean_answer_drops_bold_and_extra_space():
    assert D.clean_answer("The title is **Quantum**.  \n[p. 30]") == "The title is Quantum. [p. 30]"


def test_marathi_rule_takes_first_two_per_group():
    tests = [{"id": "r%d" % i, "group": "reworded"} for i in range(4)] + [{"id": "c1", "group": "control"}]
    got = D.marathi_questions(tests)
    assert [q["id"] for q in got] == ["r0", "r1", "c1"]


def test_key_hit_matches_the_study_rule():
    assert D.key_hit("अन्न आणि कपडे आणि निवारा", ["अन्न", "वस्त्र|कपडे", "निवारा"])
    assert not D.key_hit("अन्न आणि निवारा", ["अन्न", "वस्त्र|कपडे", "निवारा"])
    assert not D.key_hit("anything", [])


def test_devanagari_share():
    assert D.devanagari_share("नमस्कार") == 1.0
    assert D.devanagari_share("abc") == 0.0
    assert D.devanagari_share("") == 0.0


def test_load_recorded_missing_file_is_empty(tmp_path):
    assert D.load_recorded(tmp_path, "base_closed") == {}


def test_load_recorded_reads_rows(tmp_path):
    (tmp_path / "mr_answers_base_closed.jsonl").write_text(
        json.dumps({"id": "e1_01", "answer": "x"}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert D.load_recorded(tmp_path, "base_closed")["e1_01"]["answer"] == "x"


def test_model_status_all_false_when_server_down(monkeypatch):
    monkeypatch.setattr(D, "loaded_models", lambda: None)
    assert D.model_status() == {"chat": False, "embedding": False, "marathi": False}


def test_guard_blocks_other_hosts_and_counts_local():
    g = D.Guard()
    original = (socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto, socket.getaddrinfo)
    try:
        D.install_guard(g)
        s = socket.socket()
        with pytest.raises(D.Blocked):
            s.connect(("93.184.216.34", 80))                      # an internet address: refused before any packet is sent
        assert s.connect_ex(("93.184.216.34", 80)) == 111
        assert g.blocked and g.blocked[0][0] == "93.184.216.34"
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("example.org", 80)                 # a name lookup for another host is refused
        assert ("example.org", None) in g.blocked
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with pytest.raises(D.Blocked):
            udp.sendto(b"x", ("93.184.216.34", 53))               # so is a UDP datagram
        udp.close()
        assert socket.getaddrinfo("localhost", 80)                # this machine still resolves
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        threading.Thread(target=server.accept, daemon=True).start()
        c = socket.socket()
        c.connect(("127.0.0.1", server.getsockname()[1]))
        assert g.local == 1
        c.close()
        server.close()
        s.close()
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto, socket.getaddrinfo = original
