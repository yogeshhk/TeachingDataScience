"""Helpers for the offline demo page: the connection guard, LM Studio calls, the fixed question lists and the recorded Marathi answers.

Nothing here needs Streamlit, so the tests run without it. Everything is local: the guard refuses any TCP or UDP
connection and any name lookup, made from this Python process, that is not to this machine, and counts both kinds.
"""

import json
import os
import re
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
# The demo lives in the TeachingDataScience repo; the projects it shows live in the FrugalAI repo. CODE is that repo's
# code/ folder: set FRUGALAI_REPO to the FrugalAI clone, or leave it unset when both repos sit side by side.
FRUGALAI_REPO = Path(os.environ.get("FRUGALAI_REPO") or HERE.parents[3] / "FrugalAI")
CODE = FRUGALAI_REPO / "code"
if not CODE.is_dir():
    raise SystemExit("FrugalAI code folder not found at %s. Set FRUGALAI_REPO to the FrugalAI clone." % CODE)
for sub in ("common", "quantum-wiki", "finetune-slm", "workflow-agent"):          # the FrugalAI projects the demo shows
    p = str(CODE / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")                    # no Hugging Face lookups, ever
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

LLM_URL = os.environ.get("DEMO_LLM_URL", "http://localhost:1234/v1")
CHAT_MODEL = os.environ.get("DEMO_CHAT_MODEL", "qwen/qwen3-1.7b")
EMBED_MODEL = os.environ.get("DEMO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
MARATHI_MODEL = os.environ.get("DEMO_MARATHI_MODEL", "marathi-1b")
NEEDED = {"chat": CHAT_MODEL, "embedding": EMBED_MODEL, "marathi": MARATHI_MODEL}

_LOCAL = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


# ------------------------------------------------------------------ offline guard
class Blocked(ConnectionError):
    pass


class Guard:
    def __init__(self):
        self.blocked: List[tuple] = []       # attempts to reach another machine (stopped)
        self.local = 0                       # connections to this machine (LM Studio, this page)
        self.active = False


GUARD = Guard()
_lock = threading.Lock()


def _is_local(address) -> bool:
    host = str(address[0]) if isinstance(address, tuple) and address else str(address)
    return host in _LOCAL or host.startswith("127.")


def _is_ip_literal(host: str) -> bool:
    import ipaddress

    try:
        ipaddress.ip_address(str(host).split("%")[0])
        return True
    except ValueError:
        return False


def bind_problem(address) -> str:
    """Empty string when the Streamlit server is bound to this machine only, else the message to show."""
    return "" if str(address or "") in ("localhost", "127.0.0.1", "::1") else (
        "This page must be served on localhost only. Start it from this folder (streamlit run app.py) "
        "or pass --server.address localhost.")


def install_guard(guard: Guard = GUARD) -> Guard:
    """Refuse every connect() to another machine for the rest of the process. Safe to call twice."""
    with _lock:
        if guard.active:
            return guard
        original, original_ex = socket.socket.connect, socket.socket.connect_ex

        def check(address) -> bool:
            if _is_local(address):
                guard.local += 1
                return True
            guard.blocked.append((str(address[0]) if isinstance(address, tuple) else str(address),
                                  address[1] if isinstance(address, tuple) and len(address) > 1 else None))
            return False

        def connect(sock, address):
            if not check(address):
                raise Blocked("offline guard: blocked a connection to %s" % (address,))
            return original(sock, address)

        def connect_ex(sock, address):
            return original_ex(sock, address) if check(address) else 111

        original_sendto, original_getaddrinfo = socket.socket.sendto, socket.getaddrinfo

        def sendto(sock, data, *args):
            address = args[-1] if args else None   # sendto(data, address) or sendto(data, flags, address)
            if address is not None and not check(address):
                raise Blocked("offline guard: blocked a datagram to %s" % (address,))
            return original_sendto(sock, data, *args)

        def getaddrinfo(host, *args, **kwargs):
            name = host.decode("ascii", "replace") if isinstance(host, bytes) else host
            if name and not _is_ip_literal(name) and not _is_local(name):   # a name lookup; literals are checked at connect
                guard.blocked.append((str(name), None))
                raise socket.gaierror("offline guard: blocked a name lookup for %s" % name)
            return original_getaddrinfo(host, *args, **kwargs)

        socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
        socket.socket.sendto, socket.getaddrinfo = sendto, getaddrinfo
        guard.active = True
    return guard


# ------------------------------------------------------------------ LM Studio
def _get(path: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(LLM_URL.rstrip("/") + path, timeout=timeout) as r:
        return json.load(r)


def loaded_models() -> Optional[List[str]]:
    """Model ids the server reports, or None when LM Studio is not running."""
    try:
        return [m["id"] for m in _get("/models").get("data", [])]
    except Exception:
        return None


def model_status() -> Dict[str, bool]:
    """For each model the demo needs: is it available on the server? (All False when the server is down.)"""
    ids = loaded_models() or []
    return {k: v in ids for k, v in NEEDED.items()}


def chat(model: str, user: str, max_tokens: int = 120, system: Optional[str] = None) -> dict:
    """One chat completion from LM Studio. Returns text, seconds and token counts."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    body = json.dumps({"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(LLM_URL.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return {"text": r["choices"][0]["message"]["content"].strip(), "seconds": time.perf_counter() - t0,
            "tokens": r.get("usage", {}).get("completion_tokens", 0)}


# ------------------------------------------------------------------ fixed question lists
# Rule, fixed before any answer was seen: the FIRST held-out question of each of six kinds, in file order.
QWIKI_KINDS = ("status", "list", "subtopics", "numbers", "cross", "unanswerable")
# Marathi rule: the FIRST TWO test questions of each group, in file order.
MARATHI_GROUPS = ("reworded", "raw_only", "control")


def qwiki_questions(heldout: List[dict]) -> List[dict]:
    seen, out = set(), []
    for q in heldout:
        if q["kind"] in QWIKI_KINDS and q["kind"] not in seen:
            seen.add(q["kind"])
            out.append(q)
    return sorted(out, key=lambda q: QWIKI_KINDS.index(q["kind"]))


def marathi_questions(tests: List[dict], per_group: int = 2) -> List[dict]:
    out = []
    for g in MARATHI_GROUPS:
        out += [q for q in tests if q["group"] == g][:per_group]
    return out


# ------------------------------------------------------------------ the minimal demo: base model against base + RAG
# Picked by the author on 2026-10-28 after a trial run (ANALYSIS.md section 8), as clear illustrations, not a random sample.
DEMO_QUESTIONS = (
    "For how many lectures should sensing be taught within QT 01?",
    "What is the pass mark for QT 03?",
    "What is the title of QT 08?",
    "Which course is called Quantum Optics?",
    "What are the credits of QT 04?",
)


def rag_score(rows: List[dict]) -> Optional[str]:
    """One plain sentence from the rows of answers_summary.csv, or None when there is no structure-aware run."""
    mine = [r for r in rows if r.get("system") == "struct" and r.get("kind") == "all"]
    n = sum(int(r["n"]) for r in mine)
    if not n:
        return None
    right = sum(round(float(r["accuracy"]) * int(r["n"])) for r in mine)
    return ("Base + RAG answered %d%% of %d syllabus questions correctly (one run, one machine). "
            "These five questions were picked as clear examples." % (round(100 * right / n), n))


# From code/workflow-agent/README.md (author's private FAQ notes, 32 test cases written by Claude, one run).
FAQ_SCORE = ("On 32 test questions about the author's own FAQ notes, the right entry came first for all 20 reworded ones, "
             "and covered against new was right 91% of the time (one run; the cut-off was tuned on half the cases).")


def clean_answer(text: str) -> str:
    """Plain text for the page: no markdown bold, one space between words, line breaks kept as spaces."""
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


# ------------------------------------------------------------------ Marathi: recorded answers and scoring
def load_recorded(results_dir: Path, system: str) -> Dict[str, dict]:
    """Answers of one system from the measured run, keyed by question id. Empty when the local file is absent."""
    path = results_dir / ("mr_answers_%s.jsonl" % system)
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def key_hit(answer: str, keys: List[str]) -> bool:
    """The mechanical rule of the study: every key phrase (alternatives split by |) occurs in the answer."""
    return all(any(alt in answer for alt in k.split("|")) for k in keys) if keys else False


def devanagari_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if "ऀ" <= c <= "ॿ") / len(letters) if letters else 0.0


def short(text: str, n: int = 160) -> str:
    return re.sub(r"\s+", " ", text).strip()[:n]
