"""Offline demo page: ask a document, a fine-tuned Marathi model, and a fine-tuned router. Run from code/demo/:

    streamlit run app.py

LM Studio must be serving Qwen3 1.7B, nomic-embed-text and the Marathi GGUF (see README.md). The page refuses every
connection that is not to this machine and shows the count, so it doubles as the proof that nothing goes online.
"""

import csv
import html
import json
import time
from pathlib import Path

import streamlit as st

import demo_lib as D

D.install_guard()
CODE = D.CODE
QW, FT = CODE / "quantum-wiki", CODE / "finetune-slm"

st.set_page_config(page_title="Offline AI demo", layout="wide")
st.markdown(
    """<style>
html, body, [class*="css"] { font-size: 19px; }
.block-container { padding-top: 1.6rem; max-width: 1500px; }
.badge { display:inline-block; padding:2px 12px; border-radius:12px; font-weight:600; margin-right:8px; }
.ok { background:#d8f0dc; color:#0b5a1c; } .bad { background:#f8d8d8; color:#8a1010; } .live { background:#ffe9b5; color:#6b4a00; }
.rec { background:#e3e6f0; color:#2c3763; } .ans { font-size:1.15em; line-height:1.55; }
.meta { color:#555; font-size:0.85em; }
</style>""",
    unsafe_allow_html=True,
)

qp = st.query_params


def read_csv(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []


# ------------------------------------------------------------------ header and sidebar
status = D.model_status()
server_up = D.loaded_models() is not None
with st.sidebar:
    st.header("This machine only")
    st.metric("Connections to the internet", len(D.GUARD.blocked), help="Any attempt to reach another machine is refused and counted here.")
    st.metric("Connections to this PC (LM Studio)", D.GUARD.local)
    if D.GUARD.blocked:
        st.error("Blocked: " + ", ".join("%s:%s" % b for b in D.GUARD.blocked[-3:]))
    st.caption("This counter covers this page. The Wi-Fi switch shows the same for everything else, LM Studio included.")
    st.divider()
    st.subheader("Models in LM Studio")
    for key, label in (("chat", "Qwen3 1.7B (language)"), ("embedding", "nomic-embed (search)"), ("marathi", "Gemma 3 1B, Marathi tuned")):
        st.markdown('<span class="badge %s">%s</span> %s' % ("ok" if status[key] else "bad", "loaded" if status[key] else "missing", label),
                    unsafe_allow_html=True)
    st.divider()
    hide = st.toggle("Hide Marathi answers (for slides)", value=qp.get("hide", "0") == "1",
                     help="The book's licence is unconfirmed, so slides and screenshots show the questions and the measured table only.")

st.title("Small AI, no internet")
st.caption("Everything below runs on this laptop. The models are 1 to 2 billion parameters, served by LM Studio on this machine.")
if not server_up:
    st.error("LM Studio is not running. Start it, start its server (Developer tab, or `lms server start`) and load the three models.")
    st.stop()

SECTIONS = ["Ask a document", "Fine-tuned Marathi", "Fine-tuned router"]
section = st.radio("Demo", SECTIONS, index=int(qp.get("s", "0")), horizontal=True, label_visibility="collapsed")


def badge(text, kind):
    return '<span class="badge %s">%s</span>' % (kind, text)


# ------------------------------------------------------------------ section 1: ask a document
@st.cache_resource(show_spinner="Loading the syllabus index (cached embeddings, no internet)...")
def load_qwiki():
    from qwiki import cards as C, config
    from qwiki.chunks import naive_chunks, structure_chunks
    from qwiki.embed import lm_studio_embeddings
    from qwiki.llm import Chat
    from qwiki.retrieval import Retrievers

    cards = C.load_cards()
    tables = json.loads((config.PARSED / "tables.json").read_text(encoding="utf-8"))
    table_md = next((t["markdown"] for t in tables if t["page"] == 14), "")
    emb = lm_studio_embeddings(config.PARSED / "embeddings.json")
    struct = Retrievers(structure_chunks(cards, table_md), emb)
    naive = Retrievers(naive_chunks(), emb)
    return cards, struct, naive, Chat()


def run_qwiki(question: str) -> dict:
    from qwiki.answer import answer_agent, answer_rag

    cards, struct, naive, llm = load_qwiki()
    out = {}
    for name in ("naive", "agent"):
        before = llm.snapshot()
        t0 = time.perf_counter()
        res = answer_rag(question, naive, llm, mode="dense") if name == "naive" else answer_agent(question, cards, struct, llm)
        res["seconds"] = time.perf_counter() - t0
        res["calls"] = int(llm.since(before).get("llm_calls", 0))
        out[name] = res
    return out


def section_document():
    heldout = json.loads((QW / "eval" / "heldout.json").read_text(encoding="utf-8"))
    qs = D.qwiki_questions(heldout)
    st.subheader("Ask the AICTE quantum-technologies curriculum (a 39-page PDF)")
    st.caption("Two systems answer the same question with the same 1.7B model. Questions: the first held-out question of each of six kinds, "
               "fixed before any answer was seen.")
    cols = st.columns(3)
    for i, q in enumerate(qs):
        if cols[i % 3].button(q["q"], key="qw%d" % i, use_container_width=True):
            st.session_state["qw_q"] = q["q"]
    typed = st.text_input("Or type your own question")
    if typed and st.button("Ask"):
        st.session_state["qw_q"] = typed
    auto = qp.get("q")
    if auto is not None and "qw_q" not in st.session_state and auto.isdigit() and int(auto) < len(qs):
        st.session_state["qw_q"] = qs[int(auto)]["q"]
    question = st.session_state.get("qw_q")
    if not question:
        return
    st.markdown("### " + question)
    cache = st.session_state.setdefault("qw_cache", {})
    if question not in cache:
        with st.spinner("Both systems are answering..."):
            cache[question] = run_qwiki(question)
    res = cache[question]
    left, right = st.columns(2)
    for col, key, title in ((left, "naive", "Plain PDF search + model"), (right, "agent", "Structure-aware agent")):
        r = res[key]
        with col:
            st.markdown("#### " + title)
            st.markdown('<div class="ans">%s</div>' % html.escape(r["answer"]).replace("\n", "<br>"), unsafe_allow_html=True)
            pages = ", ".join(str(p) for p in r.get("pages", [])) or "none"
            st.markdown('<div class="meta">%s %s pages: %s | model calls: %d | %.1f s</div>' % (
                badge("no model used" if not r.get("used_llm") else "model read the evidence", "ok" if not r.get("used_llm") else "live"),
                "route: " + r.get("route", ""), pages, r["calls"], r["seconds"]), unsafe_allow_html=True)
    rows = {(r["system"], r["set"]): r for r in read_csv(QW / "results" / "answers_summary.csv") if r["kind"] == "all"}
    if rows:
        def a(sys_, set_):
            return rows[(sys_, set_)]["accuracy"] if (sys_, set_) in rows else "not run"
        st.markdown("**Measured on 110 questions (one run, one machine):** plain search + model %s; structure-aware retrieval + model %s; agent %s "
                    "(tuned set) and %s (held-out set). The questions were drafted with Claude's help and the author has not reviewed them."
                    % (a("naive", "dev"), a("struct", "dev"), a("agent", "dev"), a("agent", "heldout")))


# ------------------------------------------------------------------ section 2: fine-tuned Marathi
def section_marathi():
    questions = json.loads((FT / "eval" / "mr_test_questions.json").read_text(encoding="utf-8"))
    qs = D.marathi_questions(questions)
    st.subheader("A Marathi model fine-tuned on one book, against the base model")
    st.caption("Gemma 3 1B, LoRA, trained on a laptop CPU. Questions: the first two of each test group, fixed in advance. "
               "The middle answer is generated now; the other two are from the measured run (the base model is not loaded here).")
    labels = {"reworded": "reworded", "raw_only": "raw text only", "control": "control"}
    cols = st.columns(3)
    for i, q in enumerate(qs):
        if cols[i % 3].button("%s: %s" % (labels[q["group"]], q["q"][:60]), key="mr%d" % i, use_container_width=True):
            st.session_state["mr_i"] = i
    auto = qp.get("m")
    if auto is not None and "mr_i" not in st.session_state and auto.isdigit() and int(auto) < len(qs):
        st.session_state["mr_i"] = int(auto)
    i = st.session_state.get("mr_i")
    if i is None:
        return
    q = qs[i]
    st.markdown("### " + q["q"])
    results = FT / "results"
    base, rag = D.load_recorded(results, "base_closed"), D.load_recorded(results, "base_rag")
    tuned_rec = D.load_recorded(results, "tuned_closed")
    cache = st.session_state.setdefault("mr_cache", {})
    if q["id"] not in cache:
        with st.spinner("The tuned model is answering..."):
            cache[q["id"]] = D.chat(D.MARATHI_MODEL, q["q"], max_tokens=90)
    live = cache[q["id"]]
    keys = (tuned_rec.get(q["id"]) or base.get(q["id"]) or {}).get("keys", [])
    cols = st.columns(3)
    panels = (("Base model, no help", base.get(q["id"]), "rec", "recorded"),
              ("Fine-tuned model", {"answer": live["text"], "seconds": live["seconds"], "new_tokens": live["tokens"]}, "live", "live now"),
              ("Base model + 2 book passages", rag.get(q["id"]), "rec", "recorded"))
    for col, (title, r, kind, tag) in zip(cols, panels):
        with col:
            st.markdown("#### " + title)
            if not r:
                st.info("No recorded answer on this machine.")
                continue
            ok = D.key_hit(r["answer"], keys) if keys else None
            verdict = "" if ok is None else badge("correct" if ok else "not correct", "ok" if ok else "bad")
            if hide:
                st.markdown('<div class="ans meta">(answer hidden for slides)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ans">%s</div>' % html.escape(r["answer"]), unsafe_allow_html=True)
            st.markdown('<div class="meta">%s %s %s | %.1f s | %s tokens | Devanagari %.0f%%</div>' % (
                badge(tag, kind), verdict, "", r.get("seconds", 0), r.get("new_tokens", "?"), 100 * D.devanagari_share(r["answer"])),
                unsafe_allow_html=True)
    rows = [r for r in read_csv(results / "mr_summary.csv") if r["group"] == "all"]
    if rows:
        names = {"base_closed": "Base model, no help", "tuned_closed": "Fine-tuned model", "base_rag": "Base model + 2 book passages"}
        n_all = 52
        st.markdown("**Measured on %d test questions, one run (correct answers):**" % n_all)
        st.table([{"System": names[r["system"]], "Correct of 52": round(float(r["all_keys"]) * n_all), "Share": r["all_keys"],
                   "Seconds per answer": r["seconds"]} for r in rows])
        st.caption("Fine-tuning mostly taught the training questions and made answers shorter and faster. Retrieval answered more than twice as many. "
                   "The questions were drafted with Claude's help and have not yet been reviewed by the author.")


# ------------------------------------------------------------------ section 3: fine-tuned router
@st.cache_resource(show_spinner="Loading the 135M router and its adapter...")
def load_router():
    from ftslm import model as M

    return M, *M.load(adapter=FT / "adapters" / "intent")


def section_router():
    st.subheader("A 135-million-parameter model taught one job: route a helpline message")
    st.caption("LoRA, 3.7 MB adapter, trained in 22 minutes on this laptop's CPU. The answer is always one of six intents.")
    ood = [json.loads(l) for l in (FT / "data" / "test_ood.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seen, picks = set(), []
    for r in ood:                                            # fixed rule: the first test message of each intent
        if r["label"] not in seen:
            seen.add(r["label"])
            picks.append(r)
    cols = st.columns(3)
    for i, r in enumerate(picks):
        if cols[i % 3].button(r["text"][:70], key="rt%d" % i, use_container_width=True):
            st.session_state["rt_i"] = i
    auto = qp.get("r")
    if auto is not None and "rt_i" not in st.session_state and auto.isdigit() and int(auto) < len(picks):
        st.session_state["rt_i"] = int(auto)
    i = st.session_state.get("rt_i")
    if i is not None:
        r = picks[i]
        try:
            M, tok, model = load_router()
            preds, stats = M.classify(tok, model, [M.short_prompt(r["text"])])
        except Exception as e:                                # not cached offline, or torch missing
            st.error("The router could not load: %s" % e)
            return
        ok = preds[0] == r["label"]
        st.markdown("### " + r["text"])
        st.markdown('<div class="ans">Intent: <b>%s</b> %s %s</div><div class="meta">%.2f s on the CPU</div>' % (
            preds[0], badge("correct" if ok else "not correct", "ok" if ok else "bad"), badge("live", "live"), stats["seconds_per_item"]),
            unsafe_allow_html=True)
    rows = read_csv(FT / "results" / "results.csv")
    if rows:
        names = {"base_instruction": "Base model, instructions only", "base_few_shot": "Base model, instructions + examples", "tuned": "Fine-tuned"}
        table = {}
        for r in rows:
            if r["system"] in names:
                table.setdefault(names[r["system"]], {"System": names[r["system"]]})[r["split"]] = r["accuracy"]
        st.markdown("**Measured on 150 + 150 new messages (made-up data, a toy task; same-wording and new-wording tests):**")
        st.table([{"System": v["System"], "Same wording": v.get("test_iid"), "New wording": v.get("test_ood")} for v in table.values()])


{"Ask a document": section_document, "Fine-tuned Marathi": section_marathi, "Fine-tuned router": section_router}[section]()
