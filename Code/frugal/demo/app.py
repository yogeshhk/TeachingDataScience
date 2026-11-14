"""Offline demo: the same small model with and without retrieval (RAG). Run from Code/frugal/demo/:

    streamlit run app.py

LM Studio must be serving Qwen3 1.7B and nomic-embed-text (see README.md). The page refuses every connection that is not
to this machine, silently; the Wi-Fi switch is the proof. The earlier three-section page is in _backup/.
"""

import csv
import html
import json

import streamlit as st

import demo_lib as D

D.install_guard()
QW = D.CODE / "quantum-wiki"

st.set_page_config(page_title="Base model and RAG", layout="wide")
_bad = D.bind_problem(st.get_option("server.address"))
if _bad:
    st.error(_bad)
    st.stop()
st.markdown(
    """<style>
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 1300px; }
html, body, [class*="css"] { font-size: 22px; }
.q { font-size: 1.7em; font-weight: 600; margin: 1.2rem 0 1.4rem 0; }
.h { font-size: 1.05em; font-weight: 600; color: #555; margin-bottom: 0.4rem; }
.a { font-size: 1.25em; line-height: 1.5; }
.note { color: #666; font-size: 0.8em; margin-top: 2.2rem; }
</style>""",
    unsafe_allow_html=True,
)

if D.loaded_models() is None:
    st.error("LM Studio is not running. Start it and its server, then load Qwen3 1.7B and nomic-embed-text.")
    st.stop()


@st.cache_resource(show_spinner="Loading the syllabus index...")
def load_rag():
    from qwiki import cards as C, config
    from qwiki.chunks import structure_chunks
    from qwiki.embed import lm_studio_embeddings
    from qwiki.llm import Chat
    from qwiki.retrieval import Retrievers

    cards = C.load_cards()
    tables = json.loads((config.PARSED / "tables.json").read_text(encoding="utf-8"))
    table_md = next((t["markdown"] for t in tables if t["page"] == 14), "")
    emb = lm_studio_embeddings(config.PARSED / "embeddings.json")
    return Retrievers(structure_chunks(cards, table_md), emb), Chat()


def ask(question: str) -> dict:
    from qwiki.answer import answer_rag

    retrievers, llm = load_rag()
    return {"base": llm("Answer the question briefly.", question),
            "rag": answer_rag(question, retrievers, llm, mode="dense")["answer"]}


questions = D.DEMO_QUESTIONS
auto = st.query_params.get("q", "")
if auto.isdigit() and int(auto) < len(questions) and "q" not in st.session_state:
    st.session_state["q"] = questions[int(auto)]

if st.query_params.get("bare") != "1":
    cols = st.columns(len(questions))
    for col, text in zip(cols, questions):
        if col.button(text, key=text, use_container_width=True):
            st.session_state["q"] = text
else:                                                  # bare=1 is for slides: no buttons, a narrower page and larger text
    st.markdown("<style>.block-container { max-width: 1180px; } html, body, [class*=\"css\"] { font-size: 25px; }</style>",
                unsafe_allow_html=True)

question = st.session_state.get("q")
if question:
    st.markdown('<div class="q">%s</div>' % html.escape(question), unsafe_allow_html=True)
    cache = st.session_state.setdefault("cache", {})
    if question not in cache:
        with st.spinner("Both are answering..."):
            cache[question] = ask(question)
    res = cache[question]
    left, right = st.columns(2)
    for col, title, key in ((left, "Base model", "base"), (right, "Base model + RAG", "rag")):
        col.markdown('<div class="h">%s</div><div class="a">%s</div>' % (title, html.escape(D.clean_answer(res[key]))),
                     unsafe_allow_html=True)
    path = QW / "results" / "answers_summary.csv"
    score = D.rag_score(list(csv.DictReader(path.open(encoding="utf-8")))) if path.exists() else None
    if score:
        st.markdown('<div class="note">%s</div>' % html.escape(score), unsafe_allow_html=True)
