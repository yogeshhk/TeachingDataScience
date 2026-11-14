"""Offline demo, second page: is a concept already in my FAQ notes, and where else does it appear? Run from Code/frugal/demo/:

    streamlit run app_faq.py

Uses the author's own QCNP folder (FAQ entries and the concept_check.py counter), so it is for live use only: slides and
screenshots must not show its FAQ titles or file names. LM Studio must serve nomic-embed-text; no chat model is needed,
because drafting is switched off. Nothing is written anywhere.
"""

import html
import os

import streamlit as st

import demo_lib as D

D.install_guard()
os.environ.setdefault("FAQINTAKE_QCNP", str(D.CODE.parents[1] / "Publications" / "Books" / "QCNP"))
os.environ.setdefault("FAQINTAKE_RUN_CONCEPT_CHECK", "1")   # the author's own concept_check.py in that folder, opted in here

st.set_page_config(page_title="Is it already covered?", layout="wide")
_bad = D.bind_problem(st.get_option("server.address"))
if _bad:
    st.error(_bad)
    st.stop()
st.markdown(
    """<style>
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 1100px; }
html, body, [class*="css"] { font-size: 22px; }
.v { font-size: 2em; font-weight: 700; margin: 1.2rem 0 0.2rem 0; }
.near { color: #555; margin-bottom: 1.6rem; }
.h { font-size: 1.05em; font-weight: 600; color: #555; margin: 1.2rem 0 0.4rem 0; }
.row { font-size: 1.1em; line-height: 1.7; }
.note { color: #666; font-size: 0.8em; margin-top: 2.4rem; }
</style>""",
    unsafe_allow_html=True,
)

if D.loaded_models() is None:
    st.error("LM Studio is not running. Start it and its server, then load nomic-embed-text.")
    st.stop()


@st.cache_resource(show_spinner="Reading the FAQ notes...")
def load_context():
    from faqintake import checks, config, corpus
    from faqintake.embed import lm_studio_embeddings
    from faqintake.graph import Context
    from faqintake.llm import FakeChat

    entries = corpus.load_entries()
    index = checks.SemanticIndex(entries, lm_studio_embeddings())
    return Context(entries=entries, index=index, llm=FakeChat([]), cc=checks.load_concept_check(),
                   thresholds=checks.load_thresholds(), draft=False), config.USING_PRIVATE


ctx, private = load_context()
if not private:
    st.warning("The QCNP folder was not found, so this is running on the invented sample entries.")

preset = st.query_params.get("q", "")
bare = st.query_params.get("bare") == "1"                # bare=1 is for slides: the question as a heading, larger text, no input box
if bare:
    st.markdown("<style>.block-container { max-width: 1180px; } html, body, [class*=\"css\"] { font-size: 25px; }"
                " .q { font-size: 1.7em; font-weight: 600; margin: 0.6rem 0 0 0; }</style>", unsafe_allow_html=True)
    text = preset
    if text:
        st.markdown('<div class="q">%s</div>' % html.escape(text), unsafe_allow_html=True)
else:
    text = st.text_input("A concept, in your own words", value=preset, placeholder="for example: why does measuring a qubit change it?")
if text.strip():
    from faqintake.graph import run_intake

    with st.spinner("Checking..."):
        out = run_intake(ctx, text.strip())
    label = {"covered": "Already covered", "partial": "Partly covered", "new": "New concept"}[out["decision"]]
    st.markdown('<div class="v">%s</div>' % label, unsafe_allow_html=True)
    key, sim = out["nearest"][0] if out.get("nearest") else ("", 0.0)
    st.markdown('<div class="near">Closest FAQ entry: %s (similarity %.2f)</div>' % (html.escape(key), sim), unsafe_allow_html=True)
    st.markdown('<div class="h">Where it already appears</div>', unsafe_allow_html=True)
    from faqintake.graph import ORDER

    hits = out.get("hits") or {}
    for place, label in ORDER[1:]:                       # the five places after the FAQ entry
        h = hits.get(place)
        # h["files"] holds only the top five files, so the exact figure to show is the total number of mentions
        said = "not available" if h is None else ("not found" if not h["total"] else "found, %d mentions" % h["total"])
        st.markdown('<div class="row">%s: %s</div>' % (html.escape(label), said), unsafe_allow_html=True)
    note = D.FAQ_SCORE + (" These five questions were picked as examples." if bare else "")
    st.markdown('<div class="note">%s</div>' % html.escape(note), unsafe_allow_html=True)
