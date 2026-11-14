"""Draw the FAQ-intake workflow and the path one new concept took, for the slides (offline, matplotlib only).

    python make_graph_figures.py <out_dir> ["a new concept question"] [--reuse]

Two PNGs: frugal_faq_graph.png (the workflow, with its boxes and arrows read from the compiled LangGraph, so it cannot
drift from the code) and frugal_faq_path.png (the same graph with the boxes of one real run highlighted, from that run's
trace). The draft sub-graph (pick an example, model writes, rules check, repair loop) is one box here, as in the top-level
graph. The run uses the author's QCNP notes with drafting on, so LM Studio must serve nomic-embed-text and Qwen3 1.7B;
--reuse redraws from the saved run. The pictures show box names and one similarity score, never FAQ titles.
"""

import json
import os
import sys
from pathlib import Path

import demo_lib as D                                    # puts the sibling projects on sys.path

os.environ.setdefault("FAQINTAKE_QCNP", str(D.CODE.parents[1] / "Publications" / "Books" / "QCNP"))
os.environ.setdefault("FAQINTAKE_RUN_CONCEPT_CHECK", "1")   # the author's own concept_check.py in that folder, opted in here

LABELS = {
    "normalise": "Clean the question",
    "regex_check": "Count mentions in\nbook, decks and notes",
    "semantic_check": "Find the closest\nFAQ entry",
    "theme_check": "Guess the theme",
    "decide": "Covered, partly\ncovered or new?",
    "covered": "Covered:\npoint to the entry",
    "partial": "Partly covered:\nsuggest an addendum",
    "draft_flow": "New concept only:\nmodel writes a draft,\nrules check, repair loop",
    "plan": "List files to update",
    "render": "Write the report",
}
RANK = {"normalise": 1, "regex_check": 2, "semantic_check": 2, "theme_check": 2, "decide": 3,
        "covered": 4, "partial": 4, "draft_flow": 4, "plan": 5, "render": 6}
SLOT = {"regex_check": -3.75, "semantic_check": 0.0, "theme_check": 3.75, "covered": -3.75, "partial": 0.0, "draft_flow": 3.75}
CODE_FILL, MODEL_FILL, HIT_FILL = "#eceff4", "#ffd9a8", "#bfdcff"
CHOICE = {("decide", "covered"): "covered", ("decide", "partial"): "partly", ("decide", "draft_flow"): "new"}


def structure():
    """The top-level edges of the compiled graph as (source, target)."""
    from faqintake import checks, config, corpus
    from faqintake.embed import HashEmbeddings
    from faqintake.graph import Context, build_graph
    from faqintake.llm import FakeChat

    entries = corpus.load_entries(config.SAMPLE_FAQS)                  # the shape does not depend on the data
    g = build_graph(Context(entries=entries, index=checks.SemanticIndex(entries, HashEmbeddings()), llm=FakeChat([]))).get_graph()
    return [(e.source, e.target) for e in g.edges if not e.source.startswith("__") and not e.target.startswith("__")]


def draw(edges, png: Path, visited=None, draft_label=None, decision_note=""):
    """Boxes at fixed places, arrows from the graph's edges. With `visited` (a set of box ids) the run is highlighted with
    step numbers (the three checks share step 2: they run in parallel) and the rest is faded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.6, 8.0))
    ax.set_xlim(-5.75, 5.75)
    ax.set_ylim(-8.0, 0.9)
    ax.axis("off")
    pos = {n: (SLOT.get(n, 0.0), -(RANK[n] - 1) * 1.5) for n in LABELS}
    box = {}
    for n, text in LABELS.items():
        on = visited is None or n in visited
        if n == "draft_flow" and draft_label and on:
            text = draft_label
        if n == "decide" and decision_note and on:
            text += "\n" + decision_note
        if visited is not None and n in visited:
            text = "%d   %s" % (RANK[n], text.replace("\n", "\n    ", 0))
        fill = MODEL_FILL if n == "draft_flow" else CODE_FILL
        if visited is not None:
            fill = (MODEL_FILL if n == "draft_flow" else HIT_FILL) if n in visited else "#ffffff"
        x, y = pos[n]
        w, h = 3.5, 0.36 * (text.count("\n") + 1) + 0.3
        box[n] = (x, y, w, h)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.16",
                                    fc=fill, ec="#2b2b2b" if on else "#c9ced6", lw=1.8, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=15, color="#1a1a1a" if on else "#b4bac4", zorder=3, linespacing=1.25)
    for s, t in edges:
        (xs, ys, ws, hs), (xt, yt, wt, ht) = box[s], box[t]
        on = visited is None or (s in visited and t in visited)
        color = "#2b2b2b" if visited is None else ("#1f4e9c" if on else "#d2d6dd")
        ax.add_patch(FancyArrowPatch((xs, ys - hs / 2), (xt, yt + ht / 2), arrowstyle="-|>", mutation_scale=22,
                                     lw=3.2 if (visited is not None and on) else 1.8, color=color, shrinkA=1, shrinkB=1, zorder=1))
        if (s, t) in CHOICE:
            ax.text(xs + (xt - xs) * 0.62, ys - hs / 2 + (yt + ht / 2 - ys + hs / 2) * 0.62, CHOICE[(s, t)], fontsize=14, color=color,
                    ha="center", va="center", zorder=4, bbox=dict(fc="white", ec="none", pad=1.5))
    fig.savefig(png, dpi=140, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def real_run(question: str):
    """One real intake run with drafting on: (boxes visited, model attempts, rules ok at the end, decision, similarity, trace)."""
    from faqintake import checks, corpus
    from faqintake.embed import lm_studio_embeddings
    from faqintake.graph import Context, run_intake
    from faqintake.llm import Chat

    entries = corpus.load_entries()
    ctx = Context(entries=entries, index=checks.SemanticIndex(entries, lm_studio_embeddings()), llm=Chat(),
                  cc=checks.load_concept_check(), thresholds=checks.load_thresholds(), draft=True)
    out = run_intake(ctx, question)
    steps = [line.split(":")[0].strip() for line in out["trace"]]
    visited = {s for s in steps if s in LABELS} | {"plan", "render"}
    if {"pick_exemplar", "draft", "lint"} & set(steps):
        visited.add("draft_flow")
    attempts = steps.count("draft")
    lint_lines = [line for line in out["trace"] if line.startswith("lint:")]
    ok = bool(lint_lines) and lint_lines[-1].startswith("lint: ok")
    top = out["nearest"][0][1] if out.get("nearest") else 0.0
    return sorted(visited), attempts, ok, out["decision"], top, out["trace"]


def main(argv):
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    rest = [a for a in argv[1:] if not a.startswith("--")]
    question = rest[0] if rest else "What is a surface code?"
    edges = structure()
    draw(edges, out / "frugal_faq_graph.png")
    saved = out / "path_run.json"                                      # the real run is kept so a redraw does not repeat it
    if "--reuse" in argv and saved.exists():
        run = json.loads(saved.read_text(encoding="utf-8"))
    else:
        run = list(real_run(question))
        saved.write_text(json.dumps(run), encoding="utf-8")
    visited, attempts, ok, decision, top, trace = run
    label = ("Model drafts once" if attempts == 1 else "Model drafts %d times" % attempts) + ("" if ok else ",\nrules failed each time")
    draw(edges, out / "frugal_faq_path.png", set(visited), draft_label=label, decision_note="%s (%.2f)" % (decision, top))
    for line in trace:
        print("trace:", "".join(c if 32 <= ord(c) < 127 else "?" for c in line)[:110])
    print("visited:", visited, "| attempts:", attempts, "| rules ok:", ok, "| decision:", decision, "%.2f" % top)


if __name__ == "__main__":
    main(sys.argv[1:])
