# workshop_spaCy_displacy.py
# spaCy Workshop – Part 7: Visualizing NER and Dependency Parses with displacy
# Run: python workshop_spaCy_displacy.py
# Requires: pip install spacy && python -m spacy download en_core_web_sm
#
# displacy has two styles:
#   "ent"  -- highlights named entities with colored labels
#   "dep"  -- draws dependency arc diagrams
#
# In scripts, use displacy.render(...) to get an HTML string and save to file.
# In Jupyter notebooks, pass jupyter=True to render inline.

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

# ----------------------------------------------------------------
print("=" * 60)
print("PART 1 – NER visualizer: save HTML and print entity table")
print("=" * 60)

doc_ner = nlp("Apple is looking at buying U.K. startup for $1 billion.")

# Save as HTML file
html_ner = displacy.render(doc_ner, style="ent", page=True)
with open("workshop_ner_output.html", "w", encoding="utf-8") as f:
    f.write(html_ner)
print("Saved workshop_ner_output.html  (open in a browser to view colored entities)")

# Also print as plain text table
print(f"\n{'Entity text':25s}  {'Label':10s}  {'Explanation'}")
print("-" * 60)
for ent in doc_ner.ents:
    print(f"{ent.text:25s}  {ent.label_:10s}  {spacy.explain(ent.label_)}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2 – Dependency parse visualizer")
print("=" * 60)

doc_dep = nlp("Autonomous cars shift insurance liability towards manufacturers.")

html_dep = displacy.render(doc_dep, style="dep", page=True)
with open("workshop_dep_output.html", "w", encoding="utf-8") as f:
    f.write(html_dep)
print("Saved workshop_dep_output.html  (open in a browser to view arc diagram)")

# Print dependency table
print(f"\n{'Token':15s}  {'Head':15s}  {'Dep':12s}  {'Children'}")
print("-" * 65)
for token in doc_dep:
    children = [c.text for c in token.children]
    print(f"{token.text:15s}  {token.head.text:15s}  {token.dep_:12s}  {children}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3 – Visualize multiple sentences (list of docs)")
print("=" * 60)

texts = [
    "Google was founded by Larry Page.",
    "Elon Musk runs Tesla and SpaceX.",
]
docs = list(nlp.pipe(texts))
html_multi = displacy.render(docs, style="ent", page=True)
with open("workshop_ner_multi.html", "w", encoding="utf-8") as f:
    f.write(html_multi)
print("Saved workshop_ner_multi.html")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 4 – Custom entity colors in displacy")
print("=" * 60)

doc_custom = nlp("Apple and Google are competing in the AI space.")
colors  = {"ORG": "#f4a261", "GPE": "#2a9d8f", "PERSON": "#e9c46a"}
options = {"ents": ["ORG", "GPE", "PERSON"], "colors": colors}

html_custom = displacy.render(doc_custom, style="ent", page=True, options=options)
with open("workshop_ner_custom_colors.html", "w", encoding="utf-8") as f:
    f.write(html_custom)
print("Saved workshop_ner_custom_colors.html")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCISE – Visualize a sentence of your choice")
print("           and print a table of all dependency relations")
print("=" * 60)

doc_ex = nlp("The quick brown fox jumps over the lazy dog.")
html_ex = displacy.render(doc_ex, style="dep", page=True)
with open("workshop_dep_exercise.html", "w", encoding="utf-8") as f:
    f.write(html_ex)
print("Saved workshop_dep_exercise.html")

# Roots (tokens whose head is themselves)
roots = [t for t in doc_ex if t.dep_ == "ROOT"]
print(f"Root token(s): {[r.text for r in roots]}")
