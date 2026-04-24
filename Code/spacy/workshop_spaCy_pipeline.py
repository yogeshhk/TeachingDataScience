# workshop_spaCy_pipeline.py
# spaCy Workshop – Part 3: Pipeline Inspection, Disabling, Custom Components
# Run: python workshop_spaCy_pipeline.py
# Requires: pip install spacy && python -m spacy download en_core_web_sm
#
# NOTE: spaCy v3 changes from slides –
#   OLD (v2): nlp.add_pipe(func, name="x", last=True)
#   NEW (v3): decorate with @Language.component("x"), then nlp.add_pipe("x")
#   OLD (v2): matcher.add(label, None, *patterns)
#   NEW (v3): matcher.add(label, patterns)   # patterns is a list

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

# ----------------------------------------------------------------
print("=" * 60)
print("PART 1 – Inspecting the pipeline")
print("=" * 60)

nlp = spacy.load("en_core_web_sm")
print("Pipeline components:", nlp.pipe_names)
print("\nFull pipeline (name, component object):")
for name, proc in nlp.pipeline:
    print(f"  {name:15s}  {type(proc).__name__}")

# Manual iteration (shows what happens internally)
doc = nlp.make_doc("This is a sentence")
for name, proc in nlp.pipeline:
    doc = proc(doc)
print("\nAfter manual pipeline run, POS of first token:", doc[0].pos_)

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2 – Disabling pipeline components")
print("=" * 60)

# Load with specific components disabled
nlp_no_parser = spacy.load("en_core_web_sm", disable=["parser"])
print("Pipe names (parser disabled):", nlp_no_parser.pipe_names)

texts = ["Apple bought a startup.", "Google hired engineers."]
# nlp.pipe() is efficient for batches; disable what you don't need
for doc in nlp.pipe(texts, disable=["parser"]):
    print("  Entities:", [(e.text, e.label_) for e in doc.ents])

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3 – Custom component (spaCy v3 style)")
print("=" * 60)

# Register the component with the @Language.component decorator
@Language.component("doc_length_logger")
def doc_length_logger(doc):
    print(f"  [doc_length_logger] Doc has {len(doc)} tokens.")
    if len(doc) < 10:
        print("  [doc_length_logger] Short document.")
    return doc  # always return the doc

nlp2 = spacy.load("en_core_web_sm")
nlp2.add_pipe("doc_length_logger", first=True)   # insert before all others
print("Updated pipe names:", nlp2.pipe_names)
doc2 = nlp2("This is a sentence.")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 4 – Custom component as a class (PhraseMatcher, v3 style)")
print("=" * 60)

@Language.factory("animal_matcher")
def create_animal_matcher(nlp, name):
    return AnimalMatcher(nlp)

class AnimalMatcher:
    def __init__(self, nlp):
        terms = ["cat", "dog", "tree kangaroo", "giant sea spider"]
        patterns = [nlp.make_doc(t) for t in terms]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add("ANIMAL", patterns)   # v3: list, no None

    def __call__(self, doc):
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = Span(doc, start, end, label=match_id)
            doc.ents = list(doc.ents) + [span]
        return doc

nlp3 = spacy.load("en_core_web_sm")
nlp3.add_pipe("animal_matcher", after="ner")
print("Pipe names:", nlp3.pipe_names)

doc3 = nlp3("I have a cat and a tree kangaroo at home.")
print("Entities found:")
for ent in doc3.ents:
    print(f"  {ent.text:20s}  {ent.label_}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCISE – Add a component that prints sentence count")
print("=" * 60)

@Language.component("sentence_counter")
def sentence_counter(doc):
    sents = list(doc.sents)
    print(f"  [sentence_counter] {len(sents)} sentence(s) found.")
    return doc

nlp4 = spacy.load("en_core_web_sm")
nlp4.add_pipe("sentence_counter", last=True)
nlp4("I love spaCy. It is very powerful. This is sentence three.")
