# workshop_spaCy_token_attributes.py
# spaCy Workshop – Part 2: Token Attributes, POS, NER, Best Practices
# Run: python workshop_spaCy_token_attributes.py
# Requires: pip install spacy && python -m spacy download en_core_web_sm

import spacy

nlp = spacy.load("en_core_web_sm")

# ----------------------------------------------------------------
print("=" * 60)
print("PART 1 – Part-of-Speech (POS) tags")
print("=" * 60)

doc = nlp("Berlin looks like a nice city.")
print(f"{'Token':15s}  {'POS':8s}  {'TAG':8s}  {'DEP':10s}  {'Lemma'}")
print("-" * 60)
for token in doc:
    print(f"{token.text:15s}  {token.pos_:8s}  {token.tag_:8s}  {token.dep_:10s}  {token.lemma_}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2 – Named Entity Recognition (NER)")
print("=" * 60)

doc1 = nlp("Apple is looking at buying U.K. startup for $1 billion")
doc2 = nlp("Ishu ate the Apple")

for doc, label in [(doc1, "Sentence 1"), (doc2, "Sentence 2")]:
    print(f"\n{label}: {doc.text}")
    if doc.ents:
        for ent in doc.ents:
            print(f"  Entity: {ent.text:20s}  Label: {ent.label_:10s}  ({spacy.explain(ent.label_)})")
    else:
        print("  No entities found.")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3 – Useful token boolean flags")
print("=" * 60)

doc = nlp("She sold 500 seashells by the seashore.")
print(f"{'Token':15s}  {'alpha':6s}  {'digit':6s}  {'stop':6s}  {'punct':6s}")
print("-" * 55)
for token in doc:
    print(f"{token.text:15s}  {str(token.is_alpha):6s}  {str(token.is_digit):6s}  "
          f"{str(token.is_stop):6s}  {str(token.is_punct):6s}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 4 – BAD practice vs GOOD practice")
print("         (slide 'Evaluate' + 'Solution')")
print("=" * 60)

doc = nlp("Berlin looks like a nice city")

print("\n-- BAD: converting to plain lists early --")
token_texts = [token.text for token in doc]
pos_tags    = [token.pos_  for token in doc]
for index, pos in enumerate(pos_tags):
    if pos == "PROPN":
        # BUG: index+1 can exceed list bounds at last token
        if pos_tags[index + 1] == "VERB":
            print("Found (bad way):", token_texts[index])

print("\n-- GOOD: use native Doc attributes, check bounds --")
for token in doc:
    if token.pos_ == "PROPN":
        if token.i + 1 < len(doc) and doc[token.i + 1].pos_ == "VERB":
            print("Found proper noun before a verb:", token.text)

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCISE – Find all ORG entities in a longer text")
print("=" * 60)

text = ("Google was founded by Larry Page and Sergey Brin while they were "
        "students at Stanford University. Microsoft was founded by Bill Gates.")
doc = nlp(text)
orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
print("ORG entities:", orgs)
