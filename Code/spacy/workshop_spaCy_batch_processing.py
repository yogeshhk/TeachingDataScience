# workshop_spaCy_batch_processing.py
# spaCy Workshop – Part 9: Efficient Batch Processing with nlp.pipe()
# Run: python workshop_spaCy_batch_processing.py
# Requires: pip install spacy && python -m spacy download en_core_web_sm
#
# Key rule: never call nlp(text) inside a for-loop over many texts.
# nlp.pipe(texts) streams documents through the pipeline in batches,
# reusing internal buffers and reducing Python overhead significantly.

import spacy
import time

nlp = spacy.load("en_core_web_sm")

# ----------------------------------------------------------------
print("=" * 60)
print("PART 1 – Speed comparison: loop vs nlp.pipe()")
print("=" * 60)

base_texts = [
    "Apple was founded by Steve Jobs in Cupertino.",
    "Google is headquartered in Mountain View, California.",
    "Amazon acquired Whole Foods Market in 2017.",
    "Microsoft released Windows 11 in October 2021.",
    "Tesla manufactures electric vehicles in Texas.",
]
texts = base_texts * 200    # 1 000 texts

# SLOW: one call per text
t0 = time.time()
docs_loop = [nlp(t) for t in texts]
t_loop = time.time() - t0
print(f"  Loop   : {t_loop:.2f}s  for {len(docs_loop)} docs")

# FAST: batch stream
t0 = time.time()
docs_pipe = list(nlp.pipe(texts, batch_size=50))
t_pipe = time.time() - t0
print(f"  pipe() : {t_pipe:.2f}s  for {len(docs_pipe)} docs")
print(f"  Speedup: {t_loop / t_pipe:.1f}x")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2 – Disable unused components for extra speed")
print("=" * 60)

# We only need NER -- skip tagger and parser
t0 = time.time()
ner_only = list(nlp.pipe(texts, batch_size=50, disable=["tagger", "parser"]))
print(f"  NER-only pipe(): {time.time()-t0:.2f}s")

# Verify entities are still found
for doc in ner_only[:3]:
    orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
    print(f"    ORGs in '{doc.text[:45]}...' -> {orgs}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3 – Passing metadata with as_tuples=True")
print("=" * 60)

tagged_data = [
    ("Apple was founded by Steve Jobs.", {"id": 1, "source": "wiki"}),
    ("Tesla builds electric vehicles.",  {"id": 2, "source": "news"}),
    ("Google acquired DeepMind in 2014.",{"id": 3, "source": "blog"}),
    ("Amazon launched AWS in 2006.",     {"id": 4, "source": "report"}),
]

results = []
for doc, meta in nlp.pipe(tagged_data, as_tuples=True, disable=["parser"]):
    orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
    persons = [e.text for e in doc.ents if e.label_ == "PERSON"]
    results.append({"id": meta["id"], "source": meta["source"],
                    "orgs": orgs, "persons": persons})

print(f"  {'ID':4s}  {'Source':8s}  {'ORGs':30s}  {'PERSONs'}")
print("  " + "-" * 65)
for r in results:
    print(f"  {r['id']:<4d}  {r['source']:8s}  {str(r['orgs']):30s}  {r['persons']}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 4 – Collecting only what you need (memory efficient)")
print("=" * 60)

news_texts = [
    "Elon Musk visited Berlin yesterday.",
    "The Fed raised interest rates by 25 basis points.",
    "Sundar Pichai announced new AI features at Google I/O.",
    "OpenAI released GPT-4o in May 2024.",
    "The European Union passed the AI Act.",
]

# Extract only (sentence count, entity list) without storing all docs
summaries = [
    (len(list(doc.sents)), [(e.text, e.label_) for e in doc.ents])
    for doc in nlp.pipe(news_texts, disable=["tagger"])
]
for text, (n_sents, ents) in zip(news_texts, summaries):
    print(f"  '{text[:45]}...'")
    print(f"    sentences={n_sents}  entities={ents}\n")

# ----------------------------------------------------------------
print("=" * 60)
print("EXERCISE – Use nlp.pipe() to count ORG entities per text")
print("           and report which text has the most")
print("=" * 60)

exercise_texts = [
    "Microsoft and Google are competing in cloud computing.",
    "Apple, Samsung, and Huawei dominate the smartphone market.",
    "Tesla is the most valuable car company.",
    "Amazon, Walmart, and Target are major US retailers.",
]

org_counts = [
    sum(1 for e in doc.ents if e.label_ == "ORG")
    for doc in nlp.pipe(exercise_texts, disable=["parser"])
]

for text, count in zip(exercise_texts, org_counts):
    print(f"  ORGs={count}  '{text}'")

best_idx = org_counts.index(max(org_counts))
print(f"\n  Most ORGs found in text {best_idx + 1}: '{exercise_texts[best_idx]}'")
