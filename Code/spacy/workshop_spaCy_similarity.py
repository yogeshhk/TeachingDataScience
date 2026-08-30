# workshop_spaCy_similarity.py
# spaCy Workshop – Part 8: Word Vectors and Similarity
# Run: python workshop_spaCy_similarity.py
# Requires: pip install spacy
#           python -m spacy download en_core_web_md   (vectors included)
#
# NOTE: en_core_web_sm has NO word vectors -- use en_core_web_md or _lg.
# Similarity = cosine similarity between vectors; range [-1, 1].
# 1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite.

import spacy
import numpy as np

nlp = spacy.load("en_core_web_md")   # md/lg models carry 300-dim GloVe vectors

# ----------------------------------------------------------------
print("=" * 60)
print("PART 1 – Token vectors: has_vector, vector_norm, vector shape")
print("=" * 60)

doc = nlp("dog cat banana iPhone")
for token in doc:
    print(f"  {token.text:10s}  has_vector={token.has_vector}  "
          f"norm={token.vector_norm:6.2f}  vector shape={token.vector.shape}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2 – Token-to-token similarity")
print("=" * 60)

pairs = [("dog", "cat"), ("dog", "banana"), ("king", "queen"), ("car", "automobile")]
for w1, w2 in pairs:
    t1 = nlp(w1)[0]
    t2 = nlp(w2)[0]
    print(f"  {w1:12s} <-> {w2:12s}  similarity = {t1.similarity(t2):.3f}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3 – Document-level similarity (average of token vectors)")
print("=" * 60)

sentences = [
    ("I love fast food",           "I enjoy eating pizza"),
    ("I love fast food",           "The stock market crashed today"),
    ("Python is great for AI",     "Machine learning uses Python"),
    ("The cat sat on the mat",     "A dog lay on the rug"),
]
for s1, s2 in sentences:
    d1 = nlp(s1)
    d2 = nlp(s2)
    print(f"  {s1!r:40s}")
    print(f"  {s2!r:40s}")
    print(f"  similarity = {d1.similarity(d2):.3f}\n")

# ----------------------------------------------------------------
print("=" * 60)
print("PART 4 – Span similarity")
print("=" * 60)

tokens = nlp("dog cat banana apple mango")
span_a = tokens[0:2]   # "dog cat"
span_b = tokens[2:5]   # "banana apple mango"
print(f"  Span A: '{span_a.text}'")
print(f"  Span B: '{span_b.text}'")
print(f"  Similarity: {span_a.similarity(span_b):.3f}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 5 – Find the most similar word in a list (nearest neighbour)")
print("=" * 60)

query  = nlp("doctor")[0]
vocab_words = ["nurse", "teacher", "hospital", "school", "car", "surgery", "engineer"]
vocab_tokens = [nlp(w)[0] for w in vocab_words]

scored = sorted(vocab_tokens, key=lambda t: query.similarity(t), reverse=True)
print(f"  Words most similar to '{query.text}':")
for t in scored:
    print(f"    {t.text:15s}  {query.similarity(t):.3f}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCISE – Compute similarity between two sentences of your choice")
print("=" * 60)

s_a = nlp("spaCy makes natural language processing easy.")
s_b = nlp("Processing text is simple with spaCy.")
s_c = nlp("The weather is nice today.")
print(f"  A vs B (related)    : {s_a.similarity(s_b):.3f}")
print(f"  A vs C (unrelated)  : {s_a.similarity(s_c):.3f}")
