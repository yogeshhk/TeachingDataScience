# workshop_spaCy_sentences.py
# spaCy Workshop – Part 6: Sentence Segmentation
# Run: python workshop_spaCy_sentences.py
# Requires: pip install spacy && python -m spacy download en_core_web_sm
#
# spaCy assigns sentence boundaries during dependency parsing.
# Sentence start tokens have token.is_sent_start = True.
# For lightweight use (no parser), add the "sentencizer" component instead.

import spacy

# ----------------------------------------------------------------
print("=" * 60)
print("PART 1 – Basic sentence extraction with full model")
print("=" * 60)

nlp = spacy.load("en_core_web_sm")
doc = nlp("spaCy is fast. It handles many NLP tasks. Sentences are easy to extract!")

sentences = list(doc.sents)
print(f"Number of sentences: {len(sentences)}")
for i, sent in enumerate(sentences):
    print(f"  Sentence {i + 1} ({len(sent)} tokens): {sent.text}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2 – Sentence-start markers on tokens")
print("=" * 60)

for token in doc:
    if token.is_sent_start:
        print(f"  Sentence starts at token [{token.i}]: '{token.text}'")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3 – Sentence spans are Span objects")
print("=" * 60)

doc = nlp("Alice works at Google. She lives in New York.")
for sent in doc.sents:
    ents = [(e.text, e.label_) for e in sent.ents]
    print(f"  Sentence : {sent.text}")
    print(f"  Entities : {ents}")
    print()

# ----------------------------------------------------------------
print("=" * 60)
print("PART 4 – Lightweight sentencizer (no parser needed)")
print("=" * 60)

nlp_light = spacy.blank("en")        # blank pipeline -- no components
nlp_light.add_pipe("sentencizer")    # splits on . ! ?
print("Pipe names:", nlp_light.pipe_names)

doc_light = nlp_light("First sentence. Second sentence! Third one? Yes, this too.")
for sent in doc_light.sents:
    print(f"  - {sent.text}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 5 – Sentence word counts (practical pattern)")
print("=" * 60)

nlp2 = spacy.load("en_core_web_sm")
text = ("Natural language processing is a subfield of AI. "
        "It focuses on interactions between computers and human language. "
        "spaCy makes NLP accessible to everyone.")

doc2 = nlp2(text)
for i, sent in enumerate(doc2.sents):
    words = [t.text for t in sent if not t.is_punct and not t.is_space]
    print(f"  Sentence {i + 1}: {len(words)} words — '{sent.text[:50]}...'")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCISE – Count sentences and find the longest one")
print("=" * 60)

doc3 = nlp2(
    "spaCy is designed specifically for production use. "
    "It helps build real applications that process and understand large volumes of text. "
    "It can be used to build information extraction systems. "
    "It is fast."
)
sents = list(doc3.sents)
longest = max(sents, key=lambda s: len(s))
print(f"Total sentences : {len(sents)}")
print(f"Longest sentence: '{longest.text}'  ({len(longest)} tokens)")
