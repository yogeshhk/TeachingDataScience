# workshop_spaCy_basics.py
# spaCy Workshop – Part 1: Core Objects (Doc, Token, Span)
# Covers: nlp object, tokenisation, Doc, Token, Span
# Run: python workshop_spaCy_basics.py
# Requires: pip install spacy && python -m spacy download en_core_web_sm

import spacy
from spacy.lang.en import English
from spacy.lang.de import German
from spacy.tokens import Doc

print("=" * 60)
print("PART 1 – Minimal NLP object (no model, English rules only)")
print("=" * 60)

nlp_en = English()
doc = nlp_en("This is a sentence.")
print("Text:", doc.text)

# Extra: German
nlp_de = German()
doc_de = nlp_de("Ich liebe NLP!")
print("German tokens:", [t.text for t in doc_de])

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2 – Load a full pre-trained model")
print("=" * 60)

nlp = spacy.load("en_core_web_sm")
doc = nlp("I like tree kangaroos and narwhals.")

# First token
first_token = doc[0]
print("First token:", first_token.text)

# All tokens
print("\nAll tokens:")
for token in doc:
    print(f"  {token.i:2d}  {token.text:15s}  lemma={token.lemma_}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3 – Slices: Span objects")
print("=" * 60)

doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
span = doc[1:3]
print("Span text:", span.text)
print("Span tokens:", [t.text for t in span])

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 4 – Creating a Doc manually")
print("=" * 60)

words  = ["Hello", "world", "!"]
spaces = [True,    False,   False]
manual_doc = Doc(nlp.vocab, words=words, spaces=spaces)
print("Manual doc:", manual_doc.text)

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCISE – Fill in the blanks (answers shown)")
print("=" * 60)
# Task: import English, create nlp, create doc, print text
from spacy.lang.en import English          # import
nlp2 = English()                           # create nlp
doc2 = nlp2("This is a sentence.")         # create doc
print("Exercise doc text:", doc2.text)     # print text
