# workshop_spaCy_vocab_strings.py
# spaCy Workshop – Part 4: Vocab, StringStore, Hashes, Lexemes
# Run: python workshop_spaCy_vocab_strings.py
# Requires: pip install spacy && python -m spacy download en_core_web_sm

import spacy
from spacy.lang.en import English

# ----------------------------------------------------------------
print("=" * 60)
print("PART 1 – StringStore: strings <-> hash values")
print("=" * 60)

nlp = spacy.load("en_core_web_sm")
doc = nlp("I love coffee")

coffee_hash   = nlp.vocab.strings["coffee"]
coffee_string = nlp.vocab.strings[coffee_hash]
print(f"'coffee'  -->  hash : {coffee_hash}")
print(f" hash     -->  string: {coffee_string}")

# The same hash is accessible through the doc's vocab
print("\nAccessing via doc.vocab:")
print(f"  doc.vocab.strings['coffee'] = {doc.vocab.strings['coffee']}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2 – Lexeme attributes (context-independent word info)")
print("=" * 60)

doc = nlp("I love coffee")
print(f"{'text':12s}  {'orth':22s}  {'shape':10s}  {'prefix':8s}  "
      f"{'suffix':8s}  {'alpha':6s}  {'digit':6s}  {'title':6s}  {'lang'}")
print("-" * 90)
for word in doc:
    lex = doc.vocab[word.text]
    print(f"{lex.text:12s}  {lex.orth:<22d}  {lex.shape_:10s}  {lex.prefix_:8s}  "
          f"{lex.suffix_:8s}  {str(lex.is_alpha):6s}  {str(lex.is_digit):6s}  "
          f"{str(lex.is_title):6s}  {lex.lang_}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3 – Why hashes? Shared vocab across docs")
print("=" * 60)

doc1 = nlp("I love coffee")
doc2 = nlp("Do you love tea?")

# Same vocab -> same hash for "love" in both docs
hash1 = doc1.vocab.strings["love"]
hash2 = doc2.vocab.strings["love"]
print(f"Hash of 'love' in doc1: {hash1}")
print(f"Hash of 'love' in doc2: {hash2}")
print(f"Same hash? {hash1 == hash2}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCISE – Look up 'cat' hash and reverse-lookup the string")
print("=" * 60)

nlp_plain = English()
doc = nlp_plain("I have a cat")

cat_hash   = nlp_plain.vocab.strings["cat"]
cat_string = nlp_plain.vocab.strings[cat_hash]
print(f"'cat' hash  : {cat_hash}")
print(f"hash -> str : {cat_string}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 4 – Entity/POS labels are also hashed")
print("=" * 60)

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is a company in California.")
for ent in doc.ents:
    label_hash   = nlp.vocab.strings[ent.label_]
    label_string = nlp.vocab.strings[label_hash]
    print(f"  {ent.text:20s}  label={ent.label_}  hash={label_hash}  back={label_string}")
