# workshop_spaCy_matcher.py
# spaCy Workshop – Part 5: Rule-based Matching with Matcher
# Run: python workshop_spaCy_matcher.py
# Requires: pip install spacy && python -m spacy download en_core_web_sm
#
# Matcher uses token-attribute pattern dicts, unlike PhraseMatcher which
# matches on exact text sequences.  Key pattern attributes:
#   TEXT, LOWER, POS, TAG, DEP, LEMMA, IS_DIGIT, IS_ALPHA, IS_PUNCT
#   OP  -> quantifier: "?" optional, "+" one-or-more, "*" zero-or-more

import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

# ----------------------------------------------------------------
print("=" * 60)
print("PART 1 – Basic pattern: adjective + noun")
print("=" * 60)

matcher = Matcher(nlp.vocab)

# Optional adjective followed by a noun
pattern = [{"POS": "ADJ", "OP": "?"}, {"POS": "NOUN"}]
matcher.add("ADJ_NOUN", [pattern])

doc = nlp("I have a big red apple and fresh coffee.")
for match_id, start, end in matcher(doc):
    label = nlp.vocab.strings[match_id]
    print(f"  {label:15s}  '{doc[start:end].text}'")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2 – Matching product names (LOWER + IS_DIGIT)")
print("=" * 60)

matcher2 = Matcher(nlp.vocab)

iphone_pattern  = [{"LOWER": "iphone"}, {"IS_DIGIT": True}]
matcher2.add("IPHONE_MODEL", [iphone_pattern])

company_pattern = [
    {"POS": "PROPN", "OP": "+"},
    {"TEXT": {"IN": ["Inc", "Corp", "Ltd"]}}
]
matcher2.add("COMPANY", [company_pattern])

doc2 = nlp("The iPhone 15 was released by Apple Inc in 2023. Samsung Corp is a competitor.")
for match_id, start, end in matcher2(doc2):
    label = nlp.vocab.strings[match_id]
    print(f"  {label:15s}  '{doc2[start:end].text}'")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3 – Multiple alternative patterns for one rule")
print("=" * 60)

matcher3 = Matcher(nlp.vocab)

# Match "good morning" OR "good evening"
p1 = [{"LOWER": "good"}, {"LOWER": "morning"}]
p2 = [{"LOWER": "good"}, {"LOWER": "evening"}]
matcher3.add("GREETING", [p1, p2])

doc3 = nlp("Good morning! I said good evening to her.")
for match_id, start, end in matcher3(doc3):
    print(f"  Match: '{doc3[start:end].text}'")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 4 – Using OP quantifier: one-or-more proper nouns")
print("=" * 60)

matcher4 = Matcher(nlp.vocab)
# One or more consecutive proper nouns (e.g. "New York", "South Africa")
pattern4 = [{"POS": "PROPN", "OP": "+"}]
matcher4.add("MULTI_PROPN", [pattern4])

doc4 = nlp("I visited New York and South Africa last summer.")
seen = set()
for match_id, start, end in matcher4(doc4):
    span_text = doc4[start:end].text
    if span_text not in seen:          # skip sub-spans already printed
        print(f"  PROPN group: '{span_text}'")
    seen.add(span_text)

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCISE – Verb followed by a proper noun")
print("           e.g. 'acquired SolarCity', 'hired Google'")
print("=" * 60)

matcher5 = Matcher(nlp.vocab)
pattern5 = [{"POS": "VERB"}, {"POS": "PROPN"}]
matcher5.add("VERB_PROPN", [pattern5])

doc5 = nlp("Tesla acquired SolarCity and hired engineers from Google.")
for match_id, start, end in matcher5(doc5):
    print(f"  Match: '{doc5[start:end].text}'")
