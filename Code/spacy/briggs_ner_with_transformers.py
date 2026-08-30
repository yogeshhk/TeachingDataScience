# NER With Transformers and spaCy (Python) -  James Briggs https://www.youtube.com/watch?v=W8ZPQOcHnlE
import spacy
from spacy import displacy

txt = """Fastly released its Q1-21 performance on Thursday, after which the stock price dropped a whopping 27%.
        The company generated revenues of $84.9 Millions (35% YoY) vs. $85.1 million market consensus. Net loss
        per share was 0.12 vs . an expected $0.11."""
print("--- SM ---")
sm_model = spacy.load('en_core_web_sm')
doc_sm = sm_model(txt)

for ent in doc_sm.ents:
    print(ent.text, ent.label_)

print("--- TRF ---")
trf_model = spacy.load('en_core_web_trf')
doc_trf = trf_model(txt)

for ent in doc_trf.ents:
    print(ent.text, ent.label_)