# spaCy NLP Examples

Scripts demonstrating spaCy for Named Entity Recognition (NER), custom pipelines, and batch processing — with a focus on Indic languages via AI4Bharat models.

## Setup

```bash
pip install spacy
python -m spacy download en_core_web_sm
# For Indic NER:
pip install ai4bharat-transliteration
```

## Files

| File | Description |
|------|-------------|
| `spacy_indic_ner_*.py` | NER on Hindi/Marathi text using AI4Bharat models |
| `spacy_custom_ner_*.py` | Training a custom NER model (medical domain) |
| `spacy_batch_*.py` | Efficient batch processing with `nlp.pipe()` |
| `spacy_pipeline_*.py` | Adding custom components to the spaCy pipeline |

## Key Concepts

- `nlp(text)` → `Doc` → `doc.ents` for entity extraction
- Custom `EntityRuler` and `Matcher` for rule-based NER
- Training with `spacy train config.cfg` (spaCy v3 config system)
- `nlp.pipe(texts, batch_size=N)` for throughput on large corpora
