# spaCy NLP Examples

Scripts demonstrating spaCy for Named Entity Recognition (NER), custom pipelines, and batch processing, with a focus on Indic languages (Hindi and Marathi training configs, AI4Bharat Naamapadam data).

## Setup

There is no `environment.yml` here. Create a conda environment with spaCy and download the English model:

```bash
conda create -n spacy -c conda-forge python=3.10 spacy
conda activate spacy
python -m spacy download en_core_web_sm
```

## Files

| File | Description |
|------|-------------|
| `workshop_spaCy_basics.py`, `workshop_spaCy_token_attributes.py`, `workshop_spaCy_vocab_strings.py`, `workshop_spaCy_sentences.py` | Workshop scripts: Doc and tokens, token attributes, vocab and strings, sentences |
| `workshop_spaCy_pipeline.py`, `workshop_spaCy_batch_processing.py` | Pipeline components and batch processing with `nlp.pipe()` |
| `workshop_spaCy_matcher.py`, `workshop_spaCy_similarity.py`, `workshop_spaCy_displacy.py` | Rule-based matching, similarity, displaCy visualisation |
| `workshop_*.html` | displaCy output pages written by the workshop scripts |
| `ai4bharat_indic_ner.py` | Converts AI4Bharat Naamapadam JSON to IOB for training Hindi and Marathi NER |
| `base_config.cfg`, `config.cfg`, `hindi_base_config.cfg`, `hindi_config.cfg`, `marathi_base_config.cfg` | spaCy v3 training configs |
| `nichite_custom_ner_medical.py`, `nichite_custom_ner_upload.py` | Custom NER for the medical domain |
| `briggs_ner_with_transformers.py` | NER with a transformer pipeline |
| `data/` | `Corona2.json` annotations and `medical_train.spacy` |

## Key Concepts

- `nlp(text)` gives a `Doc`; `doc.ents` holds the entities
- Rule-based NER with `EntityRuler` and `Matcher`
- Training with `spacy train config.cfg` (spaCy v3 config system)
- `nlp.pipe(texts, batch_size=N)` for throughput on large corpora
