# Sarvam AI & HuggingFace API Examples

Scripts for exploring Indic language AI services: [Sarvam AI](https://www.sarvam.ai/) (speech, translation, TTS for Indian languages) and HuggingFace Inference APIs.

## Files

| File | Description |
|------|-------------|
| `savamai_testing.py` | Tests Sarvam AI endpoints: STT, translation, TTS, text analytics |
| `huggingface_apis.py` | Demonstrates HuggingFace Inference API calls for NLP tasks |

## Setup

```bash
export SARVAM_API_KEY=your_sarvam_key
export HUGGINGFACE_API_KEY=your_hf_key
pip install requests
```

## Use Case

Sarvam AI provides production-grade Indic language models (Hindi, Marathi, Tamil, etc.).
These scripts serve as quick integration tests and API exploration notebooks.

## Related

- `mahamarathi/` — Marathi-specific NLP experiments
- `nlp/` — General NLP notebooks
