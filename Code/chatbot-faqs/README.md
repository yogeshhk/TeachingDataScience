# FAQ Chatbot with RAG

A FAQ chatbot built with LlamaIndex and the HuggingFace API. It embeds the questions from a CSV of
question-answer pairs, finds the most similar question for a user query, and returns its stored
answer if the similarity is above a threshold.

## Files

| File | Purpose |
|---|---|
| `main_faq_chatbot.py` | `FAQChatbot` class (index the questions, `query()`, `get_faq_stats()`); running it tests a few questions on `data/BankFAQs.csv` |
| `streamlit_main.py` | Streamlit UI: upload a CSV in the sidebar, chat, adjust the similarity threshold |
| `benchmark_testing.py` | Benchmarks random questions from the CSV by cosine similarity (default) or an LLM judge |
| `test_chatbot_faqs.py` | Mocked pytest suite (no API calls) |
| `data/BankFAQs.csv` | Sample FAQ data (columns `Question`, `Answer`, `Class`) |
| `environment.yml`, `requirements.txt` | Dependencies (conda environment `chatbot-faqs`) |
| `storage/` | Saved index files; `benchmark_results.csv` is benchmark output |

## Run

```bash
conda env create -f environment.yml
conda activate chatbot-faqs
set HUGGINGFACE_API_KEY=your_hf_api_key_here
streamlit run streamlit_main.py
```

Other entry points:

```bash
python main_faq_chatbot.py
python benchmark_testing.py
python -m pytest test_chatbot_faqs.py -v
```

## CSV format

The first column holds the questions and the second the answers; further columns are ignored and
empty rows are dropped.

## Configuration

- Similarity threshold: the constructor default is 0.7; the Streamlit slider runs from 0.1 to 1.0
  (default 0.7) and `main_faq_chatbot.py` tests with 0.6. Lower is looser matching, higher is stricter.
- Benchmark mode: `evaluation_mode='cosine'` (sentence-transformers embeddings) or `'llm'` (LLM judge),
  set where `BenchmarkTester` is constructed.
- `HUGGINGFACE_API_KEY` must be set in the environment, otherwise the chatbot raises an error at start.
