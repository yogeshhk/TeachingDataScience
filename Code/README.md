# Code

Runnable Python scripts and Jupyter notebooks demonstrating the concepts covered in the
`LaTeX/` slide decks, drawn from many contributors and courses, reorganized here by area.
Where a subdirectory has its own `README.md`, it has more detail (setup, features) than the
one-liner below; click through for that.

Standard setup for any subdirectory with an `environment.yml`:
```bash
conda env create -f Code/<subdir>/environment.yml
conda activate <env-name>
```

## GenAI / Agents

| Directory | Covers |
|---|---|
| [`langchain/`](langchain/) | LangChain examples, fundamentals through advanced RAG and agent patterns |
| [`langgraph/`](langgraph/) | LangGraph examples covering stateful, graph-based agent orchestration |
| [`llamaindex/`](llamaindex/) | LlamaIndex fundamentals, RAG patterns, agents, Groq/Ollama/HuggingFace integrations |
| [`crewai/`](crewai/) | CrewAI multi-agent example (`researcher` subproject) |
| [`agents/`](agents/) | Proof-of-concept agents across AutoGen, CrewAI, and LangGraph |
| [`agno/`](agno/) | Minimal agent scripts using the Agno framework, backed by local LLM servers |
| [`google-adk/`](google-adk/) | Agent examples using Google's Agent Development Kit (ADK) with Gemini models |

## RAG Applications

| Directory | Covers |
|---|---|
| [`chatbot-faqs/`](chatbot-faqs/) | FAQ chatbot using RAG with LlamaIndex and HuggingFace |
| [`chatbot-multimodal/`](chatbot-multimodal/) | Multi-modal RAG (text, tables, images, code) via Docling + LlamaIndex |
| [`omni-rag/`](omni-rag/) | Research-grade multimodal RAG for complex documents (tables, headers) |
| [`parsing/`](parsing/) | Resume parsing + RAG toolkit for chatting with resume data |
| [`graphrag/`](graphrag/) | Knowledge graph construction and graph-based QA (LangChain + Groq) |

## LLM Fine-tuning & Serving

| Directory | Covers |
|---|---|
| [`fine-tuning/`](fine-tuning/) | Overview and examples of fine-tuning small language models (SLMs) |
| `amd/` | AMD Academy workshop materials: AI agents, fine-tuning, LLM serving with vLLM |
| [`LOCAL_LLM_PLAN.md`](LOCAL_LLM_PLAN.md) | Running small LLMs/SLMs locally on modest hardware (measured tok/s, model picks, LM Studio/Ollama setup); strongest use case is offline, reproducible classroom demos |

## Document Parsing

| Directory | Covers |
|---|---|
| [`docling/`](docling/) | Streamlit book-QA app using Docling for parsing + LangChain for Q&A |
| [`opendataloader/`](opendataloader/) | Tutorial suite for the `opendataloader-pdf` library |

## Deep Learning

| Directory | Covers |
|---|---|
| [`pytorch/`](pytorch/) | PyTorch fundamentals: tensors, neural networks, training loops, image classification |
| `curiosily_ai_bootcamp/` | AI Bootcamp notebooks (supersedes the earlier `dl_curiousily/`) |

## Classical ML

| Directory | Covers |
|---|---|
| [`ml/`](ml/) | ~160 notebooks covering classical ML algorithms and data analysis patterns |
| `math/` | Statistics and math tutorial notebooks (Bayes' theorem, hypothesis testing, distributions) |
| `pandas/` | Pandas tutorial notebooks |
| `python/` | Individual Python script examples (data structures, algorithms, Big-O, graphics) |
| [`interview/`](interview/) | Coding-interview practice: LeetCode/PyTorch/DSA exercises |

## NLP

| Directory | Covers |
|---|---|
| [`spacy/`](spacy/) | spaCy for NER, custom pipelines, batch processing, with an Indic-language focus |
| `nlp/` | Classical NLP tutorials (BERT, NLTK, spaCy, Doc2Vec classification, knowledge graphs) |
| `dnlp/` | Deep NLP notebooks: attention mechanisms, sentiment analysis, NER with BiLSTM-CRF |
| `dbpedia/` | Neural extraction framework for DBpedia (GSoC 2024 project) |

## GNN

| Directory | Covers |
|---|---|
| [`gnn/`](gnn/) | Graph Neural Networks with PyTorch Geometric: molecular property prediction, knowledge graphs |
| `AIinGraphs/` | Standalone write-up on graph-based AI |

## Indic Language

| Directory | Covers |
|---|---|
| [`mahamarathi/`](mahamarathi/) | Marathi language dataset (Telugu-LLM-Labs Marathi Alpaca) |
| [`sarvam/`](sarvam/) | Indic language AI via Sarvam AI and HuggingFace Inference APIs (speech, translation, TTS) |
| [`orgpedia/`](orgpedia/) | Automatic Q&A generation for OrgPedia ML training |

## Research References & Tools

| Directory | Covers |
|---|---|
| [`txt2cad/`](txt2cad/) | Research reference / design docs for Text-to-CAD generation (no runnable code) |
| [`txt2sql/`](txt2sql/) | Research reference / design docs for Text-to-SQL generation (no runnable code) |
| `chromeext/` | Chrome extension project (LifeTimer) |
| `claudecode/` | Claude Code workshop materials, example configs, and community resources |
| `opencode/` | OpenCode CLI demo project |
| `reports/` | Dated repo/codebase upgrade reports |

---

**Note:** `Code/.gitignore` covers `__pycache__/`, `.ipynb_checkpoints/`, `.env`, `*.pyc`, and
model weights (`*.bin`, `*.pt`, `*.safetensors`) repo-wide.
