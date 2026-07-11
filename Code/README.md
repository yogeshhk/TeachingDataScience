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
| `awesome_llm_apps/` | Example LLM agent apps (financial coach, voice agent, game-playing agents) |
| `activeloop/` | LangChain tutorial notebooks (summarizers, QA chatbots, knowledge graphs, prompting) |
| `nvidia/` | NVIDIA LLM course notebooks: microservices, LangChain, embeddings, vector stores |
| `google_generative-ai/` | Python ports of notebooks from Google's generative-ai repo |

## RAG Applications

| Directory | Covers |
|---|---|
| [`chatbot-faqs/`](chatbot-faqs/) | FAQ chatbot using RAG with LlamaIndex and HuggingFace |
| [`chatbot-multimodal/`](chatbot-multimodal/) | Multi-modal RAG (text, tables, images, code) via Docling + LlamaIndex |
| [`omni-rag/`](omni-rag/) | Research-grade multimodal RAG for complex documents (tables, headers) |
| [`parsing/`](parsing/) | Resume parsing + RAG toolkit for chatting with resume data |
| [`graphrag/`](graphrag/) | Knowledge graph construction and graph-based QA (LangChain + Groq) |
| `gemma/` | Minimal RAG examples using Google's Gemma model |

## LLM Fine-tuning & Serving

| Directory | Covers |
|---|---|
| [`fine-tuning/`](fine-tuning/) | Overview and examples of fine-tuning small language models (SLMs) |
| [`ludwig/`](ludwig/) | Low/no-code, config-based platform for fine-tuning LLMs and other ML/DL workflows |
| `llama/` | Llama 2 fine-tuning scripts |
| `gcp_notebooks/` | Fine-tuning notebooks on GCP (Gemma, Ludwig, CodeT5) |
| `vizuara/` | Building a Small Language Model (SLM) from scratch: notebooks and slides |
| `amd/` | AMD Academy workshop materials: AI agents, fine-tuning, LLM serving with vLLM |

## Document Parsing

| Directory | Covers |
|---|---|
| [`docling/`](docling/) | Streamlit book-QA app using Docling for parsing + LangChain for Q&A |
| [`opendataloader/`](opendataloader/) | Tutorial suite for the `opendataloader-pdf` library |

## Deep Learning

| Directory | Covers |
|---|---|
| [`keras/`](keras/) | *Deep Learning with Python* (Chollet) notebooks, plus GAN examples |
| [`dl_tf2/`](dl_tf2/) | TF2/Keras tutorials: GANs, autoencoders, CNNs, RNNs, Swift for TensorFlow |
| [`pytorch/`](pytorch/) | PyTorch fundamentals: tensors, neural networks, training loops, image classification |
| `deep_rl/` | HuggingFace Deep RL course notebooks (Q-learning, Deep Q-Learning, Unity3D, units 1-8) |
| `dl_curiousily/` | Deep learning tutorials: neural networks through production deployment |
| `animesh1012/` | Plant disease image classifier (train/test notebooks + simple web app) |
| `prodramp/` | Deep learning with satellite imagery: data processing and modeling |
| `curiosily_ai_bootcamp/` | AI Bootcamp notebooks |

## Classical ML

| Directory | Covers |
|---|---|
| [`ml/`](ml/) | ~160 notebooks covering classical ML algorithms and data analysis patterns |
| `math/` | Statistics and math tutorial notebooks (Bayes' theorem, hypothesis testing, distributions) |
| `pandas/` | Pandas tutorial notebooks |
| `python/` | Individual Python script examples (data structures, algorithms, Big-O, graphics) |

## NLP

| Directory | Covers |
|---|---|
| [`spacy/`](spacy/) | spaCy for NER, custom pipelines, batch processing, with an Indic-language focus |
| `nlp/` | Classical NLP tutorials (BERT, NLTK, spaCy, Doc2Vec classification, knowledge graphs) |
| `dnlp/` | Deep NLP notebooks: attention mechanisms, sentiment analysis, NER with BiLSTM-CRF |
| `meta_bAbi_tasks/` | Assessing LLMs' logical interpretation via Meta's bAbI tasks |
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
| `pritamMarathi/` | Neural networks (Karpathy's course) applied to Marathi text generation |

## Research References & Tools

| Directory | Covers |
|---|---|
| [`txt2cad/`](txt2cad/) | Research reference / design docs for Text-to-CAD generation (no runnable code) |
| [`txt2sql/`](txt2sql/) | Research reference / design docs for Text-to-SQL generation (no runnable code) |
| `latex/` | Standalone LaTeX syntax/example files (unrelated to the main `LaTeX/` course directory) |
| `chromeext/` | Chrome extension project (LifeTimer) |
| `claudecode/` | Claude Code workshop materials, example configs, and community resources |
| `opencode/` | OpenCode CLI demo project |
| `reports/` | Dated repo/codebase upgrade reports |

---

**Note:** `Code/.gitignore` covers `__pycache__/`, `.ipynb_checkpoints/`, `.env`, `*.pyc`, and
model weights (`*.bin`, `*.pt`, `*.safetensors`) repo-wide.
