# Fine-Tuning Small Language Models (SLMs)

Notebooks and scripts for fine-tuning small language models (Gemma, Llama, Mistral, Phi, SmolLM) with LoRA, SFT, DPO and Unsloth, plus RAG-versus-fine-tuning notebooks.

```bash
conda env create -f environment.yml
```

Fine-tuning needs a GPU for anything beyond small models; `cuda_test.py` checks that CUDA is visible. When resources are short, a RAG workflow on a base model is often enough before fine-tuning is considered. Working notes are in `Notes.md`.
