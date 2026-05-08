# Awesome AI By Bharat (AABB)

This page contains a hand-curated list of Artificial Intelligence technologies built by India.

While *AI-For-Bharat* is valuable, it doesn't quite encompass the spirit of self-reliance that *AI-By-Bharat* entails. Who truly owns the AI technology when we use cloud services or open-source frameworks? It's a complex matter to define, but imagine a litmus test: if there were sanctions against India, would your AI-powered product or technology still function?

Having said that, this repo also has room for AI-For-Bharat efforts. Go ahead and suggest those as well.

A similar collection can be found at [The Indic NLP Catalog](https://github.com/AI4Bharat/indicnlp_catalog).

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/yogeshhk/Awesome-AI-By-Bharat/blob/master/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

---

# Table of Contents

- [Projects](#projects)
- [Papers](#papers)
- [Tools & Code](#tools--code)
- [APIs](#apis)
- [Datasets](#datasets)
- [Models](#models)
- [Educational](#educational)
  - [Courses](#courses)
  - [Tutorials](#tutorials)
- [Videos](#videos)
- [Books](#books)
- [Communities](#communities)
- [How to Contribute](#how-to-contribute)

---

## Projects

- **Language AI**
  - [Bhashini](https://bhashini.gov.in/en/) : Government platform enabling an inclusive future for all Indians by creating language datasets and AI technologies for Indian language solutions.
  - [Bhasha Daan](https://bhashini.gov.in/bhashadaan/en/home) : Crowdsourcing arm of Bhashini — anyone can contribute audio-to-text, image-to-text, or translation data.
  - [AI4Bharat](https://ai4bharat.iitm.ac.in/) : IIT Madras initiative focused on building open-source language AI for Indian languages — datasets, models, and applications.
  - [BharatGPT by CoRover](https://corover.ai/bharatgpt/) : Supports 12+ Indian languages and 120+ foreign languages via text, voice, and video.
  - [Sarvam AI](https://www.sarvam.ai/) : Indian AI startup building full-stack language AI for Indian languages — LLMs, ASR, TTS, and translation API platform.
  - [Krutrim by Ola](https://www.olakrutrim.com/) : India's first AI unicorn; sovereign AI cloud stack with LLM trained on 2 trillion tokens understanding 22 Indian languages.
  - [Hanooman](https://hanooman.ai/) : Multimodal LLM (text, speech, vision) supporting 98 languages including 12 Indian languages, backed by Reliance Industries.

- **Government AI Initiatives**
  - [IndiaAI Mission](https://indiaai.gov.in/) : Rs 10,372 crore ($1.2B) national program with 7 pillars — compute capacity (10,000+ GPUs), Innovation Centre for indigenous LMMs, dataset platform, startup risk capital, talent development, safe AI, and sustainable development.
  - [AIRAWAT Supercomputer](https://indiaai.gov.in/article/airawat-a-landmark-in-india-s-ai-supercomputing-journey) : India's national AI supercomputing platform at C-DAC Pune; 200+ petaflop capacity, ranked in the global Top500 list (ISC 2023).
  - [AIKosh](https://aikosh.indiaai.gov.in/) : National AI model and dataset repository under the IndiaAI Mission.
  - [AI for India 2.0](https://www.guvi.in/ai-for-india/) : Skill development initiative from the Government of India via GUVI.

- **Healthcare**
  - [Niramai](https://www.niramai.com/) : Startup providing affordable AI-powered breast cancer screening at clinics in rural India, addressing shortage of radiologists.

- **Agriculture**
  - [CropIn](https://www.cropin.com/) : Intelligent, self-evolving AI system delivering future-ready farming solutions across the agricultural sector.

- **Autonomous Driving**
  - [Swaayatt Robots](http://www.swaayattrobots.com/) : Making connected autonomous driving technology accessible, affordable, and available to everyone.

---

## Papers

- **Research**
  - [AI4Bharat Publications](https://ai4bharat.iitm.ac.in/publications) : Peer-reviewed papers covering Indic NLP, ASR, MT, and multilingual AI.
  - [Airavata: Introducing Hindi Instruction-tuned LLM (2024)](https://arxiv.org/abs/2401.15006) : Paper on Airavata, first open Hindi instruction-tuned model.
  - [IndicTrans2: Towards High-Quality and Accessible Machine Translation (2023)](https://arxiv.org/abs/2305.16307) : Open-source MT covering all 22 scheduled Indian languages.

- **Articles**
  - [How is the Government creating an AI ecosystem in India?](https://www.bennett.edu.in/media-center/blog/how-is-the-government-creating-an-ai-ecosystem-in-india/)
  - [India's Frugal AI Strategy (2025)](https://restofworld.org/2026/india-frugal-ai-sarvam-krutrim-sovereign/) : How Indian AI companies like Sarvam and Krutrim are building sovereign AI.

---

## Tools & Code

| Name | Description | URL |
| :--- | :---: | :---: |
| **AI4Bharat** | Open-source AI for Bharat — all code, datasets, and models. | [GitHub](https://github.com/AI4Bharat) |
| **AI4Bharat Tools** | Web-based tools for ASR, MT, TTS, and transliteration in Indian languages. | [AI4Bharat](https://ai4bharat.iitm.ac.in/tools) |
| **Sarvam AI API** | Full-stack Indic GenAI API: chat, ASR (Saaras), translation (22 languages), TTS. | [Docs](https://docs.sarvam.ai/) |
| **Jugalbandi** | WhatsApp chatbot for rural India using AI4Bharat ASR + Bhashini + Azure OpenAI; covers 171 government schemes in 10 Indian languages. | [GitHub](https://github.com/microsoft/Jugalbandi) |
| **Krutrim AI Labs** | Developer platform for Krutrim LLM APIs and tools. | [AI Labs](https://ai-labs.olakrutrim.com/) |
| **IndicXlit** | Transformer-based multilingual transliteration tool for 21 Indian languages (Roman ↔ native script). | [GitHub](https://github.com/AI4Bharat/IndicXlit) |
| **IndicTrans2** | Open-source MT model covering all 22 scheduled Indian languages including low-resource ones. | [GitHub](https://github.com/AI4Bharat/IndicTrans2) |

---

## APIs

| Name | Description | URL | Access |
| :--- | :---: | :---: | :---: |
| **Sarvam AI** | Indic LLM, ASR, TTS, and translation API platform | [Dashboard](https://dashboard.sarvam.ai/) | Free tier + Paid |
| **Bhashini** | Government-backed speech and translation APIs for Indian languages | [Bhashini](https://bhashini.gov.in/en/) | Open / Free |
| **Krutrim** | LLM API with 22-language understanding and 10-language generation | [Krutrim](https://www.olakrutrim.com/) | Paid |

---

## Datasets

| Name | Description | URL |
| :--- | :---: | :---: |
| **IndicCorp** | Large monolingual corpora for 11 Indian languages and Indian English — 8.5 billion words from news sources. | [HuggingFace](https://huggingface.co/datasets/ai4bharat/IndicCorp) |
| **IndicCorp v2** | Largest Indic text collection — 20.9B tokens across 23 Indic languages plus 6.5B tokens of Indian English. | [GitHub](https://github.com/AI4Bharat/IndicBERT/tree/main#indiccorp-v2) |
| **Sangraha** | Highest-quality cleaned Indic pretraining corpus — 251B tokens across 22 languages from curated web, OCR, and transcriptions. | [HuggingFace](https://huggingface.co/datasets/ai4bharat/sangraha) |
| **IndicNLG Suite** | Training and evaluation datasets for 5 language generation tasks across 11 Indic languages. | [GitHub](https://github.com/AI4Bharat/IndicNLG) |
| **BPCC (Bharat Parallel Corpus Collection)** | ~230 million bitext pairs combining human-labelled and automatically mined parallel data. | [GitHub](https://github.com/AI4Bharat/IndicTrans2) |
| **IndicXTREME** | 9-task NLU benchmark across 18 Indic languages for evaluating language models. | [GitHub](https://github.com/AI4Bharat/IndicXTREME) |
| **Bhasha-Abhijnaanam** | Language identification test set for native-script and romanized text spanning 22 Indic languages. | [GitHub](https://github.com/AI4Bharat/Bhasha-Abhijnaanam) |
| **Naamapadam** | Largest public NER dataset for 11 major Indian languages — 400k+ sentences with Person, Location, Organization tags. | [GitHub](https://github.com/AI4Bharat/Naamapadam) |
| **Samanantar** | 49.6M English–Indic sentence pairs for 11 Indian languages — previously the largest parallel corpus. | [GitHub](https://github.com/AI4Bharat/Samanantar) |
| **Aksharantar** | Largest public transliteration dataset — 26M pairs across 21 Indic languages. | [GitHub](https://github.com/AI4Bharat/Aksharantar) |
| **Shrutilipi** | Labelled ASR corpus mining parallel audio-text pairs from All India Radio bulletins for 12 Indian languages. | [GitHub](https://github.com/AI4Bharat/Shrutilipi) |
| **IndicSUPERB** | 6-task speech language understanding benchmark across 12 Indian languages (ASR, speaker verification, keyword spotting, etc.). | [GitHub](https://github.com/AI4Bharat/IndicSUPERB) |
| **Dhwani** | Unlabelled ASR corpus from YouTube and News On AIR — raw audio across 40 Indian languages. | [GitHub](https://github.com/AI4Bharat/Dhwani) |
| **IndicGLUE** | NLU benchmark for Indian languages covering a wide variety of tasks. | [GitHub](https://github.com/AI4Bharat/IndicGLUE) |
| **IndicLLMSuite / Anudesh** | 74.7M prompt-response pairs in 20 Indian languages for instruction tuning; Anudesh is the crowd-sourced Hindi prompt subset. | [GitHub](https://github.com/AI4Bharat/IndicLLMSuite) |

---

## Models

| Name | Description | URL |
| :--- | :---: | :---: |
| **OpenHathi-7B** | First open Hindi-English bilingual LLM based on Llama 2; expanded Devanagari tokenizer by Sarvam AI. | [HuggingFace](https://huggingface.co/sarvamai/OpenHathi-7B-Hi-v0.1-Base) |
| **Sarvam-1** | 2B-parameter LLM optimized for 10 Indian languages — first purpose-built Indian language foundation model. | [HuggingFace](https://huggingface.co/sarvamai/sarvam-1) |
| **Airavata** | Open-source Hindi instruction-tuned LLM fine-tuned from OpenHathi using curated Indic instruction datasets. | [HuggingFace](https://huggingface.co/ai4bharat/airavata) |
| **IndicBART** | Multilingual sequence-to-sequence model trained on IndicCorp for 11 Indian languages + English. | [HuggingFace](https://huggingface.co/ai4bharat/IndicBART) |
| **IndicTrans2** | Open-source transformer MT model for all 22 scheduled Indian languages, including low-resource (Kashmiri, Manipuri, Sindhi). | [GitHub](https://github.com/AI4Bharat/IndicTrans2) |
| **IndicBERT** | Multilingual ALBERT model trained on 12 major Indian languages — lightweight yet state-of-the-art on several tasks. | [HuggingFace](https://huggingface.co/ai4bharat/indic-bert) |
| **IndicNER** | Named entity recognition model fine-tuned on 11 Indian languages over millions of sentences. | [HuggingFace](https://huggingface.co/ai4bharat/IndicNER) |
| **KooBERT** | Masked language model trained on data from Koo, the multilingual Indian micro-blogging platform. | [HuggingFace](https://huggingface.co/KooAI/KooBERT) |
| **Indic Speech-to-Text (Conformer)** | 30M-parameter ASR conformer model for real-time speech recognition in Indian languages. | [Models Portal](https://models.ai4bharat.org/) |
| **Indic Text-to-Speech** | Multispeaker TTS models for Indian languages. | [Models Portal](https://models.ai4bharat.org/#/tts) |
| **Indic Speech-to-Speech** | Combines ASR + NMT + TTS for cross-language speech conversion among Indian languages. | [Models Portal](https://models.ai4bharat.org/#/sts) |
| **Indic Generation Models** | Headline generation, summarization, paraphrasing models fine-tuned on IndicNLG Suite. | [GitHub](https://github.com/AI4Bharat/IndicNLG) |

---

## Educational

### Courses

- [AI for India 2.0](https://www.guvi.in/ai-for-india/) — Government of India AI skill development initiative via GUVI (free).
- [AI4Bharat Workshops & Tutorials](https://ai4bharat.iitm.ac.in/) — Materials from IIT Madras covering Indic NLP, ASR, and MT research.

### Tutorials

- [Getting Started with Bhashini APIs](https://bhashini.gov.in/en/) — Official documentation for integrating Indian language APIs.
- [Sarvam AI API Quickstart](https://docs.sarvam.ai/) — Guides for using Sarvam's Indic LLM, ASR, TTS, and translation APIs.
- [IndicTrans2 Usage Guide](https://github.com/AI4Bharat/IndicTrans2) — Step-by-step instructions for running high-quality Indian language machine translation.

---

## Videos

- [AI for All: How India is carving its own path in the global AI race](https://oecd.ai/fr/wonk/india) — OECD.AI analysis of India's national AI strategy.
- [AI4Bharat YouTube Channel](https://www.youtube.com/@ai4bharat) — Talks, demos, and tutorials from the AI4Bharat research group.

---

## Books

- [Artificial Intelligence for Everyone](https://www.nasscom.in/ai/) — NASSCOM reports and whitepapers on India's AI landscape (free PDFs).

---

## Communities

- [IndiaAI Portal](https://indiaai.gov.in/) : Official government knowledge hub for India's AI ecosystem — news, research, model registry, and dataset platform.
- [NASSCOM AI](https://nasscom.in/ai/) : India's industry AI hub tracking the GenAI startup landscape (170+ startups); publishes India GenAI Landscape reports.
- [AI4Bharat](https://github.com/AI4Bharat) : Open-source research community at IIT Madras — contributions welcome.

---

# How to Contribute

We welcome contributions to this list! The goal is to document AI technologies built by India or specifically for Indian language needs — models, datasets, tools, research, and applications.

Before contributing, please review our [contribution guidelines](contributing.md). Contributions should align with the AI-By-Bharat spirit: resources that strengthen India's AI self-reliance, or that serve Indian languages and communities in ways not covered by generic global solutions.
