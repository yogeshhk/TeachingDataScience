# Course Catalog

A full map of everything in this repo, organized the way it's built: **Courses** are assembled
from **Workshops**, which are assembled from **Seminars** (see the README for what each tier means).
Every driver below is a `.tex` file in `LaTeX/` — open it directly, or compile it with `texify`
(see the README's Getting Started section).

Where a seminar is only reachable as part of a workshop (it has no standalone driver of its own),
it's listed as plain text rather than a link.

---

## Courses

### Machine Learning
Driver: [`Main_Course_MachineLearning_Presentation.tex`](LaTeX/Main_Course_MachineLearning_Presentation.tex)

- [Python for ML](LaTeX/Main_Workshop_Python_Basic_Presentation.tex) — [Intro](LaTeX/Main_Seminar_Python_Basic_Intro_Presentation.tex), [Constructs](LaTeX/Main_Seminar_Python_Basic_Constructs_Presentation.tex), [Procedures](LaTeX/Main_Seminar_Python_Basic_Procedures_Presentation.tex), [OOP](LaTeX/Main_Seminar_Python_Basic_OOP_Presentation.tex), [IO & Libraries](LaTeX/Main_Seminar_Python_Basic_IOLibraries_Presentation.tex), [Closures](LaTeX/Main_Seminar_Python_Basic_Closure_Presentation.tex)
- [Foundations](LaTeX/Main_Workshop_ML_Foundations_Presentation.tex) — [Intro](LaTeX/Main_Seminar_ML_Intro_Presentation.tex), [Data Prep](LaTeX/Main_Seminar_ML_DataPrep_Presentation.tex)
- [Regression](LaTeX/Main_Workshop_ML_Regression_Presentation.tex) — [Regression](LaTeX/Main_Seminar_ML_Regression_Presentation.tex)
- [Tree-Based & Ensemble](LaTeX/Main_Workshop_ML_TreeBased_Presentation.tex) — [Decision Trees](LaTeX/Main_Seminar_ML_DecisionTree_Presentation.tex), [Ensemble](LaTeX/Main_Seminar_ML_Ensemble_Presentation.tex)
- [Supervised II](LaTeX/Main_Workshop_ML_SupervisedII_Presentation.tex) — [KNN](LaTeX/Main_Seminar_ML_KNN_Presentation.tex), [SVM & Naive Bayes](LaTeX/Main_Seminar_ML_SVM_NB_Presentation.tex)
- [Unsupervised & Deployment](LaTeX/Main_Workshop_ML_Unsupervised_Presentation.tex) — [Clustering](LaTeX/Main_Seminar_ML_Clustering_Presentation.tex), [Dimensionality Reduction](LaTeX/Main_Seminar_ML_DimReduction_Presentation.tex), [Deployment](LaTeX/Main_Seminar_ML_Deployment_Presentation.tex)

There's also a standalone bundle of just the 5 ML-specific workshops (no Python, no demos):
[`Main_Workshop_MachineLearning_Presentation.tex`](LaTeX/Main_Workshop_MachineLearning_Presentation.tex).

### Python
Driver: [`Main_Course_Python_Presentation.tex`](LaTeX/Main_Course_Python_Presentation.tex)

- [Basic Python](LaTeX/Main_Workshop_Python_Basic_Presentation.tex) — same 6 seminars as above (Intro, Constructs, Procedures, OOP, IO & Libraries, Closures)
- [Advanced Python](LaTeX/Main_Workshop_Python_Advanced_Presentation.tex) — [OOP & Iteration](LaTeX/Main_Seminar_Python_Advanced_OOPIteration_Presentation.tex), [Functions & OS](LaTeX/Main_Seminar_Python_Advanced_FunctionsOS_Presentation.tex), [Strings & Web](LaTeX/Main_Seminar_Python_Advanced_StringsWeb_Presentation.tex), [Data Libraries](LaTeX/Main_Seminar_Python_Advanced_DataLibs_Presentation.tex), [Visualization](LaTeX/Main_Seminar_Python_Advanced_Visualization_Presentation.tex), [Problems](LaTeX/Main_Seminar_Python_Advanced_Problems_Presentation.tex)

### Maths for ML
Driver: [`Main_Course_MathsML_Presentation.tex`](LaTeX/Main_Course_MathsML_Presentation.tex)

- [Basics](LaTeX/Main_Workshop_MathsML_Basics_Presentation.tex) — [Numbers & Equations](LaTeX/Main_Seminar_MathsML_Basics_NumbersEquations_Presentation.tex), [Sets & Proofs](LaTeX/Main_Seminar_MathsML_Basics_SetsProofs_Presentation.tex)
- [Linear Algebra](LaTeX/Main_Workshop_MathsML_LinearAlgebra_Presentation.tex) — [Vectors](LaTeX/Main_Seminar_MathsML_LinearAlgebra_Vectors_Presentation.tex), [Matrices](LaTeX/Main_Seminar_MathsML_LinearAlgebra_Matrices_Presentation.tex)
- [Calculus](LaTeX/Main_Workshop_MathsML_Calculus_Presentation.tex) — [Functions & Limits](LaTeX/Main_Seminar_MathsML_Calculus_FunctionsLimits_Presentation.tex), [Derivatives & Optimization](LaTeX/Main_Seminar_MathsML_Calculus_DerivativesOptimization_Presentation.tex)
- [Statistics](LaTeX/Main_Workshop_MathsML_Statistics_Presentation.tex) — [Probability Foundations](LaTeX/Main_Seminar_MathsML_Statistics_ProbabilityFoundations_Presentation.tex), [Random Distributions](LaTeX/Main_Seminar_MathsML_Statistics_RandomDistributions_Presentation.tex), [Central Tendency & Spread](LaTeX/Main_Seminar_MathsML_Statistics_CentralTendencySpread_Presentation.tex), [Distributions & Expected Value](LaTeX/Main_Seminar_MathsML_Statistics_DistributionsExpectedValue_Presentation.tex), [Hypothesis Testing](LaTeX/Main_Seminar_MathsML_Statistics_HypothesisTesting_Presentation.tex), [Tests in Practice](LaTeX/Main_Seminar_MathsML_Statistics_TestsPractice_Presentation.tex)

### Deep Learning
Driver: [`Main_Course_DeepLearning_Presentation.tex`](LaTeX/Main_Course_DeepLearning_Presentation.tex)

- [Deep Learning Workshop](LaTeX/Main_Workshop_DL_Presentation.tex) — Foundations (workshop-only), TensorFlow core (workshop-only), [PyTorch](LaTeX/Main_Seminar_DL_Pytorch_Presentation.tex)
- Course-specific extras (no dedicated workshop): AI/tech intro, Python syntax primer, self-organizing maps, autoencoders

### Generative AI
Driver: [`Main_Course_GenerativeAI_Presentation.tex`](LaTeX/Main_Course_GenerativeAI_Presentation.tex)

- [Natural Language Processing](LaTeX/Main_Workshop_NLP_Presentation.tex) — NLP basics, POS & NER (both workshop-only), [NLP + ML](LaTeX/Main_Seminar_NLP_ML_Presentation.tex), NLP advanced (workshop-only)
- [Deep NLP](LaTeX/Main_Workshop_NLP_Deep_Presentation.tex) — word embeddings, word2vec, text generation (topic files, not a seminar chain)
- [LLMs](LaTeX/Main_Workshop_LLM_Presentation.tex) — [Intro](LaTeX/Main_Seminar_LLM_Intro_Presentation.tex), [Transformers](LaTeX/Main_Seminar_LLM_Transformers_Presentation.tex), [ChatGPT From Zero (Short)](LaTeX/Main_Seminar_LLM_ChatGPT_FromZeroShort_Presentation.tex), [Evaluation](LaTeX/Main_Seminar_LLM_Evaluation_Presentation.tex), [Prompt Engineering](LaTeX/Main_Seminar_LLM_PromptEngg_Presentation.tex), [Fine-Tuning](LaTeX/Main_Seminar_LLM_FineTuning_Presentation.tex), [RAG](LaTeX/Main_Seminar_LLM_RAG_Presentation.tex), [Agents](LaTeX/Main_Seminar_LLM_Agents_Presentation.tex), [Production](LaTeX/Main_Seminar_LLM_Production_Presentation.tex), [Reasoning](LaTeX/Main_Seminar_LLM_Reasoning_Presentation.tex), [LangChain](LaTeX/Main_Seminar_LLM_LangChain_Presentation.tex), [LlamaIndex](LaTeX/Main_Seminar_LLM_LlamaIndex_Presentation.tex), LLM applications (workshop-only)
- Course-specific extras: ChatGPT applications (BDO/IMI/HR/journalism), advanced RAG, LangGraph deep-dive

---

## Standalone Workshops

These aren't part of any of the 5 courses above — each is a complete, independent session.

| Workshop | Driver | Covers |
|---|---|---|
| AI (broad survey) | [`Main_Workshop_AI_Presentation.tex`](LaTeX/Main_Workshop_AI_Presentation.tex) | Python → AI → Data Analytics → ML → DL → NLP → Word Embeddings → LLM Intro → Agents → Career in Data Science, all via existing standalone seminars |
| RAG (core) | [`Main_Workshop_LLM_RAG_Presentation.tex`](LaTeX/Main_Workshop_LLM_RAG_Presentation.tex) | [RAG](LaTeX/Main_Seminar_LLM_RAG_Presentation.tex), [LangChain](LaTeX/Main_Seminar_LLM_LangChain_Presentation.tex), [Evaluation](LaTeX/Main_Seminar_LLM_Evaluation_Presentation.tex), advanced RAG (workshop-only) |
| RAG to Riches | [`Main_Workshop_RAGToRiches_Presentation.tex`](LaTeX/Main_Workshop_RAGToRiches_Presentation.tex) | End-to-end journey: Python → ML → NLP → Word Embeddings → RAG → Docling → LangChain → Evaluation |
| RAG2Riches (bootcamp) | [`Main_Workshop_LLM_RAG2Riches_Presentation.tex`](LaTeX/Main_Workshop_LLM_RAG2Riches_Presentation.tex) | A full 3-week (or 12-week part-time) day-by-day bootcamp curriculum: Python → DSA → System Design → NLP → ML → LLMs → RAG → Production, with daily projects and a capstone |
| LLM Agents | [`Main_Workshop_LLM_Agents_Presentation.tex`](LaTeX/Main_Workshop_LLM_Agents_Presentation.tex) | Agent concepts, MCP, eval, [Agents seminar](LaTeX/Main_Seminar_LLM_Agents_Presentation.tex) |
| Document Parsing (Docling) | [`Main_Workshop_LLM_Docling_Presentation.tex`](LaTeX/Main_Workshop_LLM_Docling_Presentation.tex) | Docling architecture, multimodal parsing, resume parsing, production/evals |
| LangChain | [`Main_Workshop_LLM_LangChain_Presentation.tex`](LaTeX/Main_Workshop_LLM_LangChain_Presentation.tex) | Framework, implementation, what's new |
| LangGraph | [`Main_Workshop_LLM_LangGraph_Presentation.tex`](LaTeX/Main_Workshop_LLM_LangGraph_Presentation.tex) | Intro, advanced, implementation |
| Transformers | [`Main_Workshop_LLM_Transformers_Presentation.tex`](LaTeX/Main_Workshop_LLM_Transformers_Presentation.tex) | History (through word2vec, seq2seq/attention) → architecture → pretraining → BERT → HuggingFace → applications |
| Graph Databases | [`Main_Workshop_Graph_Database_Presentation.tex`](LaTeX/Main_Workshop_Graph_Database_Presentation.tex) | Graph basics, Neo4j, graph data science (full workshop versions) |
| Geometric Deep Learning | [`Main_Workshop_Graph_GeometricDeepLearning_Presentation.tex`](LaTeX/Main_Workshop_Graph_GeometricDeepLearning_Presentation.tex) | [GDL](LaTeX/Main_Seminar_Graph_GeometricDeepLearning_Presentation.tex), [GNNs](LaTeX/Main_Seminar_Graph_NeuralNetworks_Presentation.tex), PyTorch Geometric, knowledge graphs, category theory (workshop-only) |
| Knowledge Graphs | [`Main_Workshop_Graph_KnowledgeGraph_Presentation.tex`](LaTeX/Main_Workshop_Graph_KnowledgeGraph_Presentation.tex) | Graph algorithms, KG semantics, KG + LLMs, implementations |
| Chatbots (Rasa) | [`Main_Workshop_NLP_Chatbot_Rasa_Presentation.tex`](LaTeX/Main_Workshop_NLP_Chatbot_Rasa_Presentation.tex) | Chatbot design, Rasa install/concepts/theory, slots/forms/deployment, a full IPL-bot walkthrough |
| spaCy | [`Main_Workshop_NLP_SpaCy_Presentation.tex`](LaTeX/Main_Workshop_NLP_SpaCy_Presentation.tex) | Pipelines, POS/NER, classification, med7 |
| Data Analytics | [`Main_Workshop_Data_Analytics_Presentation.tex`](LaTeX/Main_Workshop_Data_Analytics_Presentation.tex) | Data concepts, dimensionality, pandas, prep, exploration, visualization, with demo/assignment case studies |
| Reinforcement Learning | [`Main_Workshop_ML_ReinforcementLearning_Presentation.tex`](LaTeX/Main_Workshop_ML_ReinforcementLearning_Presentation.tex) | MDPs, Q-learning, deep Q-learning, RLlib, OpenAI Gym, tic-tac-toe implementation |
| Blockchain | [`Main_Workshop_Tech_Blockchain_Presentation.tex`](LaTeX/Main_Workshop_Tech_Blockchain_Presentation.tex) | Bitcoin, cryptocurrency implementation, Ethereum, smart contracts |
| Software Engineering | [`Main_Workshop_Tech_Software_Presentation.tex`](LaTeX/Main_Workshop_Tech_Software_Presentation.tex) | Complexity, data structures (arrays/queues/maps/trees/graphs), algorithms (recursion/search/sort/dynamic programming), system design, LeetCode practice |
| All-in-one ML bundle | [`Main_Workshop_MachineLearning_Presentation.tex`](LaTeX/Main_Workshop_MachineLearning_Presentation.tex) | All 5 ML-specific workshops chained together (no Python, no demos) |

---

## Other Notable Standalone Seminars

Beyond the courses and workshops above, dozens of independent 1-hour seminars exist. A sample by theme:

- **AI overviews for different audiences**: [General](LaTeX/Main_Seminar_AI_Presentation.tex), [For Educators](LaTeX/Main_Seminar_AI_for_Educators_Presentation.tex), [For Kids](LaTeX/Main_Seminar_AI_for_Kids_Presentation.tex), [For Non-Tech](LaTeX/Main_Seminar_AI_for_NonTech_Presentation.tex), [For Biz Leaders](LaTeX/Main_Seminar_AI_BizLeaders_Presentation.tex), [For Tech Leaders](LaTeX/Main_Seminar_AI_TechLeaders_Presentation.tex), [AI × ML](LaTeX/Main_Seminar_AI-ML_Presentation.tex)
- **ChatGPT / LLM intros at different depths**: [From Zero](LaTeX/Main_Seminar_LLM_ChatGPT_FromZero_Presentation.tex), [From Zero (Short)](LaTeX/Main_Seminar_LLM_ChatGPT_FromZeroShort_Presentation.tex), [Mechanics](LaTeX/Main_Seminar_LLM_ChatGPT_Mech_Presentation.tex), [Non-Tech](LaTeX/Main_Seminar_LLM_ChatGPT_NonTech_Presentation.tex), [Tech (Short)](LaTeX/Main_Seminar_LLM_ChatGPT_TechShort_Presentation.tex), [General](LaTeX/Main_Seminar_LLM_ChatGPT_Presentation.tex)
- **Deep dives / overviews**: [Data Concepts](LaTeX/Main_Seminar_Data_Concepts_Overview_Presentation.tex), [DL Foundations](LaTeX/Main_Seminar_DL_Foundations_Overview_Presentation.tex), [Python Syntax](LaTeX/Main_Seminar_Python_Syntax_Overview_Presentation.tex), [Deep NLP](LaTeX/Main_Seminar_NLP_DNLP_Overview_Presentation.tex), [Word Embeddings](LaTeX/Main_Seminar_NLP_WordEmbeddings_Presentation.tex)
- **Deep Learning platforms**: [TensorFlow](LaTeX/Main_Seminar_DL_Tensorflow_Presentation.tex), [PyTorch](LaTeX/Main_Seminar_DL_Pytorch_Presentation.tex), [Data + TensorFlow](LaTeX/Main_Seminar_DL_Data_TensorFlow_Presentation.tex), [Swift for TensorFlow](LaTeX/Main_Seminar_DL_Swift4TensorFlow_Presentation.tex), [Satellite Imagery](LaTeX/Main_Seminar_DL_SatelliteImagery_Presentation.tex)
- **Google Cloud Platform**: [Overview](LaTeX/Main_Seminar_GCP_Presentation.tex), [GenAI](LaTeX/Main_Seminar_GCP_GenAI_Presentation.tex), [Vertex AI](LaTeX/Main_Seminar_GCP_VertexAI_Presentation.tex), [Document AI](LaTeX/Main_Seminar_GCP_DocAI_Presentation.tex)
- **Graph topics**: [Neo4j](LaTeX/Main_Seminar_Graph_Neo4j_Presentation.tex), [Graph Data Science](LaTeX/Main_Seminar_Graph_DataScience_Presentation.tex), [Graph + NLP](LaTeX/Main_Seminar_Graph_NLP_Presentation.tex), [Graph RAG](LaTeX/Main_Seminar_Graph_RAG_Presentation.tex), [Knowledge Graphs](LaTeX/Main_Seminar_Graph_KnowledgeGraphs_Presentation.tex), [LLM + Knowledge Graphs](LaTeX/Main_Seminar_LLM_KnowledgeGraphs_Presentation.tex)
- **Applied ML**: [Explainable AI](LaTeX/Main_Seminar_ML_ExplainableAI_Presentation.tex), [Matrix Profile](LaTeX/Main_Seminar_ML_MatrixProfile_Presentation.tex), [Reinforcement Learning](LaTeX/Main_Seminar_ML_ReinforcementLearning_Presentation.tex), [Text Mining](LaTeX/Main_Seminar_NLP_TextMining_Presentation.tex), [SQL + RAG](LaTeX/Main_Seminar_LLM_SQL_RAG_Presentation.tex), [Seq2Seq](LaTeX/Main_Seminar_LLM_SeqSeg_Presentation.tex)
- **Career & meta**: [Career in Data Science](LaTeX/Main_Seminar_Tech_CareerInDataScience_Presentation.tex) ([Short version](LaTeX/Main_Seminar_Tech_CareerInDataScience_Short_Presentation.tex)), [Mentoring](LaTeX/Main_Seminar_Tech_Mentoring_Presentation.tex), [Gartner Hype Cycles](LaTeX/Main_Seminar_Tech_HypeCycles_Gartner_Presentation.tex), [LaTeX for Research](LaTeX/Main_Seminar_Tech_LaTeX_Research_Presentation.tex), [Claude Code](LaTeX/Main_Seminar_AI_ClaudeCode_Presentation.tex)

This list is a sample, not exhaustive — every `Main_Seminar_*_Presentation.tex` in `LaTeX/` is an independent, compilable session; browsing that naming pattern directly is the fastest way to find something not listed here.
