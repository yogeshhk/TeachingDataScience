# CoEP Mech Eng AI-ML Syllabi Analysis

## Executive Summary

This note analyzes the three AI/ML-related courses in the COEP Mechanical Engineering curriculum:

* TY Machine Learning (PEC-4-8, 3 credits),  
* TY Deep Learning (VI-4.12, 4 credits)  
* Final Year course "AI and ML for Mechanical Engineers" (3 credits).

The three were evidently drafted independently rather than as a single progression. The result is heavy content duplication across TY and Final Year, an internally inconsistent difficulty gradient within Final Year, and an unstated math/programming prerequisite that a real subset of students (those entering via diploma lateral entry) do not have.

Ten critical mismatches are listed below, followed by twelve recommendations. All recommendations work within the courses' existing credit and contact-hour allocations: no proposal here adds or removes credits from any course, only reorders or replaces content inside each course's own hour budget.

## Critical Mismatches

**1\. TY-ML and Final Year open with near-duplicate introductions.** TY-ML Unit 1 (AI history, AI vs. Data Science, cybernetics/symbolic/statistical approaches, supervised/unsupervised/RL as the three ML paradigms) is repeated almost point for point in Final Year's Introduction section, 7 hours after students already saw it in TY.

**2\. TY-ML contains a full Deep Learning unit that a separate, parallel TY-DL course covers in far more depth.** TY-ML Unit 5 (CNN, RNN, predictive maintenance) is a compressed preview of the entire TY-DL course, taken in the same academic year. Students effectively get two passes at the same material before either course is complete.

**3\. Reinforcement Learning appears in four places with no clear owner of depth.** TY-ML Unit 1 (as a basic paradigm), TY-ML Unit 6 (Advanced Topics), TY-DL Unit 5 (DQN, policy gradients), and Final Year (inverse RL, safe RL, PSRL). Final Year's RL treatment is conceptually more advanced than TY-DL's, but its own framing ("brute force, value function") reads more introductory than what TY-DL already taught, meaning depth does not increase monotonically across the sequence.

**4\. Core supervised algorithms are taught twice at nearly the same depth.** Decision trees, random forests, SVM, KNN, and logistic regression appear fully in TY-ML Unit 3 and again in Final Year's Classification & Regression section, with only Naive Bayes and SVR added. No progression in rigor, scale, or application is indicated.

**5\. Feature extraction and selection is taught twice with the harder material unflagged.** TY-ML Unit 2 and Final Year's Feature Extraction and Selection section cover nearly the same ground. Final Year does add wrapper methods (exhaustive, best-first, greedy search), which are genuinely new and harder, but nothing in the syllabus marks them as the new content, so they read as a repeat rather than an advance.

**6\. Unsupervised learning regresses at Final Year, and is miscategorized.** TY-ML has a dedicated unit covering hierarchical clustering, t-SNE, and anomaly detection. Final Year drops all three and reduces unsupervised learning to a single line, "K-Means," wedged inside a section titled Classification & Regression, which is a category error as much as a content gap.

**7\. RL is front-loaded into TY-ML's very first unit.** It is listed as a basic ML paradigm alongside supervised and unsupervised learning before students have seen supervised learning itself (Unit 3), then not touched again until Unit 6\.

**8\. No math or Python prerequisite is stated anywhere, in either syllabus.** PCA (TY-ML Unit 2), gradient descent and backpropagation (TY-DL Unit 1), and entropy/information gain (both TY-ML and Final Year) are all assumed from the first unit onward. This is a genuine open gap, not one covered by an existing bridge course, and it falls hardest on students who entered via diploma lateral entry with no formal programming or linear algebra/calculus background.

**9\. Final Year's internal difficulty gradient is incoherent.** "History of AI" sits in the same 3-credit, 45-hour course as Partially Supervised Reinforcement Learning and Inverse RL, genuinely advanced topics, with no scaffolding between them.

**10\. No Generative AI, NLP, attention/transformer, or LLM content exists anywhere across all three courses.** GANs and VAEs (taught twice, in TY-DL and again in Final Year) are the closest analogue, but neither course extends toward the subfield that is currently most industry-relevant.

Lower-priority, worth a footnote: TY-ML's CO4 explicitly requires hands-on implementation in Python, scikit-learn, and TensorFlow, but the laboratory evaluation columns (ISE, ESE) are blank. The course outcome and the assessment scheme do not match.

## Recommendations

**1\. Remove TY-ML Unit 5 (Deep Learning and Neural Networks) entirely.** Its content is redundant with the parallel TY-DL course. Repurpose its 8 hours within TY-ML for a Python and Math Refresher unit: array/vector and matrix operations (needed for PCA), basic derivatives and the chain rule (needed for gradient descent), and probability basics. This directly addresses mismatches 2 and 8 without changing TY-ML's total hours or credits.

**2\. Remove RL from TY-ML Unit 1's list of basic paradigms.** Keep only supervised and unsupervised learning as the two starting concepts; RL already has a home in Unit 6\. Addresses mismatch 7\.

**3\. Keep TY-ML Unit 6's RL coverage conceptual only** (what RL is, why it matters for control problems), explicitly deferring algorithmic depth (Q-learning, DQN, policy gradients) to TY-DL Unit 5\. Establishes TY-DL as RL's first real technical treatment.

**4\. Add an explicit prerequisite line to TY-DL's syllabus header** stating it assumes the Python/Math Refresher from TY-ML Recommendation 1\. Makes the dependency visible rather than implicit.

**5\. Establish TY-DL Unit 5 as the canonical first depth pass on RL** (DQN, policy gradients), and require Final Year to build on it explicitly rather than re-teach it. Addresses mismatch 3\.

**6\. Compress Final Year's Introduction section into a Recap and Bridge section.** Cut the near-duplicate AI-history and approaches content to roughly 1 to 2 hours (from 7), with explicit forward pointers to what is new in Final Year. Reuse the freed hours, within the same section, for Recommendation 10 below. Addresses mismatch 1\.

**7\. Cut the redundant re-teach in Final Year's Classification & Regression section.** Remove decision trees, random forests, SVM, and logistic regression as full re-teaches; keep only a one-line recap. Use the freed hours for the genuinely new content: Naive Bayes, SVR, and ensemble comparisons across the algorithms already known from TY-ML. Addresses mismatch 4\.

**8\. Move K-Means out of Classification & Regression into a properly labeled Unsupervised Learning (Advanced) section**, together with hierarchical clustering, t-SNE, and anomaly detection brought forward from TY-ML rather than dropped. Addresses mismatches 5 and 6\.

**9\. Compress Final Year's Deep & Reinforced Learning section to assume TY-DL's CNN, RNN, LSTM, GAN, and RL content is already known.** State this assumption explicitly, then use the freed hours to go deeper only on what is genuinely new to Final Year: deep belief networks, extreme learning machines, deep residual networks, inverse RL, safe RL, and PSRL. Addresses mismatches 3 and 9\.

**10\. Use the hours freed by Recommendation 6 to add explicit scaffolding before the advanced RL variants**, a short "why go beyond standard RL" motivation connecting safety and partial supervision to real control problems, rather than jumping straight from AI history to PSRL. Addresses mismatch 9\.

**11\. Replace Final Year's Advanced & Recent Techniques grab-bag with a focused NLP/GenAI arc, proposed as the strongest option among three considered.**

- *Proposed: NLP-based Generative AI.* Tokenization and embeddings, attention and transformers, sequence-to-sequence models, and a case study applying an LLM to engineering documentation or design-report generation. This directly closes mismatch 10, extends naturally from the GAN/VAE content students already have from TY-DL, and gives Final Year a distinct identity instead of overlapping TY-DL's territory.  
- *Alternative considered: Advanced Deep Learning / production topics* (model compression, serving at scale, MLOps for DL). Rejected as the lead choice because it extends TY-DL's existing deployment unit rather than filling a gap the curriculum currently has none of.  
- *Alternative considered: Applied ML and MLOps* (experiment tracking, CI/CD for models, monitoring). Rejected as the lead choice for the same reason: valuable, but an extension of existing content rather than new ground, and less differentiated from industry electives students may take elsewhere.

**12\. Add an explicit prerequisite line to both TY-ML's and Final Year's syllabus headers**: "Assumes basic Python programming, linear algebra (vectors, matrices), and introductory calculus and probability. Students without this background should complete \[bridge module, per Recommendation 1\] first." This makes the assumption identified in mismatch 8 auditable rather than implicit, and is a documentation-only change requiring no additional hours.

## Next Steps

Once these are reviewed and approved, the next task is to draft updated syllabi for TY-ML, TY-DL, and Final Year (proposed as an NLP/GenAI course) that implement the changes above, unit by unit, within each course's existing credit and hour allocation.

**Status (2026-08-28):** Recommendations above are approved by email from HoD. The three draft syllabi below implement them as much as possible. GenAI is brand new and more raw.

## Proposed Syllabus: TY Machine Learning Elective (Draft)

### Course Info

| Course Code | Course Name | L | T | P | S | Cr | MSE (Theory) | TA (Theory) | ESE (Theory) | ISE (Lab) | ESE (Lab) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| ME-xxxx | Machine Learning | 3 | 0 | 0 | 1 | 3 | 30 | 20 | 50 | 100 | \-- |

**Prerequisites:** Assumes basic familiarity with Python syntax and high-school-level mathematics. Unit 2, Python and Math Refresher for ML, consolidates the linear algebra, calculus, and probability foundations used through the rest of this course and carried forward into TY Deep Learning.

### Course Outcomes

Students who successfully complete this course will have demonstrated an ability to:

**CO1:** Explain the fundamental concepts, mathematical and programming foundations, and applications of machine learning in engineering and interdisciplinary contexts.  
**CO2:** Formulate supervised and unsupervised learning problems and apply appropriate algorithms for classification, regression, and clustering.  
**CO3:** Analyze model performance using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.  
**CO4:** Implement machine learning algorithms using programming tools and libraries such as Python, scikit-learn, and TensorFlow.  
**CO5:** Evaluate the ethical, societal, and security implications of machine learning systems in real-world applications.  
**CO6:** Design and optimize machine learning workflows for engineering problems involving data preprocessing, feature selection, and model tuning.

### Syllabus

| Units | Contents | Hrs. |
| :---- | :---- | :---- |
| 1 | **Introduction to AI & Machine Learning in Mechanical Engineering:** History and evolution of AI and ML, Differences between AI, ML, and Data Science. Importance of ML in engineering, Overview of ML approaches: Supervised, Unsupervised. Basic concepts: Reasoning, problem-solving, knowledge representation, planning, perception. | 6 |
| 2 | **Python and Math Refresher for ML:** Python and NumPy essentials: arrays, vectors, and matrix operations, Basic differentiation and the chain rule as the foundation for gradient descent, Probability basics: random variables, distributions, expectation. | 8 |
| 3 | **Feature Engineering and Data Preprocessing:** Importance of data quality and preprocessing, Techniques for feature extraction: Statistical features, Principal Component Analysis (PCA), Feature selection methods, Dimensionality reduction techniques | 7 |
| 4 | **Supervised Learning Algorithms:** Linear and logistic regression, Decision trees and random forests, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Model evaluation metrics: Accuracy, precision, recall, F1-score | 6 |
| 5 | **Unsupervised Learning and Clustering:** Clustering algorithms: K-Means, Hierarchical clustering, Dimensionality reduction: PCA, t-SNE, Anomaly detection techniques, Applications in mechanical systems: Fault detection, pattern recognition | 7 |
| 6 | **Reinforcement Learning and Applications in Mechanical Engineering:** Reinforcement learning: motivation and conceptual overview, what it is and why it matters for control problems (algorithmic depth covered in TY Deep Learning). Physics-informed machine learning models, Integration of ML with Computer-Aided Engineering (CAE) tools, Case studies: Digital twins, smart manufacturing, structural health monitoring. | 7 |

Total: 41 hours.

### Suggested Learning Resources

**Textbooks:**

1. Aurelien Geron, Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems, O'Reilly Media, Inc.  
2. Introduction to Machine Learning with Python by Andreas C. Muller & Sarah Guido  
3. Machine Learning For Dummies by John Paul Mueller & Luca Massaron

**Reference Books:**

1. Tom Mitchell, "Machine Learning," McGraw Hill Publication, ISBN: 0070428077, 9780070428072  
2. Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong, Mathematics for Machine Learning, Cambridge University Press (23 April 2020\)  
3. Jake VanderPlas, Python Data Science Handbook: Essential Tools for Working with Data, O'Reilly Media

### CO-PO Mapping

| CO Code | Mapped POs | Description and Justification |
| :---- | :---- | :---- |
| CO1 | PO1, PO2 | Students apply mathematical and engineering fundamentals to understand machine learning principles and algorithms. They analyze problem domains and identify suitable learning paradigms, supporting foundational and analytical competencies. |
| CO2 | PO2, PO3, PO4 | Learners formulate supervised and unsupervised learning problems and select appropriate algorithms. They design and implement models for classification, regression, and clustering, integrating problem analysis, solution design, and research-based methods. |
| CO3 | PO2, PO4 | Students evaluate model performance using statistical metrics and validation techniques. This involves analytical thinking and interpretation of experimental results, reinforcing problem analysis and research-based evaluation. |
| CO4 | PO5, PO12 | Learners implement machine learning algorithms using modern programming tools and libraries. This fosters proficiency in contemporary technologies and promotes lifelong learning through hands-on experimentation and tool adaptation. |
| CO5 | PO6, PO8 | Students assess the ethical, societal, and security implications of machine learning systems. They reflect on responsible AI practices, data privacy, and fairness, supporting societal awareness and professional ethics. |
| CO6 | PO3, PO5, PO12 | Learners design and optimize machine learning workflows, including data preprocessing and model tuning. This outcome strengthens solution design, tool usage, and continuous learning in evolving data-driven environments. |

## 

## Proposed Syllabus: TY Deep Learning Elective (Draft)

### Course Info

| Course Code | Course Name | L | T | P | S | Cr | MSE (Theory) | TA (Theory) | ESE (Theory) | ISE (Lab) | ESE (Lab) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| \<tbd\> | Deep Learning | 3 | 1 | 0 | 0 | 4 | 30 | 20 | 50 | \-- | \-- |

**Prerequisites:** Assumes the Python and Math Refresher for ML delivered in TY-ML Unit 2 (array/vector/matrix operations, differentiation and the chain rule, probability basics). Gradient descent and backpropagation in Unit 1 below build directly on that foundation.

### Course Outcomes

**CO1:** Explain the foundational concepts of neural networks, activation functions, and learning algorithms used in deep learning.  
**CO2:** Analyze the architecture and training dynamics of deep neural networks, including convolutional and recurrent models.  
**CO3:** Apply optimization techniques and regularization strategies to improve model performance and generalization.  
**CO4:** Design and implement deep learning models for classification, regression, and sequence modeling tasks using appropriate frameworks.  
**CO5:** Evaluate model performance using metrics, validation techniques, and interpretability tools to ensure robustness and fairness.  
**CO6:** Explore the role of deep learning in interdisciplinary applications such as computer vision, natural language processing, and healthcare analytics.

### Syllabus

| Unit | Contents | Hrs. |
| :---- | :---- | :---- |
| 1 | **Introduction to Deep Learning and Neural Networks:** Overview of Machine Learning and Deep Learning, Biological vs. Artificial Neural Networks, Activation Functions and Loss Functions, Optimization Techniques: Gradient Descent, Backpropagation, Overfitting, Underfitting, and Regularization Methods | 8 |
| 2 | **Convolutional Neural Networks (CNNs) for Mechanical Applications:** Convolution Operations and Pooling Layers, Architectures: LeNet, AlexNet, VGG, ResNet, Transfer Learning and Fine-Tuning, Applications: Defect Detection, Thermal Imaging Analysis | 7 |
| 3 | **Recurrent Neural Networks (RNNs) and Time-Series Analysis:** RNN Architectures: Vanilla RNNs, LSTM, GRU, Sequence Modeling and Prediction, Applications: Vibration Analysis, Predictive Maintenance, Challenges: Vanishing/Exploding Gradients, Sequence Length Handling | 7 |
| 4 | **Autoencoders and Generative Models:** Autoencoders: Structure and Training, Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Applications: Design Optimization, Anomaly Detection | 6 |
| 5 | **Reinforcement Learning in Mechanical Systems:** Fundamentals of Reinforcement Learning (RL), Deep Q-Networks (DQN) and Policy Gradient Methods, Applications: Robotics Control, Adaptive Systems, Simulation Environments: OpenAI Gym, Custom Simulators | 6 |
| 6 | **Integration and Deployment of Deep Learning Models:** Model Deployment Strategies: Edge Computing, Cloud Services, Tools: TensorFlow Lite, ONNX, Docker, Graph Neural Networks for CAD and Structural Analysis, Case Studies: Real-world Applications in Mechanical Engineering, Ethical Considerations and Model Interpretability | 8 |

Total: 42 hours.

### Suggested Learning Resources

**Textbooks:**

1. Aurelien Geron, Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems, O'Reilly Media, Inc.  
2. Ian Goodfellow, Yoshua Bengio, Aaron Courville, Deep Learning, MIT Press  
3. Charu C. Aggarwal, Neural Networks and Deep Learning, Springer Nature  
4. Francois Chollet, Deep Learning with Python, Manning, First Edition

**Weblinks:**

[https://onlinecourses.nptel.ac.in/noc20\_cs62/preview](https://onlinecourses.nptel.ac.in/noc20_cs62/preview)

### CO-PO Mapping

| CO | PO1 | PO2 | PO3 | PO4 | PO5 | PO6 | PO7 | PO8 | PO9 | PO10 | PO11 | PO12 | PSO1 | PSO2 | PSO3 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| CO1 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 |
| CO2 | 2 | 3 | 1 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 2 | 0 |
| CO3 | 2 | 2 | 2 | 2 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0 |
| CO4 | 1 | 2 | 3 | 2 | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0 |
| CO5 | 1 | 1 | 2 | 3 | 3 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 2 | 3 | 0 |
| CO6 | 1 | 1 | 0 | 1 | 1 | 3 | 2 | 1 | 0 | 0 | 1 | 3 | 2 | 3 | 0 |

## 

## Proposed Syllabus: Final Year Elective, Generative AI (Draft)

Proposed rename from "AI and ML for Mechanical Engineers." Content is refocused onto Generative AI and NLP, since TY-ML and TY-DL already cover classical ML and DL in depth; this course builds forward from that baseline instead of repeating it.

### Course Info

| Course Code | Course Name | L | T | P | S | TA | MSE | ESE (Theory) | ISE (Lab) | ESE (Lab) | Credits |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| \<tbd\> | Generative AI | 3 | 0 | 0 | 0 | 20 | 30 | 50 | \-- | \-- | 3 |

**Prerequisites:** Assumes basic Python programming and the machine learning and deep learning foundations from TY-ML and TY-DL, including supervised/unsupervised learning and CNN/RNN/GAN architectures.

### Course Outcomes

Students who successfully complete this course will have demonstrated an ability to:

1. Explain the foundations of generative AI, including how generative models differ from discriminative and traditional ML approaches, and their applications in engineering and manufacturing.  
2. Apply NLP techniques, tokenization, embeddings, named entity recognition, and text classification, to process and analyze engineering documents and technical literature.  
3. Apply transformer architecture and prompt engineering techniques to use large language models effectively for engineering tasks.  
4. Design and implement a Retrieval-Augmented Generation system, document parsing, chunking, embeddings, vector storage, and retrieval, for an engineering documentation use case.  
5. Design and implement agentic AI systems, reasoning, planning, tool use, and multi-agent orchestration, to automate engineering workflows.  
6. Evaluate the ethical, computational, and deployment considerations of generative AI systems in engineering practice.

### Syllabus

| Unit | Contents | Hrs. |
| :---- | :---- | :---- |
| 1 | **Foundations of Generative AI:** Evolution and capabilities of modern generative models, Generative vs. discriminative models, Comparison with GAN/VAE approaches, Industry applications and transformation opportunities in engineering and manufacturing, Computational resources and implementation considerations, Ethical frameworks and responsible AI deployment | 7 |
| 2 | **NLP and Text Processing for Engineers:** NLP fundamentals: tokenization, word embeddings (Word2Vec, GloVe), semantic similarity, NLP pipelines: Named Entity Recognition, part-of-speech tagging, parsing, Processing engineering documents and technical specifications, Information extraction from patents and research literature, Text classification for technical documentation management, Workshop: building a technical document analysis system | 9 |
| 3 | **Large Language Models in Action:** Transformer architecture and attention mechanism, Prompt engineering techniques for engineering tasks, Chain-of-thought reasoning for complex technical problems, Working with LLM APIs, evaluation frameworks for LLM performance in technical domains, Workshop: creating effective prompts for engineering applications | 9 |
| 4 | **Retrieval-Augmented Generation Systems:** RAG architecture: chunking, embeddings, retrieval, generation, and evaluation, Multi-format document parsing (PDF, DOCX, PPTX) for engineering documentation, LangChain framework: chains, vector stores, and retrieval systems, Vector databases for knowledge management, Workshop: developing a RAG system for engineering specifications and design-report generation | 9 |
| 5 | **Agentic AI Systems:** What are AI agents: reasoning, planning, memory, and tool use, Agent frameworks and design patterns (e.g., LangGraph, CrewAI, ReAct), Multi-agent orchestration for complex engineering workflows, Deployment strategies and integration with existing systems, Workshop: building a multi-step agentic assistant for an engineering task | 7 |

Total: 41 hours.

### Suggested Learning Resources

**Textbooks:**

1. Lewis Tunstall, Leandro von Werra, Thomas Wolf, Natural Language Processing with Transformers, O'Reilly Media  
2. David Foster, Generative Deep Learning, O'Reilly Media, Second Edition  
3. Daniel Jurafsky and James H. Martin, Speech and Language Processing, freely available draft edition  
4. Steven Bird, Ewan Klein, Edward Loper, Natural Language Processing with Python, O'Reilly Media

**Weblinks:**

1. LangChain Documentation: [https://python.langchain.com/](https://python.langchain.com/)  
2. LangGraph Documentation: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)  
3. Hugging Face NLP Course: [https://huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course)  
4. spaCy Documentation: [https://spacy.io/](https://spacy.io/)

