# AI-ML for Mechanical Engineers – 2026

A 20-hour course (20 sessions x 1 hour) introducing mechanical engineering students to the
foundational concepts, tools, and techniques of machine learning. By bridging the gap between
traditional engineering principles and modern AI methodologies, the course equips learners with
the skills needed to tackle real-world problems in mechanical engineering (predictive
maintenance, quality control, process optimization, design automation).

## Course Logistics

- **Instructor**: Yogesh H Kulkarni (Professor of Practice, CoEP)
- **Email**: firstnamelastname \[at\] yahoo \[dot\] com (Subject "AIML4ME2026...")
- **Phone**: nine eight nine zero two five one four zero six (for SMS, No WhatsApp)
- **Class**: Final Year B.Tech Mech, 8th Semester, Elective
- **Duration**: 20 hours (20 sessions x 1 hour each)
- **Venue**: UG-Class Room 301 (Bajaj Building 3rd Floor)???
- **Timing**: Thursdays & Fridays 12:30 am to 13:30 pm (1-hour sessions, twice per week)
- **Academic Year**: Jul – Nov 2026

## Course Objectives

- Understand the basic concepts of Machine Learning and AI
- Study various machine learning techniques and implement them in Python
- Apply learnt knowledge to real-world mechanical engineering problems
- Design and outline deployment of production ML systems

## Learning Outcomes

Upon completion, students will:

1. Load, clean, explore, and engineer features from raw engineering datasets
2. Build and evaluate supervised learning models (regression, classification, ensemble methods)
3. Apply unsupervised learning techniques for clustering and dimensionality reduction
4. Systematically select features and tune hyperparameters, validating models using cross-validation
5. Compare algorithms and select the best for a given engineering problem context
6. Interpret and communicate results to technical and non-technical stakeholders
7. Design and outline the architecture for deploying ML systems in production environments
8. Apply ML to real mechanical engineering scenarios (bearing diagnostics, quality prediction, etc.)

## Prerequisites

### Skills Required

- **Programming**: Any high-level language (C/C++, Java, or basic Python)
- **Mathematics**: Linear algebra, calculus, probability, and statistics fundamentals
- **Reference**: [How to Become a Data Scientist? – Yogesh Kulkarni](https://medium.com/technology-hits/how-to-become-a-data-scientist-f673a30cafcd)

### Technical Setup

- Personal laptop with Python 3.8+ installed
- Recommended: Anaconda/Miniconda for environment management
- Tools: Jupyter Notebook, Git (for code backup)
- Cloud backup: Google Drive or GitHub for assignment submissions

## 20-Session Lesson Plan

Each session is delivered as its own self-contained Beamer deck and CheatSheet, and also
compiled together into one all-in-one course driver (`Main_Course_MLCoEP_{Presentation,
CheatSheet}.tex` under `LaTeX/`). Sessions are independent, not cumulative prerequisites of
each other.

| # | Topic | Focus |
|---|-------|-------|
| **1** | AI Overview | AI history, ML basics, career paths for mechanical engineers |
| **2** | Python | Programming fundamentals for ML |
| **3** | EDA | Why EDA matters, data prep theory, end-to-end walkthrough (telecom churn) |
| **4** | Pandas | Hands-on data manipulation (machine sensor logs dataset) |
| **5** | ML Intro | Classification/regression/clustering/ranking problem types, model selection |
| **6** | ML Concepts | Bias/variance, cross-validation, precision/recall/F1, hyperparameter tuning intuition |
| **7** | Sklearn Workflow | Estimator API, data prep + evaluation metrics with scikit-learn (Pima diabetes, Boston housing) |
| **8** | Feature Selection | Filter vs. wrapper selection, hyperparameter tuning, end-to-end hands-on pipeline (Steel Plates Faults) |
| **9** | Linear Regression | Least squares, gradient descent, the advertising-budget worked example |
| **10** | Logistic Regression | Classification, decision boundary, ROC/AUC |
| **11** | Decision Trees | Splitting criteria, entropy/information gain |
| **12** | Ensemble Methods | Bagging, boosting, Random Forest |
| **13** | Support Vector Machines | Hyperplanes, margins, kernel trick |
| **14** | Naive Bayes | Bayes' theorem, conditional independence, Laplace smoothing |
| **15** | K-Nearest Neighbors | Distance metrics, choosing k, lazy learning |
| **16** | K-Means Clustering | Unsupervised clustering, choosing k |
| **17** | PCA | Dimensionality reduction |
| **18** | Titanic Capstone | End-to-end classification pipeline (Kaggle Titanic) |
| **19** | MLOps & Deployment | Production ML systems, predictive analytics case study |
| **20** | ME Applications | Applied mechanical-engineering ML scenarios, worked project exemplars |

## Evaluation & Assessment

### Assessment Methods

- **Midterm Exam**: Sessions 1-12, 10 descriptive (3 marks each) + 10 numerical (4 marks each)
  + 10 code/syntax (2 marks each) = 90 marks
- **Final/EndSem Exam**: Sessions 1-20, 20 descriptive (3 marks each) + 20 numerical
  (4 marks each) + 20 code/syntax (2 marks each) = 180 marks

Question banks are maintained outside this repository (this is a public, open-source repo,
so exam content is never committed here).

### Grading Criteria

- **Code Correctness**: Does it run without errors? Correct results?
- **Clarity**: Readable code, documented steps, clear explanations?
- **Efficiency**: Best practices, algorithmic efficiency?
- **Insights**: Reflections show understanding? Justified decisions?

## Academic Integrity Policy

### Guidelines

- This course is for your learning. Build your own thinking.
- Always cite sources (papers, blogs, Stack Overflow, documentation).
- **Avoid generating code by GenAI** (ChatGPT, Copilot, etc.); use your own code and words.
- You may reference tutorials and documentation, but copy-paste code verbatim is not permitted.
- Collaboration is encouraged, but each student must submit their own work.
- Plagiarism will result in 0 marks for the assignment.

## Course Materials & Resources

### Slides

All session slides (Presentation + CheatSheet PDFs) are built from `LaTeX/` in this repo
(session content files `seminar_mlcoep_session_<N>_content.tex`, chained into
`course_mlcoep_content.tex`). Compiled PDFs are not committed to this public repo by policy;
compile locally with MikTeX's `texify`, or ask the instructor for the current PDFs.

### Hands-On Code & Datasets

- `Code/mlcoep/sessions/session08_feature_selection/`: runnable feature-selection pipeline
  scripts (Session 8)
- `Code/mlcoep/datasets/`: offline copies of the datasets used in Sessions 4, 7, and 8, so
  those sessions run without internet access

### Textbooks

- None. Topic-wise course material available on [GitHub TeachingDataScience](https://github.com/yogeshkulkarni/TeachingDataScience).
- **Recommended supplementary reading**:
  - [Scikit-learn documentation](https://scikit-learn.org/)
  - *Hands-On Machine Learning with Scikit-Learn and TensorFlow* (Geron)
  - *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani) – Free PDF

### Tools & Technologies

- **Languages**: Python 3.8+
- **Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Jupyter Notebooks
- **Datasets**: Kaggle, UCI ML Repository, manufacturing datasets

### Communication & Support

- Email for questions
- Office hours (by appointment)
- Discussion forum (if available)

## Real-World Applications (Mechanical Engineering Focus)

Students will apply ML to solve practical mechanical engineering problems:

1. **Predictive Maintenance**: Diagnose bearing/equipment faults before failure
2. **Quality Control**: Classify manufacturing defects and anomalies
3. **Process Optimization**: Predict and optimize production parameters (temperature, pressure, cycle time)
4. **Equipment Segmentation**: Group equipment by failure modes or performance characteristics
5. **Design Automation**: Predict component lifespan or performance under conditions

## Warm-up Exercises

Complete before the course starts:

1. **Programming**: Write compilable Python code for Fibonacci Series (10 lines)
2. **Mathematics**: Compute the dot product of two 3D vectors by hand
3. **Critical Thinking**: What is Machine Learning? Write your thoughts (5 lines)
4. **Data Exploration**: Download a CSV from Kaggle and explore it in Pandas

## Who Should Attend

- **Final-year B.Tech mechanical engineering students**
- Engineers seeking ML skills for process automation and quality control
- R&D personnel interested in data-driven product development
- Quality and maintenance teams wanting predictive analytics
- Technical professionals transitioning to data-driven roles

## Key Benefits

- Practical skills for ML-based engineering solutions
- Industry-standard tools (Python, scikit-learn)
- Reduce equipment downtime through predictive analytics
- Optimize manufacturing processes and quality
- Make data-driven decisions confidently
- Build portfolio projects for career advancement

## Instructor Background

**Yogesh H Kulkarni** is a Visiting Faculty at CoEP with expertise in machine learning, data science, and software engineering. He has designed and delivered AI/ML courses for engineering students and professionals, emphasizing practical application and career-ready skills.

## Recommended Next Courses

- **Deep Learning**: Neural networks, CNN, RNN
- **Advanced ML**: Reinforcement learning, generative models
- **MLOps & Deployment**: Production systems, cloud platforms
- **Specialized Applications**: Time series forecasting, anomaly detection

## FAQ

**Q: Do I need prior ML experience?**
A: No. Beginner-friendly; we start with Python and math fundamentals.

**Q: Is a laptop required?**
A: Yes. Personal laptop with Python installed.

**Q: What if I miss a session?**
A: Notes will be shared. Attendance strongly advised.

**Q: How much time outside class?**
A: 2-3 hours per week for practice.

**Q: Can I use AI tools (ChatGPT, Copilot)?**
A: No. Write your own code. GenAI usage may violate academic integrity.

---

**Transform your mechanical engineering career with Machine Learning.**
*Learn from industry experience. Build real-world projects. Join the data-driven revolution.*

*Designed and delivered by Yogesh H Kulkarni | CoEP Elective 2026*
