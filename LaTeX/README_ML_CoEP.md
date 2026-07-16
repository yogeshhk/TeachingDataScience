# Machine Learning for Mechanical Engineers – CoEP

A 24-hour intensive course introducing ML fundamentals, practical algorithms, and real-world engineering applications. Designed for final-year B.Tech students.

## Course Logistics

- **Instructor**: Yogesh H Kulkarni (Visiting Faculty, CoEP)
- **Email**: firstnamelastname \[at\] yahoo \[dot\] com (Subject "ML@CoEP...")
- **Phone**: nine eight nine zero two five one four zero six (for SMS/WhatsApp)
- **Duration**: 24 hours (24 sessions × 1 hour each)
- **Format**: Theory + Demo/Assignment paired structure
- **Target Audience**: Final-year B.Tech Mechanical Engineering students (Elective)
- **Venue**: CoEP Classroom (TBD)
- **Timing**: Two 1-hour sessions per day, twice per week (preferred)

## Course Objectives

- Understand the basic concepts of Machine Learning and AI
- Study various machine learning techniques and implement them in Python
- Apply learnt knowledge to real-world engineering problems (predictive maintenance, quality control, process optimization)
- Design and deploy production ML systems

## Learning Outcomes

Upon completion, students will:

1. Load, clean, explore, and engineer features from raw engineering datasets
2. Build and evaluate supervised learning models (regression, classification, ensemble methods)
3. Apply unsupervised learning techniques for clustering and dimensionality reduction
4. Systematically tune hyperparameters and validate models using cross-validation
5. Compare algorithms and select the best for a given problem context
6. Interpret and communicate results to technical and non-technical stakeholders
7. Design and outline the architecture for deploying ML systems in production

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

## Lesson Plan (24 Sessions)

### **FOUNDATIONS (Sessions 1–11, 11 hours)**

| Session | Topic | Key Concepts | Focus |
|---------|-------|--------------|-------|
| **1** | AI Overview & Career | AI history, ML basics, career paths | Why ML matters for engineers; opportunities |
| **2–3** | Python Fundamentals | Syntax, data structures, control flow, functions | Programming essentials for ML |
| **4–5** | Mathematical Foundations | Vectors, matrices, calculus, data types (NOIR) | Linear algebra and data structures |
| **6–7** | Statistics & Data Manipulation | Descriptive stats, correlation, Pandas | Statistical thinking and data wrangling |
| **8–9** | Data Preparation | EDA, visualization, feature engineering, scaling | Churn dataset demo → data pipeline |
| **10–11** | ML Fundamentals | Supervised vs unsupervised, metrics, CV, scikit-learn | Core ML concepts and tools |

### **ALGORITHMS (Sessions 12–20, 9 hours)**

| Sessions | Topic | Key Concepts | Demo/Assignment |
|----------|-------|--------------|-----------------|
| **12–13** | Regression | Linear regression, logistic regression, regularization | Housing price prediction + Alice classification |
| **14–15** | Tree-based Methods | Decision trees, ensembles, random forests, feature importance | UCI Adults + heart disease prediction |
| **16** | Support Vector Machines | Hyperplanes, margins, kernel trick (linear, RBF, polynomial) | MNIST digit classification |
| **17** | Naive Bayes | Bayes' theorem, conditional independence, probability | Text classification (sentiment/topic) |
| **18** | K-Nearest Neighbors | Distance metrics, choosing k, lazy learning | Wine quality classification |
| **19–20** | Unsupervised Learning | K-means clustering, PCA, dimensionality reduction | Customer segmentation + PCA visualization |

### **PRODUCTION & INTEGRATION (Sessions 21–24, 4 hours)**

| Sessions | Topic | Key Concepts | Focus |
|----------|-------|--------------|-------|
| **21–22** | Model Evaluation & Hyperparameter Tuning | Confusion matrix, cross-validation, grid search, ROC-AUC | Pima Indians case study; model comparison |
| **23–24** | Capstone & Production | End-to-end ML pipeline, model deployment, MLOps | Titanic case study + deployment design |

## Evaluation & Assessment

### Assessment Methods

- **Session Assignments** (50%): 12 assignment pairs (one per 2 sessions)
  - Coding exercises, Jupyter notebooks, written reflections
  - Topics: Python problems, EDA, model building, hyperparameter tuning

- **Project Demonstrations** (20%): Embedded in sessions
  - Churn, housing, heart disease, customer segmentation, Titanic
  - Outputs and insights documented

- **Capstone Project** (30%): Sessions 23–24
  - Design Document (2–3 pages): problem, architecture, deployment
  - Python code skeleton: training, evaluation, serialization
  - Presentation (5 min): problem, results, deployment plan

### Grading Criteria

- **Code Correctness**: Does it run without errors? Correct results?
- **Completeness**: All required tasks finished?
- **Clarity**: Readable code, documented steps, clear outputs?
- **Insights**: Reflections show understanding? Justified decisions?
- **Capstone**: Professional design? Addresses scalability, monitoring?

## Academic Integrity Policy

### Guidelines

- This course is for your learning. Build your own thinking.
- Always cite sources (papers, blogs, Stack Overflow, documentation).
- **Avoid generating code by GenAI** (ChatGPT, Copilot, etc.); use your own code and words.
- You may reference tutorials and documentation, but copy-paste code verbatim is not permitted.
- Collaboration is encouraged, but each student must submit their own work.
- Plagiarism will result in 0 marks for the assignment.

## Course Materials & Resources

### Textbooks

- None. Topic-wise course material available on the [GitHub repository](https://github.com/yogeshkulkarni/TeachingDataScience).
- **Recommended supplementary reading**:
  - [Scikit-learn documentation](https://scikit-learn.org/)
  - *Hands-On Machine Learning with Scikit-Learn and TensorFlow* (Géron)
  - *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani) – Free PDF

### Tools & Technologies

- **Languages**: Python 3.8+
- **Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Jupyter Notebooks
- **Datasets**: Kaggle, UCI ML Repository, built-in scikit-learn datasets

### Communication & Support

- Email for questions (see Course Logistics)
- Discussion forum (coming soon)
- Office hours (by appointment)

## Warm-up Exercises

Complete these before the course starts:

1. **Programming**: Write compilable Python code for Fibonacci Series (10 lines)
2. **Mathematics**: Compute the dot product of two 3D vectors by hand
3. **Critical Thinking**: What is Machine Learning? Write your thoughts (5 lines)
4. **Data Exploration**: Download a CSV from Kaggle and explore it in Pandas

## Real-World Applications & Case Studies

### Engineering Use Cases

1. **Predictive Maintenance**: Diagnose equipment faults before failure
2. **Quality Control**: Classify manufacturing defects or anomalies
3. **Process Optimization**: Predict and optimize production parameters
4. **Equipment Segmentation**: Group by characteristics or behavior
5. **Performance Prediction**: Forecast equipment lifespan or energy consumption

### Datasets & Projects

- **Churn**: EDA and feature engineering
- **Housing**: Linear regression
- **Alice**: Logistic regression
- **UCI Adults**: Tree-based classification
- **Heart Disease**: Random forest
- **MNIST Digits**: SVM for image classification
- **Wine Quality**: K-NN for taste prediction
- **Pima Indians**: Model evaluation and hyperparameter tuning
- **Titanic**: End-to-end ML pipeline

## Hands-On Learning Approach

Each topic pair follows a consistent structure:

1. **Theory Session (1 hour)**
   - Concepts, mathematics, motivation
   - Why this algorithm matters
   - When to use it, trade-offs

2. **Demo/Assignment Session (1 hour)**
   - Live walkthrough on real data
   - Hands-on coding exercise
   - Reflection or comparison task

## Who Should Attend

- **Final-year B.Tech mechanical engineering students**
- **Manufacturing engineers** seeking ML skills
- **R&D personnel** interested in data-driven development
- **Quality and maintenance teams** wanting predictive analytics
- **Technical professionals** transitioning to data-driven roles

## Key Benefits

- ✅ Practical skills for ML-based engineering solutions
- ✅ Industry-standard tools (Python, scikit-learn)
- ✅ Reduce downtime through predictive analytics
- ✅ Optimize manufacturing and quality
- ✅ Build data-driven decision-making capability
- ✅ Portfolio-building projects for career advancement
- ✅ Network with AI/ML enthusiasts

## Instructor Background

**Yogesh H Kulkarni** is a Visiting Faculty at CoEP with expertise in machine learning, data science, and software engineering. He has designed and delivered AI/ML courses for engineering students, professionals, and industry teams.

## Recommended Next Courses

- **Deep Learning**: Neural networks, CNN, RNN
- **Natural Language Processing**: Text analysis, LLMs
- **Advanced ML**: Reinforcement learning, generative models
- **MLOps & Deployment**: Production systems, cloud platforms
- **Time Series & Forecasting**: Predictive maintenance, anomaly detection

## Frequently Asked Questions

**Q: Do I need prior ML experience?**  
A: No. Beginner-friendly; we start with Python and math fundamentals.

**Q: Is a laptop required?**  
A: Yes. Personal laptop with Python installed.

**Q: Can I attend if I know Python?**  
A: Yes. You'll progress quickly through Sessions 2–3.

**Q: What if I miss a session?**  
A: Recordings and notes shared (subject to availability). Attendance strongly advised.

**Q: How much time outside class?**  
A: 2–3 hours per week for assignments. The more you code, the better you learn.

**Q: Can I use AI tools (ChatGPT, Copilot)?**  
A: No. Generate your own code. GenAI usage violates academic integrity.

## Contact & Registration

- **Email**: firstnamelastname@yahoo.com
- **Phone**: nine eight nine zero two five one four zero six (SMS/WhatsApp)
- **Venue**: CoEP Campus, Classroom TBD
- **Dates**: [TBD – typically January–April for B.Tech electives]

**Secure your spot today. Limited seats.**

## References & Resources

1. **IITB ML Course**: https://www.cse.iitb.ac.in/~pjyothi/cs419/
2. **Scikit-learn**: https://scikit-learn.org/
3. **"An Introduction to Statistical Learning"** – Free PDF
4. **Kaggle Learn**: https://www.kaggle.com/learn
5. **GitHub**: [TeachingDataScience ML at CoEP](https://github.com/yogeshkulkarni/TeachingDataScience)

---

**Transform your engineering career with Machine Learning.**  
*Learn from industry experience. Build real-world projects. Join the data-driven revolution.*

*Designed and delivered by Yogesh H Kulkarni | CoEP Continuing Education*
