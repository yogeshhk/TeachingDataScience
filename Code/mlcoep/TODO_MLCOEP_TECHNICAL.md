# TODO: MLCOEP — Technical Deliverables

**Scope**: Code, data, notebooks, assignments, exams, resources  
**Status**: Course in progress (Jul 16, 2026 – Nov 30, 2026)  
**Last Updated**: Jul 16, 2026

---

## **🔴 URGENT: Environment & Data Setup (Due By Jul 18)**

- [x] **PYTHON_SETUP** — Create Python Setup Guide & Environment File ✅ DONE
  - [x] `SETUP_GUIDE.md` — Installation, conda setup, Jupyter launch, troubleshooting (8KB)
  - [x] `environment.yml` — Conda dependencies (python=3.9, numpy, pandas, scikit-learn, matplotlib, seaborn, jupyter) (1KB)
  - **Completed**: Jul 16, 2026 | **Effort**: 1h | **Status**: Ready for students on Jul 20
  - **Location**: `mlcoep/SETUP_GUIDE.md`, `mlcoep/environment.yml`

- [x] **DATASETS_DOWNLOAD** — Download & Document Engineering Datasets ✅ DOCS DONE
  - [x] Manufacturing equipment logs — README.md created (2KB) | Dataset download: Aug 1
  - [x] Quality metrics dataset — README.md created (2KB) | Dataset download: Aug 5
  - [x] Predictive maintenance (CMAPSS/bearing) — README.md created (3KB) | Dataset download: Aug 15
  - [x] Capstone dataset (Titanic) — README.md created (5KB) | Ready to download (public)
  - [x] For each: created `README.md` with source URL placeholder, data dictionary template, license, download steps
  - **Docs Status**: COMPLETE (Jul 16) | **Dataset Downloads**: Staggered Aug 1-15 | **Effort**: 0.5h (docs) + 1-2h (downloads later)
  - **Location**: `mlcoep/datasets/raw/`
  - **Subfolder structure**: ✅ Created with README.md in each:
    ```
    datasets/raw/
    ├── manufacturing_equipment/
    │   └── README.md ✅
    ├── quality_metrics/
    │   └── README.md ✅
    ├── predictive_maintenance/
    │   └── README.md ✅
    └── capstone/
        └── README.md ✅ (Titanic: ready to download)
    ```

---

## **🟡 MEDIUM PRIORITY: Content & Assessment Materials**

### **Practice Notebooks (Student + Solutions)**

- [ ] **T2.8-1** — Notebook 1: Python Fundamentals Practice
  - Topics: Variables, data types, collections, basic operations
  - **Create by**: End of Jul (before Session 3) | **Effort**: 1.5h
  - **Location**: `mlcoep/notebooks/practice/01_Python_Fundamentals.ipynb`

- [ ] **T2.8-2** — Notebook 2: NumPy & Pandas Exercises
  - Topics: Arrays, series, dataframes, indexing, groupby, merging
  - **Create by**: Early Aug (before Session 7) | **Effort**: 1.5h
  - **Location**: `mlcoep/notebooks/practice/02_NumPy_Pandas.ipynb`

- [ ] **T2.8-3** — Notebook 3: EDA & Visualization
  - Topics: Statistical summaries, missing values, plotting (histograms, scatter, box plots)
  - **Create by**: Mid-Aug (before Session 8) | **Effort**: 1.5h
  - **Location**: `mlcoep/notebooks/practice/03_EDA_Visualization.ipynb`

- [ ] **T2.8-4** — Notebook 4: Regression Implementation
  - Topics: Linear regression, logistic regression, train/test split, evaluation metrics
  - **Create by**: Late Aug (before Session 12) | **Effort**: 1.5h
  - **Location**: `mlcoep/notebooks/practice/04_Regression.ipynb`

- [ ] **T2.8-5** — Notebook 5: Classification & Unsupervised
  - Topics: Decision trees, random forests, SVM, K-means, dimensionality reduction
  - **Create by**: Mid-Sep (before Session 19) | **Effort**: 1.5h
  - **Location**: `mlcoep/notebooks/practice/05_Classification_Unsupervised.ipynb`

- [ ] **T2.8-6** — Notebook 6: Model Evaluation & Tuning
  - Topics: Cross-validation, hyperparameter tuning, model comparison, regularization
  - **Create by**: Late Sep (before Session 21) | **Effort**: 1.5h
  - **Location**: `mlcoep/notebooks/practice/06_Evaluation_Tuning.ipynb`

- [ ] **T2.9-1 through T2.9-6** — Solution Notebooks (Teacher Versions)
  - Corresponding solutions for notebooks 1-6, with detailed explanations & best practices
  - **Create after** each practice notebook | **Effort**: 0.5h each (3h total)
  - **Location**: `mlcoep/notebooks/solutions/01_Python_Fundamentals_Solution.ipynb` (etc.)

---

### **Assignments (Problem Statements, Solutions, Rubrics)**

- [ ] **T3.1** — Assignment A1: Python + Data Types
  - **Files**:
    - `A1_Problem_Statement.md` — 3 coding problems (Fibonacci, equation solver, data analysis)
    - `A1_Solution.py` — working solution code
    - `A1_Rubric.md` — grading criteria (40% correctness, 20% clarity, 20% efficiency, 20% docs)
    - `A1_TestCases.py` — optional test cases for auto-checking
  - **Due**: Jul 27, 2026 | **Create by**: Jul 20 | **Effort**: 2h
  - **Location**: `mlcoep/assignments/A1_Python_DataTypes/`

- [ ] **T3.2** — Assignment A2: Math + Stats
  - **Files**:
    - `A2_Problem_Statement.md` — theory exercises + EDA on provided dataset
    - `A2_Solution.py` / `A2_Solution.ipynb` — working solution
    - `A2_Rubric.md` — grading criteria
    - `A2_Dataset.csv` — provided dataset for practical portion
  - **Due**: Aug 5, 2026 | **Create by**: Jul 30 | **Effort**: 2h
  - **Location**: `mlcoep/assignments/A2_Math_Stats/`

- [ ] **T3.3** — Assignment A3: ML Fundamentals
  - **Files**:
    - `A3_Problem_Statement.md` — implement baseline classifier, compute metrics, reflect
    - `A3_Solution.ipynb` — working solution
    - `A3_Rubric.md` — grading criteria
    - `A3_Dataset.csv` — classification dataset
  - **Due**: Aug 15, 2026 | **Create by**: Aug 8 | **Effort**: 2h
  - **Location**: `mlcoep/assignments/A3_ML_Fundamentals/`

- [ ] **T3.4** — Assignment A4: Regression + Trees
  - **Files**:
    - `A4_Problem_Statement.md` — regression vs tree comparison, report writing
    - `A4_Solution.ipynb` — working solution
    - `A4_Rubric.md` — grading criteria
    - `A4_QualityDataset.csv` + `A4_MaintenanceDataset.csv` — two datasets for comparison
  - **Due**: Aug 25, 2026 | **Create by**: Aug 18 | **Effort**: 2h
  - **Location**: `mlcoep/assignments/A4_Regression_Trees/`

- [ ] **T3.5** — Assignment A5: Classification + Unsupervised
  - **Files**:
    - `A5_Problem_Statement.md` — SVM + K-means, reflection on supervised vs unsupervised
    - `A5_Solution.ipynb` — working solution
    - `A5_Rubric.md` — grading criteria
    - `A5_ClassificationDataset.csv` + `A5_BehaviorDataset.csv` — two datasets
  - **Due**: Sep 5, 2026 | **Create by**: Aug 28 | **Effort**: 2h
  - **Location**: `mlcoep/assignments/A5_Classification_Unsupervised/`

- [ ] **T3.6** — Assignment A6: Model Evaluation & Tuning
  - **Files**:
    - `A6_Problem_Statement.md` — hyperparameter tuning report, before/after comparison
    - `A6_Solution.ipynb` — working solution with tuning examples
    - `A6_Rubric.md` — grading criteria
    - `A6_Dataset.csv` — dataset for tuning experiments
  - **Due**: Oct 5, 2026 | **Create by**: Sep 28 | **Effort**: 2h
  - **Location**: `mlcoep/assignments/A6_Evaluation_Tuning/`

---

### **Exams (Problem Statements & Solutions)**

- [ ] **T3.7** — Midterm Exam
  - **Files**:
    - `Midterm_Questions.md` — 15 MCQs (1 mark) + 3 short answer (3 marks) + 1 coding (5 marks)
    - `Midterm_Solution.md` — detailed solutions with marking scheme
    - `Midterm_CodingTemplate.py` — starter code for coding question
  - **Scheduled**: Late Aug / early Sep 2026 | **Create by**: Early Aug | **Effort**: 3h
  - **Location**: `mlcoep/exams/midterm/`

- [ ] **T3.8** — EndSem Exam
  - **Files**:
    - `EndSem_Questions.md` — 10 MCQs (1) + 5 short answer (3) + 2 long answer (5) + 1 design (5)
    - `EndSem_Solution.md` — detailed solutions with marking scheme
    - `EndSem_CodingTemplate.py` — starter code for any coding questions
  - **Scheduled**: Late Nov 2026 | **Create by**: Oct | **Effort**: 4h
  - **Location**: `mlcoep/exams/endsem/`

- [ ] **T3.9** — Exam Solutions (For distribution after exams)
  - `Midterm_DistributionVersion.md` — clean solution document (without detailed marking)
  - `EndSem_DistributionVersion.md` — clean solution document
  - **Create after exam** is given | **Effort**: 1h each

---

### **Question Banks (Optional, for Quizzes/Formative Assessments)**

- [ ] **T3.10** — MCQ Question Bank (150+ questions)
  - CSV or JSON file: `QuestionBank_MCQ.csv` (columns: topic, question, options A-D, correct_answer, difficulty, explanation)
  - Organized by: topic (Sessions 1-24), difficulty (Basic/Intermediate/Advanced)
  - **Create by**: Sep 2026 | **Effort**: 6h | **Optional**
  - **Location**: `mlcoep/resources/QuestionBank_MCQ.csv`

- [ ] **T3.11** — Short Answer Question Bank (40+ questions)
  - Markdown file: `QuestionBank_ShortAnswer.md` (question + model answer + rubric)
  - Organized by topic, 3-5 marks each
  - **Create by**: Sep 2026 | **Effort**: 4h | **Optional**
  - **Location**: `mlcoep/resources/QuestionBank_ShortAnswer.md`

- [ ] **T3.12** — Design Question Bank (10+ questions)
  - Markdown file: `QuestionBank_Design.md` ("Design an ML system for..." scenarios)
  - Model solutions with trade-off analysis
  - **Create by**: Sep 2026 | **Effort**: 3h | **Optional**
  - **Location**: `mlcoep/resources/QuestionBank_Design.md`

---

## **📁 Folder Structure (for reference)**

```
mlcoep/
├── SETUP_GUIDE.md
├── environment.yml
├── datasets/
│   ├── raw/
│   │   ├── manufacturing_equipment/
│   │   │   └── README.md (+ data files)
│   │   ├── quality_metrics/
│   │   │   └── README.md (+ data files)
│   │   ├── predictive_maintenance/
│   │   │   └── README.md (+ data files)
│   │   └── capstone/
│   │       └── README.md (+ data files)
│   └── processed/
├── assignments/
│   ├── A1_Python_DataTypes/
│   │   ├── A1_Problem_Statement.md
│   │   ├── A1_Solution.py
│   │   ├── A1_Rubric.md
│   │   └── A1_TestCases.py (optional)
│   ├── A2_Math_Stats/
│   ├── A3_ML_Fundamentals/
│   ├── A4_Regression_Trees/
│   ├── A5_Classification_Unsupervised/
│   └── A6_Evaluation_Tuning/
├── notebooks/
│   ├── practice/
│   │   ├── 01_Python_Fundamentals.ipynb
│   │   ├── 02_NumPy_Pandas.ipynb
│   │   ├── 03_EDA_Visualization.ipynb
│   │   ├── 04_Regression.ipynb
│   │   ├── 05_Classification_Unsupervised.ipynb
│   │   └── 06_Evaluation_Tuning.ipynb
│   └── solutions/
│       ├── 01_Python_Fundamentals_Solution.ipynb
│       ├── 02_NumPy_Pandas_Solution.ipynb
│       └── ... (etc.)
├── exams/
│   ├── midterm/
│   │   ├── Midterm_Questions.md
│   │   ├── Midterm_Solution.md
│   │   ├── Midterm_DistributionVersion.md (after exam)
│   │   └── Midterm_CodingTemplate.py
│   └── endsem/
│       ├── EndSem_Questions.md
│       ├── EndSem_Solution.md
│       ├── EndSem_DistributionVersion.md (after exam)
│       └── EndSem_CodingTemplate.py
├── projects/
│   ├── T1_Milestone/
│   └── T2_Milestone/
└── resources/
    ├── QuestionBank_MCQ.csv (optional)
    ├── QuestionBank_ShortAnswer.md (optional)
    └── QuestionBank_Design.md (optional)
```

---

## **SUMMARY: Technical Deliverables**

| Item | Status | Create By | Location |
|------|--------|-----------|----------|
| Python Setup Guide | ⏳ | Jul 17 | `mlcoep/SETUP_GUIDE.md` |
| Environment.yml | ⏳ | Jul 17 | `mlcoep/environment.yml` |
| Datasets (raw) | ⏳ | Jul 18 – Aug 15 | `mlcoep/datasets/raw/` |
| Practice Notebooks 1-6 | ⏳ | Jul – Sep | `mlcoep/notebooks/practice/` |
| Solution Notebooks 1-6 | ⏳ | Jul – Sep | `mlcoep/notebooks/solutions/` |
| Assignments A1-A6 | ⏳ | Jul 20 – Sep 28 | `mlcoep/assignments/` |
| Midterm Exam | ⏳ | Early Aug | `mlcoep/exams/midterm/` |
| EndSem Exam | ⏳ | Oct | `mlcoep/exams/endsem/` |
| Question Banks (optional) | ⏳ | Sep | `mlcoep/resources/` |

---

**Last Updated**: Jul 16, 2026  
**Next Review**: Jul 20, 2026 (course launch)
