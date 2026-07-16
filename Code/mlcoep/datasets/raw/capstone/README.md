# Capstone Dataset: Titanic Survival Prediction

**Purpose**: Capstone case study & complete ML pipeline  
**Sessions**: 23-24 (Capstone & Production)  
**Project**: T2 Project (Final capstone project)

---

## **Dataset Information**

| Property | Value |
|----------|-------|
| **Name** | Titanic: Machine Learning from Disaster |
| **Type** | Structured tabular (binary classification) |
| **Format** | CSV |
| **Size** | ~61 KB |
| **Rows** | 891 (train) + 418 (test) |
| **Columns** | 11 features + 1 target |
| **License** | Public / CC0 |

---

## **Download Instructions**

**Source URL**:
- [ ] **Kaggle**: https://www.kaggle.com/c/titanic/data
  - Requires Kaggle account (free)
  - Download: `train.csv` and `test.csv`

- [ ] **Direct**: https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv

**Steps**:
1. Visit Kaggle or Stanford link (see above)
2. Download `train.csv` and optionally `test.csv`
3. Place in this folder: `mlcoep/datasets/raw/capstone/`
4. Verify file names:
   ```
   capstone/
   ├── train.csv    (891 rows)
   └── test.csv     (418 rows, optional)
   ```

---

## **Data Dictionary**

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| **PassengerId** | Int | 1-891 | Unique passenger ID |
| **Survived** | Binary | {0, 1} | **Target**: Survived (1) or not (0) |
| **Pclass** | Int | {1, 2, 3} | Passenger class (1=1st, 2=2nd, 3=3rd) |
| **Name** | String | — | Passenger name |
| **Sex** | Categorical | {male, female} | Passenger gender |
| **Age** | Float | 0.42-80.0 | Age in years (some missing) |
| **SibSp** | Int | 0-8 | # of siblings/spouses aboard |
| **Parch** | Int | 0-6 | # of parents/children aboard |
| **Ticket** | String | — | Ticket number |
| **Fare** | Float | 0-512.33 | Ticket price in pounds (some missing) |
| **Cabin** | String | — | Cabin number (mostly missing) |
| **Embarked** | Categorical | {C, Q, S} | Port of embarkation (some missing) |

---

## **Basic Statistics**

```
Training Set: 891 rows, 11 features
Test Set: 418 rows, 11 features (optional)

Target Distribution (train):
  - Survived (1): 342 (38.4%)
  - Not Survived (0): 549 (61.6%)
  
Missing Values (train):
  - Age: 177 missing (19.9%)
  - Cabin: 687 missing (77.2%)
  - Embarked: 2 missing (0.2%)
  
Data Types:
  - Numeric: PassengerId, Pclass, Age, SibSp, Parch, Fare
  - Categorical: Sex, Embarked
  - Text: Name, Ticket, Cabin
```

---

## **Usage in Course**

### **Session 23 (Titanic Case Study)**
Complete ML pipeline demonstration:
1. **Load & Explore**: EDA, missing values, visualizations
2. **Feature Engineering**: 
   - Extract title from Name
   - Create family size feature
   - Encode categorical variables
3. **Model Building**: Train logistic regression + tree models
4. **Evaluation**: Accuracy, precision, recall, F1, ROC-AUC
5. **Predictions**: Make predictions on test set

### **Session 24 (MLOps & Deployment)**
Production considerations:
- Model serialization (pickle/joblib)
- REST API for predictions
- Monitoring predictions in production
- Handling new data

### **T2 Project (Final Capstone)**
Students choose from capstone options:
- **Option 1**: Titanic (provided dataset)
- **Option 2**: Real manufacturing dataset (if available)
- **Option 3**: Custom dataset (with approval)

**Expected deliverable**:
- Jupyter notebook with full pipeline
- Trained model file
- Predictions on test set
- Report on methodology & results
- Presentation (10-15 min)

---

## **License & Citation**

**License**: CC0 (Public Domain)
- Public dataset: Yes
- Commercial use allowed: Yes
- Attribution required: No (but appreciated)

**Citation**:
```
Titanic: Machine Learning from Disaster. Kaggle.
URL: https://www.kaggle.com/c/titanic
```

---

## **Interesting Facts (for context)**

- **Real event**: RMS Titanic sank on April 15, 1912 (1,502 deaths out of 2,224 passengers/crew)
- **Class effect**: First-class passengers had higher survival rates (~62%) vs third-class (~24%)
- **Gender effect**: ~74% of women survived vs ~19% of men (women & children first policy)
- **Why this dataset?**: Classic ML problem with real historical significance + engaging narrative

---

## **Why Titanic for Capstone?**

1. **Complete**: Has all phases of ML pipeline (EDA, feature engineering, modeling, evaluation)
2. **Practical**: Real classification problem with business context
3. **Beginner-friendly**: Clean, well-structured, commonly used for ML education
4. **Engagement**: Interesting historical context keeps students motivated
5. **Balanced**: Not too easy (requires some feature engineering) nor too hard (solvable in 4 weeks)

---

## **Notes**

- **Dataset is already available** online — no custom collection needed
- **Small size**: ~61 KB — downloads instantly, no storage concerns
- **Perfect for finale**: Wraps up all techniques learned in Sessions 1-22
- **Fallback option**: If manufacturing data unavailable, Titanic is always ready

---

**Status**: Ready to use — Official capstone dataset for course  
**Download deadline**: By Jul 20, 2026 (course start)
