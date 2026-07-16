# Predictive Maintenance Dataset

**Purpose**: Predictive maintenance & bearing failure prediction  
**Sessions**: 12-15 (Regression & Trees), 16-18 (Advanced Classification), 19-20 (Unsupervised)  
**Assignment**: A5 (Classification + Unsupervised) — Primary use

---

## **Dataset Information**

| Property | Value |
|----------|-------|
| **Name** | Bearing Failure / Predictive Maintenance / CMAPSS |
| **Type** | Time series or sensor data |
| **Format** | CSV or TXT |
| **Size** | TBD (to be determined during download) |
| **Rows** | TBD (typically 100K-1M) |
| **Columns** | TBD (~20-30 sensor features) |
| **License** | TBD (check source) |

---

## **Download Instructions**

**Source URL**: (To be filled in)
- [ ] Kaggle: https://www.kaggle.com/search?q=bearing+failure (TBD)
- [ ] NASA CMAPSS: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/ (potential source)
- [ ] UCI ML Repository: https://archive.ics.uci.edu/ml/... (TBD)
- [ ] Other: ________________

**Steps**:
1. Visit the source URL above
2. Download the dataset file(s)
3. Place in this folder: `mlcoep/datasets/raw/predictive_maintenance/`
4. Verify file name(s): `bearing_data.csv` or similar
5. Check file size & row count

**Expected file**: 
```
predictive_maintenance/
└── bearing_data.csv
```

---

## **Data Dictionary**

| Column Name | Data Type | Range / Values | Description |
|-------------|-----------|----------------|-------------|
| (TBD) | (TBD) | (TBD) | (TBD) |
| (TBD) | (TBD) | (TBD) | (TBD) |
| **Target** | Binary/Multiclass | {0, 1} or {Normal, Failed} | Machine/bearing status |

*To be filled in once dataset is downloaded*

---

## **Basic Statistics**

```
Total Rows: TBD
Total Columns: TBD
Missing Values: TBD
Duplicates: TBD
Target Classes: TBD (e.g., Normal / Failed)
Class Distribution: TBD (balanced or imbalanced?)
```

---

## **Usage in Course**

### **Session 12-15 (Regression & Trees)**
- Data exploration & preparation
- Feature correlation analysis
- Train baseline tree models

### **Session 16-18 (Advanced Classification)**
- **Classification problem**: Predict bearing failure
- Train SVM, Naive Bayes, K-NN models
- Compare classifier performance
- Handle class imbalance (if present)

### **Session 19-20 (Unsupervised)**
- **A5 Assignment**: K-means clustering on equipment behavior
- Cluster equipment by sensor patterns
- Identify anomalous patterns
- Reflect on supervised vs unsupervised methods

---

## **License & Citation**

**License**: (To be filled in — check source)
- Public dataset: Yes / No
- Commercial use allowed: Yes / No / Restricted
- Attribution required: Yes / No

**Citation**: (If applicable)
```
(To be filled in)
```

---

## **Notes**

- Dataset selected for: Predictive maintenance (bearing failure prediction)
- Challenge: Likely imbalanced (failures are rare) — discuss class imbalance in Session 18
- Time series aspect: May include temporal patterns (optional for unsupervised learning)
- Use case: Multi-class classification or binary classification (failure vs. normal)

---

**Status**: Placeholder — Awaiting dataset selection & download (by Aug 15, 2026)
