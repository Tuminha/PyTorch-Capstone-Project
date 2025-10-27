# Project 01: Diabetes Prediction (Tabular Data)

## Problem Statement & Success Metrics

### Problem
Predict diabetes status (**trinary classification**: No Diabetes, Prediabetes, Diabetes) from health factors including BMI, age, exercise habits, and other demographic/clinical variables.

### Questions to Explore
- Which health factors are most predictive of diabetes status?
- Can we achieve acceptable screening performance across all three classes?
- How do classical baselines compare to a neural network?
- How to handle severe class imbalance (84% vs 14% vs 2%)?

### Success Metrics
- **Primary:** Macro-averaged F1-score ≥ 0.60
- **Secondary:** Weighted ROC-AUC ≥ 0.75 (multi-class)
- **Per-class Performance:** Focus on minority classes (Prediabetes, Diabetes)
- **Interpretability:** Feature importance analysis

---

## Data Description

### Source
- **Dataset:** Diabetes BRFSS 2015 (CDC Behavioral Risk Factor Surveillance System)
- **Location:** `../data/diabetes_BRFSS2015.csv`
- **Size:** 253,680 records × 22 columns

### Key Fields
- **Target:** Diabetes (trinary classification: No Diabetes, Prediabetes, Diabetes)
- **Features:** Age, BMI, exercise, education, income, health status, etc.
- **Type:** Mixed (numeric + categorical)

### Target Distribution

The dataset has a **trinary diabetes classification** with severe class imbalance:

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| 0 | No Diabetes | 213,703 | 84% |
| 1 | Prediabetes | 4,631 | 2% |
| 2 | Diabetes | 35,346 | 14% |

![Class Distribution](images/class_balance.png)

**Key Observations:**
- Extreme imbalance (84% vs 14% vs 2%)
- Requires stratified sampling for train/val/test splits
- May need class weights or resampling techniques during training
- Focus on metrics beyond accuracy (F1, precision, recall, ROC-AUC)

### Data Quality
- **Missing Values:** None detected ✅
- **Data Types:** 4 numeric (BMI, GenHlth, MentHlth, PhysHlth), 18 object columns
- **Schema Issues:** Many object columns need encoding (Yes/No, categorical variables)

---

## Method Overview

### Baseline Models
1. **Logistic Regression** — Linear baseline with class weights
2. **Random Forest** — Non-linear baseline with feature importance

### PyTorch Model
- **Architecture:** Feed-Forward Network (FFN)
- **Layers:** Input → Hidden(s) → Dropout → Output
- **Loss:** CrossEntropyLoss (for multi-class classification)
- **Training:** Adam optimizer, early stopping, validation monitoring
- **Handling Imbalance:** Class weights or stratified sampling

### Evaluation
- **Metrics:** 
  - Macro-averaged: Precision, Recall, F1 across all classes
  - Weighted averages (accounts for class imbalance)
  - Per-class metrics (especially for minority classes)
- **Visualizations:** Multi-class confusion matrix, per-class ROC curves
- **Special Considerations:** Class imbalance handling strategies (class weights, stratified sampling)

---

## How to Run

### Prerequisites
- Complete root-level setup (see main README)
- Install dependencies: `pip install -r requirements.txt`

### Notebook Order
1. `01_project_goals_and_data.ipynb` — Define problem & metrics
2. `02_load_and_inspect.ipynb` — Load data, check dtypes, visualize target
3. `03_cleaning.ipynb` — Handle missing values, outliers, rename columns
4. `04_eda_visualization.ipynb` — Exploratory analysis, correlations
5. `05_preprocessing_splits_balance.ipynb` — Encode, scale, split, handle imbalance
6. `06_baselines_logreg_rf.ipynb` — Train & evaluate baselines
7. `07_pytorch_ffn_build_train.ipynb` — Build & train neural network
8. `08_evaluation_and_conclusions.ipynb` — Final evaluation & write-up
9. `99_lab_notes.ipynb` — Reflections (ongoing)

### Expected Outputs
- **Data:** Cleaned DataFrame in memory after notebook 03
- **Features:** Encoded/scaled train/val/test splits after notebook 05
- **Models:** Trained objects in memory
- **Metrics:** Tables comparing all models
- **Plots:** ROC curves, confusion matrices, training curves

---

## Progress Status

### ✅ Completed
- [x] **Project Goals & Data** (Notebook 01) — Problem definition and metrics established
- [x] **Load and Inspect** (Notebook 02) — Data loaded, schema analyzed, class distribution documented
  - Dataset: 253,680 rows × 22 columns
  - Trinary classification confirmed (No Diabetes, Prediabetes, Diabetes)
  - Severe class imbalance identified (84% / 14% / 2%)
  - No missing values detected
  - Schema issues identified (18 object columns need encoding)
- [x] **Cleaning** (Notebook 03) — Data cleaned and outliers handled
  - Columns renamed to snake_case
  - Categorical variables intentionally kept as objects for proper encoding later
  - BMI extreme outliers capped at 60 (physiologically reasonable upper limit)
  - MentHlth and PhysHlth verified as valid skewed distributions (0-30 days)
  - GenHlth verified within valid range (1-5)
- [x] **EDA & Visualization** (Notebook 04) — Exploratory analysis completed
  - Correlation heatmap: GenHlth ↔ PhysHlth (0.52), MentHlth ↔ PhysHlth (0.35)
  - BMI distributions differ significantly across diabetes groups
  - General health ratings worsen from no diabetes (2) → prediabetes/diabetes (3-4)
  - Surprising finding: Prediabetes shows highest physical health burden
  - 8 visualizations generated and saved

### 🚧 In Progress
- [ ] **Preprocessing** (Notebook 05)
- [ ] **Baselines** (Notebook 06)
- [ ] **PyTorch Model** (Notebook 07)
- [ ] **Evaluation** (Notebook 08)

---

## Results Snapshot

*Fill this section after completing all notebooks*

### Final Metrics

| Model | Accuracy | Weighted F1 | Macro F1 | Class 0 F1 | Class 1 F1 | Class 2 F1 |
|-------|----------|-------------|-----------|------------|------------|------------|
| Logistic Regression | - | - | - | - | - | - |
| Random Forest | - | - | - | - | - | - |
| PyTorch FFN | - | - | - | - | - | - |

*Class 0: No Diabetes, Class 1: Prediabetes, Class 2: Diabetes*

### Key Findings
- [ ] Item 1
- [ ] Item 2
- [ ] Item 3

### Operating Threshold
- **Chosen threshold:** 0.XX
- **Rationale:** [Explain why this threshold makes sense for screening vs. diagnostics]

---

## Limitations & Ethics

### Data Limitations
- Self-reported survey data (recall bias)
- Possible confounding variables not included
- Sample may not represent all populations equally

### Model Limitations
- Not validated on external cohorts
- No causal inference (associations only)
- Limited by training data quality

### Ethical Considerations
- **Do not use for clinical diagnosis**
- Screening vs. diagnostics trade-offs
- False negatives (missed diabetes) may have serious consequences
- False positives (over-treatment) create patient anxiety
- Consider demographic fairness in predictions

---

## Next Steps

- [ ] Feature engineering (polynomial, interactions)
- [ ] Model explainability (SHAP/LIME analysis)
- [ ] Cross-validation for robust metrics
- [ ] Ensemble methods
- [ ] Deploy as educational demo (not for clinical use)

---

## References

- BRFSS: https://www.cdc.gov/brfss/
- Diabetes Classification Literature
- PyTorch Documentation: https://pytorch.org/docs/

