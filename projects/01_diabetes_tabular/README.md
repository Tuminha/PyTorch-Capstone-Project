# Project 01: Diabetes Prediction (Tabular Data)

## Problem Statement & Success Metrics

### Problem
Predict the presence of diabetes (binary classification) from health factors including BMI, age, exercise habits, and other demographic/clinical variables.

### Questions to Explore
- Which health factors are most predictive of diabetes?
- Can we achieve acceptable screening performance (high recall for positive class)?
- How do classical baselines compare to a neural network?

### Success Metrics
- **Primary:** ROC-AUC ≥ 0.75
- **Secondary:** F1-score for positive class ≥ 0.60
- **Interpretability:** Feature importance analysis

---

## Data Description

### Source
- **Dataset:** Diabetes BRFSS 2015 (CDC Behavioral Risk Factor Surveillance System)
- **Location:** `../data/diabetes_BRFSS2015.csv`
- **Size:** ~250,000+ records

### Key Fields
- **Target:** Diabetes (Yes/No)
- **Features:** Age, BMI, exercise, education, income, health status, etc.
- **Type:** Mixed (numeric + categorical)

### Target Distribution
- Check notebook `02_load_and_inspect.ipynb` for class balance
- Expect imbalance favoring "No diabetes"

---

## Method Overview

### Baseline Models
1. **Logistic Regression** — Linear baseline with class weights
2. **Random Forest** — Non-linear baseline with feature importance

### PyTorch Model
- **Architecture:** Feed-Forward Network (FFN)
- **Layers:** Input → Hidden(s) → Dropout → Output
- **Loss:** BCEWithLogitsLoss (numerically stable)
- **Training:** Adam optimizer, early stopping, validation monitoring

### Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC
- **Visualizations:** ROC curve, PR curve, Confusion Matrix
- **Optional:** Threshold sweep for operating point selection

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

## Results Snapshot

*Fill this section after completing all notebooks*

### Final Metrics

| Model | Accuracy | ROC-AUC | F1 (Positive) | Precision | Recall |
|-------|----------|---------|---------------|-----------|--------|
| Logistic Regression | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| PyTorch FFN | - | - | - | - | - |

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

