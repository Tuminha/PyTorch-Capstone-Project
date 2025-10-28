# PyTorch Capstone — Health Factors (Diabetes), Medical Text, Retinal DR

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Status](https://img.shields.io/badge/Status-Educational-yellow.svg)

**Learning Machine Learning through applied healthcare projects — one dataset at a time**

[🎯 Overview](#-capstone-overview) • [🚀 How to Use](#-how-to-use-this-repo) • [📊 Projects](#-projects) • [⚙️ Environment](#-environment)

</div>

---

## 🎯 Capstone Overview

This repository contains **three independent projects** exploring different ML paradigms using PyTorch:

1. **Diabetes Prediction (Tabular)** — Binary classification from health factors
2. **Medical Text Classification** — Multi-class categorization of clinical notes
3. **Retinal Diabetic Retinopathy** — Image classification for DR severity

Each project follows a **learning-first, reflection-heavy** approach. This is **educational material only** — not for clinical use.

### 📚 Pedagogy

- **Concept Primer** → **Objectives** → **Acceptance Criteria** → **Numbered TODO cells** → **Reflection prompts**
- Plain, beginner-friendly language
- No complete solutions — you build and learn through guided TODOs
- Consistent variable naming within each project
- Emphasis on understanding shapes, dtypes, and data flow

---

## 🚀 How to Use This Repo

### 1. Setup Environment

**Prerequisites:**
- Python ≥3.10
- Jupyter notebook or JupyterLab

**Install Dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision transformers jupyter
```

Or use the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Run Projects in Order

Each project has numbered notebooks (`01_*.ipynb`, `02_*.ipynb`, etc.). Follow them sequentially.

**Inside each notebook:**
1. Read the **Concept Primer** section
2. Review **Objectives** and **Acceptance Criteria**
3. Complete the **TODO** cells (follow hints)
4. Answer **Reflection** prompts
5. Save visualizations to `/images/` with descriptive names

### 3. Log Your Progress

After each notebook:
- Fill in the **Reflection** section
- Record what clicked, what confused you
- Note any decisions made

Use `99_lab_notes.ipynb` in each project for ongoing reflections.

---

## 📊 Projects

### Project 01: Diabetes Prediction (Tabular) ✅ COMPLETE

**Goal:** Predict diabetes status (No Diabetes, Prediabetes, Diabetes) from health factors (BMI, age, exercise, etc.)

**Approach:** 
- Two baselines (Logistic Regression, Random Forest)
- PyTorch Feed-Forward Network for multi-class classification
- Evaluation: Weighted F1, Macro F1, per-class metrics

**Notebooks:** 8 + lab notes  
**Time Invested:** ~10 hours

**Status:** ✅ **ALL NOTEBOOKS COMPLETE** (01-08)
- Data inspection, cleaning, EDA & visualization ✓
- Preprocessing & splits, baseline models ✓
- PyTorch FFN training & evaluation ✓
- Comprehensive reflections & conclusions ✓

**Results Highlight:** PyTorch model achieved **71.7% accuracy** (vs. LR: 64.4%, RF: 67.9%), with best per-class F1 scores across all three diabetes classes. Only model to successfully learn the minority Prediabetes class.

[📖 Project README](projects/01_diabetes_tabular/README.md) | [📊 Final Results](projects/01_diabetes_tabular/README.md#results-snapshot)

---

### Project 02: Medical Text Classification

**Goal:** Classify medical Q&A text into categories

**Approach:**
- Baseline: Mean-pooled embeddings + Linear classifier
- Fine-tuned Transformer (TinyBERT/RoBERTa)
- Evaluation: Macro-F1, per-class metrics, error analysis

**Notebooks:** 6 + lab notes  
**Estimated Time:** 6-10 hours

[📖 Project README](projects/02_medical_text/README.md)

---

### Project 03: Retinal Diabetic Retinopathy

**Goal:** Classify retinal images by DR severity (0-4)

**Approach:**
- Simple CNN scaffold
- Training with augmentations
- Evaluation: Weighted-F1, confusion matrix, threshold tuning

**Notebooks:** 5 + lab notes  
**Estimated Time:** 6-10 hours

[📖 Project README](projects/03_retinal_dr/README.md)

---

## ⚙️ Environment

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥3.10 | Runtime |
| NumPy | Latest | Numerical operations |
| Pandas | Latest | Data manipulation |
| Matplotlib | Latest | Visualization |
| Seaborn | Latest | Statistical plots |
| Scikit-learn | Latest | ML utilities & baselines |
| PyTorch | Latest | Deep learning |
| Torchvision | Latest | Image transforms & models |
| Transformers | Latest | NLP models (Project 02) |
| Jupyter | Latest | Notebook environment |

### Installation

```bash
# Option 1: Individual packages
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision transformers jupyter

# Option 2: From requirements.txt
pip install -r requirements.txt
```

### Reproducibility

Set random seeds for consistency:
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

---

## ⚖️ Ethics & Limitations

### Educational Use Only

- These models are **not validated for clinical use**
- Do **not** use predictions for diagnostic decisions
- Projects are designed for learning PyTorch and ML workflows

### Dataset Considerations

- **Class imbalance** exists in all datasets
- **Bias** may be present (demographic, sampling, annotation)
- **Data leakage** risks — carefully inspect preprocessing
- **Interpretability** is crucial in healthcare — use SHAP/LIME for explanations

### Responsible AI Principles

When working on healthcare ML:
1. Understand dataset collection methodology
2. Assess demographic representation
3. Document limitations transparently
4. Consider real-world operating contexts
5. Acknowledge uncertainty

---

## 📦 Deliverables

### Per Project

After completing a project, produce:

1. **Metrics Table** — Compare baselines vs. PyTorch model
2. **Visualizations** — ROC curves, confusion matrices, training plots
3. **Short Write-up** — Key findings, limitations, next steps

Save these under each project's folder.

### Reproducibility Notes

Document:
- Random seeds used
- Package versions (`pip freeze > requirements.txt`)
- Hardware (CPU/GPU, memory)
- Training time per epoch

---

## 🎓 Learning Journey

This capstone builds toward:
- **Tabular ML** — Feature engineering, encoding, scaling
- **NLP** — Tokenization, embeddings, transformers
- **Computer Vision** — CNNs, data augmentation, transfer learning
- **Evaluation** — Metrics, threshold tuning, calibration
- **Best Practices** — Splits, validation, reproducibility

Each project reinforces core concepts while introducing domain-specific nuances.

---

## 📝 Project Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── data/                          # Raw datasets (placeholders)
│   ├── diabetes/
│   ├── medtext/
│   └── retinal_dr/
├── images/                        # Exported charts/figures
├── src/                           # Utility placeholders
│   ├── __init__.py
│   ├── utils/
│   └── models/
└── projects/
    ├── 01_diabetes_tabular/
    │   ├── README.md
    │   └── notebooks/            # 8 notebooks + lab notes
    ├── 02_medical_text/
    │   ├── README.md
    │   └── notebooks/            # 6 notebooks + lab notes
    └── 03_retinal_dr/
        ├── README.md
        └── notebooks/            # 5 notebooks + lab notes
```

---

## 🤝 Contributing

This is a personal learning repository. Suggestions and improvements welcome via issues or pull requests.

---

## 📄 License

MIT License — feel free to use for your own learning journey.

---

## 🙏 Acknowledgments

- Datasets used are publicly available for educational purposes
- Special thanks to the PyTorch community for excellent documentation
- Healthcare AI practitioners who advocate for responsible development

---

<div align="center">

**⭐ Happy Learning! ⭐**  
*Building ML skills through applied healthcare projects* 🚀

</div>

