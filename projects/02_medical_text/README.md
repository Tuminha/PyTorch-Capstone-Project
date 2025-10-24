# Project 02: Medical Text Classification

## Problem Statement & Success Metrics

### Problem
Classify medical Q&A text into categories (e.g., Cardiology, Dermatology, etc.) using NLP techniques.

### Questions to Explore
- Can a simple embedding-based classifier achieve reasonable performance?
- How much improvement do transformers provide over baselines?
- What are the optimal preprocessing choices?

### Success Metrics
- **Primary:** Macro-F1 ≥ 0.65
- **Secondary:** Per-class F1 scores
- **Interpretability:** Error analysis by category

---

## Data Description

### Source
- **Dataset:** MedQuad (Medical Q&A)
- **Location:** `../data/medquad.csv`
- **Size:** ~2,000+ medical questions

### Key Fields
- **Target:** Category label (multi-class)
- **Features:** Question text + answer text
- **Type:** Text (strings)

### Target Distribution
- Check notebook `01_project_scope_and_data.ipynb` for category balance
- Expect some imbalance across medical specialties

---

## Method Overview

### Baseline Model
1. **Mean-Pooled Embeddings** — Simple word embeddings averaged + Linear classifier

### PyTorch Model
- **Architecture:** Fine-tuned Transformer (TinyBERT/RoBERTa/DistilBERT)
- **Training:** CrossEntropyLoss, AdamW optimizer
- **Approach:** Fine-tune pretrained model on medical text

### Evaluation
- **Metrics:** Macro-F1, per-class F1, Accuracy
- **Visualizations:** Confusion Matrix, Error Analysis
- **Focus:** Identify which categories confuse the model

---

## How to Run

### Prerequisites
- Complete root-level setup (see main README)
- Install transformers: `pip install transformers`

### Notebook Order
1. `01_project_scope_and_data.ipynb` — Define categories & metrics
2. `02_load_clean_tokenize.ipynb` — Text preprocessing
3. `03_vocab_encoding_padding.ipynb` — Build vocab, encode sequences
4. `04_baseline_classifier.ipynb` — Train simple baseline
5. `05_transformer_setup_train.ipynb` — Fine-tune transformer
6. `06_eval_and_error_analysis.ipynb` — Final evaluation & errors
7. `99_lab_notes.ipynb` — Reflections (ongoing)

### Expected Outputs
- **Data:** Cleaned & tokenized text
- **Models:** Baseline + Fine-tuned transformer
- **Metrics:** Comparison table
- **Visualizations:** Confusion matrix, error patterns

---

## Results Snapshot

*Fill this section after completing all notebooks*

### Final Metrics

| Model | Macro-F1 | Accuracy | Notes |
|------|----------|----------|-------|
| Baseline (Mean Pool) | - | - | - |
| Fine-tuned Transformer | - | - | - |

### Key Findings
- [ ] Item 1
- [ ] Item 2
- [ ] Item 3

---

## Limitations & Ethics

### Data Limitations
- Small dataset may limit generalization
- Medical terminology varies across sources
- Possible annotation inconsistencies

### Model Limitations
- Not validated on clinical text
- May not generalize to unseen categories
- Limited explainability

### Ethical Considerations
- **Do not use for clinical decisions**
- Text classification carries risk of misinterpretation
- Consider bias in medical terminology
- Respect patient privacy in text data

---

## Next Steps

- [ ] Experiment with different transformer architectures
- [ ] Add data augmentation (back-translation, paraphrasing)
- [ ] Implement attention visualization
- [ ] Expand dataset with more medical texts
- [ ] Add explainability (SHAP for text)

---

## References

- MedQuad Dataset
- Hugging Face Transformers Documentation
- PyTorch NLP Tutorials

