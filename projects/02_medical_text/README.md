# Project 02: Medical Text Classification

## Problem Statement & Success Metrics

### Problem
Classify medical Q&A text into **medical specialties** (e.g., Oncology, Cardiology, Neurology, etc.) using NLP techniques.

**Key Challenge:** The raw dataset contains 5,126 fine-grained `focus_area` labels (e.g., "Breast Cancer", "Lung Cancer", "Prostate Cancer") with severe class imbalance. **Solution:** Build a data-driven taxonomy to group related conditions into 15-20 medical specialties using embeddings + clustering.

### Questions to Explore
- How can we create a meaningful specialty taxonomy from 5,126 fine-grained labels?
- Can semantic embeddings (BioBERT) + clustering discover natural medical groupings?
- How much improvement do transformers provide over simple embedding baselines?
- What are the optimal preprocessing choices for medical text?

### Success Metrics
- **Primary:** Macro-F1 ≥ 0.65 (across medical specialties)
- **Secondary:** Per-class F1 scores, especially for minority specialties
- **Interpretability:** Error analysis by specialty category
- **Taxonomy Quality:** Each specialty with ≥50 samples, semantically coherent

---

## Data Description

### Source
- **Dataset:** MedQuad (Medical Q&A)
- **Location:** `../../../datasets/medquad.csv`
- **Size:** 16,412 samples (medical question-answer pairs)
- **Original Labels:** 5,126 unique `focus_area` categories
- **Final Labels:** 15-20 medical `specialty` categories (created via taxonomy)

### Key Fields
- **Target (Original):** `focus_area` — Fine-grained medical condition (5,126 unique values)
- **Target (Engineered):** `specialty` — Medical specialty grouping (15-20 categories)
- **Features:** `question` + `answer` text (combined for classification)
- **Type:** Text (strings)

### Target Distribution Challenge
**Original Problem:** 5,126 classes with extreme imbalance
- Most focus_areas have <10 samples
- Median: ~3 samples per focus_area
- **Impossible to train a classifier!**

**Solution (Notebook 00):** Data-driven taxonomy using:
1. BioBERT embeddings (semantic similarity)
2. Clustering (HDBSCAN / k-means)
3. Manual labeling → 15-20 medical specialties
4. Zero-shot validation

**Result:** Manageable multi-class problem with balanced specialties

---

## Method Overview

### Phase 0: Taxonomy Construction (Notebook 00) ⭐ NEW
**Problem:** 5,126 focus_areas → too many classes, too few samples  
**Solution:** Data-driven taxonomy using unsupervised learning

**Steps:**
1. **Embed** focus_area strings with BioBERT
2. **Reduce** dimensionality with UMAP (visualization)
3. **Cluster** with HDBSCAN/k-means (discover groupings)
4. **Label** clusters as medical specialties (manual + zero-shot validation)
5. **Export** `df_with_specialty.parquet` for downstream use

**Output:** 15-20 balanced medical specialties ready for classification

---

### Phase 1: Baseline Model (Notebooks 01-04)
1. **Mean-Pooled Embeddings** — Simple word embeddings averaged + Linear classifier

### Phase 2: PyTorch Transformer (Notebooks 05-06)
- **Architecture:** Fine-tuned Transformer (TinyBERT/RoBERTa/BioBERT/DistilBERT)
- **Training:** CrossEntropyLoss, AdamW optimizer, class weights
- **Approach:** Fine-tune pretrained model on medical specialty classification

### Evaluation
- **Metrics:** Macro-F1, Weighted F1, per-class F1, Accuracy
- **Visualizations:** Confusion Matrix, Error Analysis
- **Focus:** Identify which specialties confuse the model (e.g., Cardiology vs. Pulmonology)

---

## How to Run

### Prerequisites
- Complete root-level setup (see main README)
- **Essential packages:**
  ```bash
  pip install transformers sentence-transformers umap-learn hdbscan scikit-learn
  ```

### Notebook Order

**⚠️ IMPORTANT: Start with Notebook 00!**

0. **`00_build_specialty_taxonomy.ipynb`** ⭐ **START HERE**
   - Build data-driven medical specialty taxonomy
   - Transform 5,126 focus_areas → 15-20 specialties
   - Export `df_with_specialty.parquet`
   - **Time:** 2-3 hours (includes manual labeling)

1. **`01_project_scope_and_data.ipynb`**
   - Load specialty-labeled data
   - Define classification problem & metrics
   - **Time:** 30 minutes

2. **`02_load_clean_tokenize.ipynb`**
   - Text preprocessing & cleaning
   - **Time:** 30 minutes

3. **`03_vocab_encoding_padding.ipynb`**
   - Build vocabulary, encode sequences
   - **Time:** 45 minutes

4. **`04_baseline_classifier.ipynb`**
   - Train simple embedding-based baseline
   - **Time:** 45 minutes

5. **`05_transformer_setup_train.ipynb`**
   - Fine-tune pretrained transformer on specialties
   - **Time:** 1-2 hours (GPU recommended)

6. **`06_eval_and_error_analysis.ipynb`**
   - Final evaluation & error analysis
   - **Time:** 45 minutes

7. **`99_lab_notes.ipynb`**
   - Reflections & lessons learned (ongoing)

### Expected Outputs
- **Phase 0:** `df_with_specialty.parquet`, taxonomy mappings
- **Phase 1-2:** Cleaned & tokenized text, vocabularies
- **Models:** Baseline + Fine-tuned transformer
- **Metrics:** Comparison table (Macro-F1, per-specialty F1)
- **Visualizations:** Confusion matrix, UMAP clusters, error patterns

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

## Progress Status

**Current Phase:** Building specialty taxonomy (Notebook 00)

- [x] Created `00_build_specialty_taxonomy.ipynb` with full scaffold
- [ ] Complete Notebook 00 (taxonomy construction)
- [ ] Complete Notebook 01 (project scope)
- [ ] Complete Notebook 02 (tokenization)
- [ ] Complete Notebook 03 (encoding)
- [ ] Complete Notebook 04 (baseline)
- [ ] Complete Notebook 05 (transformer)
- [ ] Complete Notebook 06 (evaluation)
- [ ] Complete Notebook 99 (lab notes)

---

## Next Steps

**Immediate:**
- [ ] Complete taxonomy construction (Notebook 00)
- [ ] Validate specialty labels with zero-shot classification
- [ ] Export clean `df_with_specialty.parquet`

**Future Improvements:**
- [ ] Experiment with different transformer architectures (BioBERT, ClinicalBERT)
- [ ] Add data augmentation (back-translation, paraphrasing)
- [ ] Implement attention visualization (highlight medical terms)
- [ ] Expand dataset with more medical texts
- [ ] Add explainability (SHAP for text, attention weights)
- [ ] Test on held-out clinical notes (domain adaptation)

---

## References

### Datasets & Models
- **MedQuad Dataset** — Medical Q&A corpus
- **BioBERT** — Pretrained BERT for biomedical text
- **Hugging Face Transformers** — State-of-the-art NLP models

### Techniques & Libraries
- **UMAP** — Dimensionality reduction for visualization
- **HDBSCAN** — Density-based clustering algorithm
- **Sentence Transformers** — Framework for semantic embeddings
- **Zero-Shot Classification** — Model-based validation without training

### Documentation
- PyTorch NLP Tutorials
- Hugging Face Transformers Documentation
- Sentence-Transformers Documentation

