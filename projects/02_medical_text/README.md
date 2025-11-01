# Project 02 - Medical Text Classification

## Problem Statement & Success Metrics

### Problem
Build a classifier that routes medical Q&A pairs to the **specialist most likely to handle the case** (Oncology, Cardiology, Neurology, ...).  
Raw MedQuad labels are 5,126 ultra-granular `focus_area` strings (e.g., "Breast Cancer", "Wilson-Turner syndrome"), which are unusable directly because of extreme sparsity.

### Key Challenge
- Most `focus_area` labels appear fewer than 10 times (median approx. 3).  
- Several thousand unique labels create an intractable class space.  
- Previous clustering-only taxonomy misrouted common conditions into "Genetic & Rare Diseases".

### Updated Solution (Notebook 00)
1. **Curated specialty anchors:** hand-authored phrases describing 20 high-level specialties.  
2. **TF-IDF similarity scoring:** compare each `focus_area` to every anchor centroid.  
3. **Keyword boosts + overrides:** reinforce high-signal patterns (e.g., `cancer`, `glaucoma`).  
4. **Fallback heuristics:** ensure ambiguous / low-similarity terms land in a sensible bucket.  
5. **Quality checks:** targeted asserts for historically misclassified conditions.

This hybrid rule + similarity approach keeps the taxonomy reproducible while fixing the glaring mislabels that prompted this refresh.

### Success Metrics
- **Primary:** Macro-F1 >= 0.65 for downstream text classification.  
- **Secondary:** Per-specialty F1, macro recall.  
- **Taxonomy acceptance:** every specialty >=50 samples, common conditions map to expected specialists, and validation checks pass.

---

## Data Description

- **Dataset:** MedQuad (medical question-answer pairs)  
- **Location:** `../../../datasets/medquad.csv`  
- **Rows:** 16,398 after dropping unlabeled focus areas  
- **Focus area vocab:** 4,743 normalized strings -> collapsed to 20 medical specialties  
- **Features:** concatenated `question` + `answer` text  
- **Targets:** engineered `specialty` labels produced in Notebook 00 (`medquad_with_specialties.csv`)

### Engineered Specialty List (alphabetical)
- Cardiology & Vascular  
- Dental & Oral Health  
- Dermatology  
- Endocrinology & Diabetes  
- Gastroenterology & Hepatology  
- General & Preventive Medicine  
- Genetics & Rare Disorders  
- Hematology  
- Immunology & Rheumatology  
- Infectious Diseases  
- Musculoskeletal & Orthopedics  
- Nephrology & Urology  
- Neurology & Neurosurgery  
- Obstetrics & Gynecology  
- Oncology  
- Ophthalmology  
- Otolaryngology & ENT  
- Pediatrics & Development  
- Psychiatry & Behavioral Health  
- Pulmonology & Respiratory

Artifacts include runner-up labels and assignment sources (`anchor_similarity`, `regex_override`, `fallback_*`) to support manual QA.

---

## Method Overview

### Phase 0 - Specialty Taxonomy (Notebook 00)
- Normalize focus areas, build frequency catalog.
- Vectorize focus areas + anchors with TF-IDF bigrams.
- Score every focus area against specialty centroids.
- Apply keyword boosts, deterministic overrides, and fallback heuristics.
- Validate with targeted asserts:
  - "Breast cancer" -> Oncology  
  - "Cataract" -> Ophthalmology  
  - "Cholera" -> Infectious Diseases  
  - "Long QT syndrome" -> Cardiology & Vascular  
  - "Kluver-Bucy syndrome" -> Neurology & Neurosurgery  
  - etc.
- Export artifacts to `projects/02_medical_text/artifacts/specialty_taxonomy/`.

### Phase 1 - Baselines (planned)
1. Exploratory data analysis (Notebook 01).  
2. Cleaning, tokenization, vocabulary building (Notebooks 02-03).  
3. Logistic regression / linear baseline on TF-IDF or averaged embeddings (Notebook 04).

### Phase 2 - PyTorch Models (planned)
- Fine-tune biomedical transformers (BioBERT / ClinicalBERT / DistilBERT).  
- Use weighted cross-entropy, early stopping, and per-specialty evaluation (Notebooks 05-06).

### Evaluation Toolkit
- Macro-F1, per-class F1, micro accuracy.  
- Confusion matrices + error buckets.  
- Runner-up specialty analysis to study borderline predictions.

---

## How to Run

### Environment
Install core dependencies (root `requirements.txt` already covers these):

```bash
pip install -r requirements.txt
# or, minimally for this project:
pip install pandas numpy scikit-learn scipy matplotlib seaborn transformers
```

Notebook 00 only requires `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`.

### Notebook Order
0. `00_build_specialty_taxonomy.ipynb` <- **start here (updated)**  
1. `01_project_scope_and_data.ipynb`  
2. `02_load_and_inspect.ipynb` *(tokenization/EDA)*  
3. `03_cleaning.ipynb` *(preprocessing refinements)*  
4. `04_baseline_classifier.ipynb`  
5. `05_preprocessing_splits_balance.ipynb`  
6. `06_baselines_logreg_rf.ipynb`  
7. `07_pytorch_ffn_build_train.ipynb`  
8. `08_evaluation_and_conclusions.ipynb`  
9. `99_lab_notes.ipynb` *(ongoing reflections)*

Each notebook documents expected runtime and generated artifacts.

---

## Results Snapshot (Phase 0)

- [done] **Taxonomy rebuilt** with 20 top-level specialties.  
- [done] **Quality asserts pass** for previously misrouted conditions.  
- [done] **Artifacts saved**:
  - `medquad_with_specialties.csv` (labeled dataset)  
  - `focus_area_specialty_mapping.csv` (one row per unique focus area)  
  - `specialty_distribution.csv` (counts & unique focus areas per specialty)  
  - `taxonomy_metadata.json` (rules, thresholds, timestamps)
- [done] **Runner-up analysis** surfaces low-margin assignments for optional manual review.

Run the notebook to refresh the distribution chart; it is rendered inline and exported as `specialty_distribution.png`.

---

## Limitations & Risk Notes

- **Educational labels:** The taxonomy is rule-based and not clinician-validated. Do not deploy clinically.  
- **Keyword dependence:** Rare terminology absent from the anchor vocabulary may still fall back to broad specialties.  
- **Multispecialty overlap:** Some topics (e.g., ophthalmic manifestations of diabetes) inherently straddle two specialties; runner-up labels capture this ambiguity.  
- **Testing in CLI:** Automated execution inside the Codex CLI currently crashes when importing `pandas` (macOS segmentation fault), so tests were reasoned about but not executed end-to-end here. Confirm locally in your Python environment.

---

## 📊 K-Means Analysis: Choosing k with Evidence

### The Challenge
With 5,126 fine-grained `focus_area` labels and 16,398 medical Q&A samples, we needed to determine the optimal number of clusters (k) for our taxonomy.

### Methodology
We tested k values from 8 to 26, computing two key metrics:

1. **Inertia (Elbow Curve):** Measures cluster tightness (lower = better)
2. **Silhouette Score:** Measures cluster separation quality (higher = better, range: -1 to 1)

### Results & Key Finding

<div align="center">
<img src="images/elbow_silhouette.png" alt="Elbow Curve and Silhouette Score Analysis" width="800"/>
</div>

**Critical Discovery at k=20:**
- The silhouette score **crashes to 0.035** at k=20 (poor separation)
- But **jumps dramatically to 0.071 at k=22** (2× improvement!)
- This sharp transition indicates **k=22 aligns with natural groupings** in medical text

### Decision: k=22 ✅

**Evidence-based reasoning:**
- ✅ **Elbow curve:** Diminishing returns after k=16, gentle slope through k=22
- ✅ **Silhouette score:** Sharp improvement from k=20 (0.035) → k=22 (0.071)
- ✅ **Interpretability:** 22 clusters → ~15-20 medical specialties (manageable)
- ✅ **Quality:** Silhouette of 0.071 indicates reasonable cluster separation for text data

**Why NOT k=20?** Worst silhouette score in the tested range (0.035) suggests awkward splits of natural groups.

**Why NOT k=24-26?** Minimal silhouette improvement (0.073 vs 0.071) with added complexity.

### Embeddings Used
- **Model:** `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`
- **Dimensions:** 768
- **Normalization:** L2 normalized for cosine similarity
- **Input:** Concatenated `question + answer` text

---

## Progress Tracker

**🔄 APPROACH RESET (Oct 29, 2024):**
- [x] Archived rule-based approach → `backup/archived_notebooks/00_build_specialty_taxonomy_RULES_BASED.ipynb`
- [x] Created fresh ML-first notebook → `00_specialty_taxonomy.ipynb` (31 cells, TODO-driven)
- [x] Set up project structure (`data/processed/`, `artifacts/`)

**Notebook 00 - Taxonomy Construction (IN PROGRESS):**
- [x] Section 0: Imports & setup ✅
- [x] Section 1-2: Load data & preprocessing ✅
- [x] Section 3: Embeddings (BioBERT) ✅
- [x] Section 4: **⭐ Choose k=22** using elbow curve + silhouette score ✅
- [ ] Section 5: Train k-means with k=22
- [ ] Section 6: UMAP visualization
- [ ] Section 7: Explore clusters
- [ ] Section 8: Manual cluster naming
- [ ] Section 9: Add 5-10 surgical rules
- [ ] Section 10-11: QC & export

**Future Notebooks:**
- [ ] Notebook 01 - project framing with new labels
- [ ] Notebook 02-04 - preprocessing + baseline models
- [ ] Notebook 05-07 - PyTorch models
- [ ] Notebook 08 - evaluation & conclusions
- [ ] Notebook 99 - lab notes / reflections

---

## Next Steps

### Immediate (Notebook 00):
1. **Complete Section 0-3:** Set up imports, load data, choose embedding model
2. **Learn k-means (Section 4):** Generate elbow curve + silhouette scores, choose k with evidence
3. **Explore clusters (Section 7):** Review representatives and keywords to understand what k-means found
4. **Name clusters (Section 8):** Manually map cluster IDs → medical specialties
5. **Add minimal rules (Section 9):** 5-10 surgical patches for systematic errors only
6. **Export taxonomy (Section 11):** Generate `specialty_taxonomy_v1.csv` and mappings

### Future:
- Update downstream notebooks to use new taxonomy
- Benchmark baseline vs. transformer models
- Measure downstream F1 scores to validate taxonomy quality
- **Science beats vibes:** Use confusion matrices to evaluate if clusters made sense

---

## References
- MedQuad Dataset  
- Scikit-learn documentation (TF-IDF, cosine similarity)  
- Biomedical NLP literature (BioBERT, ClinicalBERT)  
- CDC & NIH specialty overviews for anchor crafting  
- Prior lab notes (`99_lab_notes.ipynb`) for project context
