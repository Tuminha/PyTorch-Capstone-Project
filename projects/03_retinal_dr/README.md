# Project 03: Retinal Diabetic Retinopathy Detection

## Problem Statement & Success Metrics

### Problem
Classify retinal fundus images by diabetic retinopathy (DR) severity levels (0-4).

### Questions to Explore
- Can a simple CNN achieve reasonable DR classification?
- What are the optimal image augmentations?
- How do different architectures compare?

### Success Metrics
- **Primary:** Weighted-F1 ≥ 0.70
- **Secondary:** Per-class F1 scores
- **Clinical:** Screening vs diagnostics threshold analysis

---

## Data Description

### Source
- **Dataset:** IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **Location:** `../data/diabetic_retinopathy_images/`
- **Size:** ~500+ retinal images

### Key Fields
- **Target:** DR Severity (0-4)
  - 0: No DR
  - 1: Mild DR
  - 2: Moderate DR
  - 3: Severe DR
  - 4: Proliferative DR
- **Features:** Retinal fundus images (RGB)
- **Type:** Image (JPG)

### Target Distribution
- Check notebook `01_project_scope_and_data.ipynb` for class balance
- Expect severe imbalance (most cases are No DR or Mild)

---

## Method Overview

### Baseline Model
1. **Simple CNN** — Custom architecture with Conv blocks + FC head

### PyTorch Model
- **Architecture:** CNN with conv layers, ReLU, MaxPool
- **Training:** CrossEntropyLoss, Adam optimizer
- **Augmentation:** Random flips, rotations, color jitter (train only)

### Evaluation
- **Metrics:** Weighted-F1, per-class F1, Accuracy
- **Visualizations:** Confusion Matrix, Training Curves
- **Optional:** Threshold sweep for screening use case

---

## How to Run

### Prerequisites
- Complete root-level setup (see main README)
- Install torchvision: `pip install torchvision`

### Notebook Order
1. `01_project_scope_and_data.ipynb` — Define problem & metrics
2. `02_transforms_and_loaders.ipynb` — Image transforms & DataLoaders
3. `03_simple_cnn_scaffold.ipynb` — Build CNN architecture
4. `04_train_validate_metrics.ipynb` — Training loop & validation
5. `05_test_eval_and_thresholding.ipynb` — Final evaluation
6. `99_lab_notes.ipynb` — Reflections (ongoing)

### Progress Notes
- **Notebook 04 (Training & validation):**
  - Stratified train/val/test split after merging class 1→0; class-weighted loss to counter imbalance.
  - Early stopping (patience=5) retained epoch 3 checkpoint (**val acc 0.4242**, val loss 1.37).
  - Long 30-epoch run overfit badly (train acc → 0.80, val loss > 2.5). Plots: `images/training_validation_metrics_30_epochs_overfitting.png`, `images/training_validation_metrics.png`.
  - Next iterations will swap in a pretrained backbone + stronger augmentation to chase Weighted-F1 ≥ 0.70.

### Expected Outputs
- **Data:** Preprocessed images with augmentations
- **Model:** Trained CNN
- **Metrics:** Test set performance
- **Visualizations:** Confusion matrix, training curves

---

## Results Snapshot

| Model | Weighted-F1 | Accuracy | Notes |
|------|-------------|----------|-------|
| Simple CNN (scratch) | 0.43 | 0.52 | Classes 2 & 3 receive 0 predicted samples; relies on classes 0/1 |

### Key Findings
- Baseline CNN overfits after ~8 epochs despite early stopping safeguards.
- Severe class imbalance (support 15 & 10) collapses moderate/severe predictions; confusion matrix shows funnel into class 1.
- Transfer learning + augmentation and/or resampling needed before targeting Weighted-F1 ≥ 0.70.

### Operating Threshold
- **Chosen threshold:** Softmax argmax (no threshold tuning yet)
- **Rationale:** Baseline underperforms; hold off on screening thresholds until stronger model is in place.

---

## Limitations & Ethics

### Data Limitations
- Small dataset may limit generalization
- Single imaging device/scanner
- Possible annotation inconsistencies
- Geographic bias (Indian population)

### Model Limitations
- Not validated on external cohorts
- Simple CNN may miss subtle features
- No explainability (attention maps)

### Confusion Matrix

![Confusion matrix](images/confusion_matrix.png)

### Ethical Considerations
- **Do not use for clinical diagnosis**
- False negatives in severe DR are dangerous
- False positives cause patient anxiety
- Screening vs diagnostics contexts differ
- Population bias considerations

---

## Next Steps

- [ ] Transfer learning with pretrained ResNet/EfficientNet
- [ ] Add Grad-CAM for explainability
- [ ] Experiment with different augmentations
- [ ] Ensemble methods
- [ ] Cross-validation for robust metrics
- [ ] Focus on detecting severe cases (grouping 3+4)

---

## References

- IDRiD Dataset: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
- Diabetic Retinopathy Detection Literature
- PyTorch Vision Documentation

