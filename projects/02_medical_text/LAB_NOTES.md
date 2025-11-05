# 🧪 Lab Notes - Project 02: Medical Text Classification

## 📊 Project Overview

**Goal:** Build a medical specialty classifier for 16K+ medical Q&A pairs using PyTorch and transformers

**Duration:** ~2-3 weeks (November 2024)

**Final Result:** BioBERT model achieving **83.79% F1 Macro** on 13-class classification (2.16x better than baseline!)

---

## 🗺️ Journey Summary

This project took me from **unsupervised taxonomy construction** → **baseline neural network** → **state-of-the-art transformer** → **production-ready model**.

**Key milestones:**
1. **Notebook 00:** Built specialty taxonomy using k-means clustering (ML-first approach)
2. **Notebooks 01-03:** Created NLP preprocessing pipeline (tokenization, vocab, encoding)
3. **Notebook 04:** Trained baseline classifier, discovered overfitting
4. **Notebook 05:** Fine-tuned BioBERT (83.73% validation F1 in 1 epoch!)
5. **Notebook 06:** Test evaluation revealed baseline collapse (39% F1) vs BioBERT excellence (84% F1)

**Most important learning:** Transfer learning with domain-specific pre-trained models (BioBERT) is a game-changer for specialized tasks!

---

## 🎓 Overall Lessons Learned

### **1. Transfer Learning > Training from Scratch**
- BioBERT (pre-trained on PubMed): **83.79% F1**
- Baseline (trained from scratch): **38.73% F1**
- **2.16x improvement** from using pre-trained knowledge!

### **2. Overfitting is Sneaky**
- Validation metrics can mislead (baseline: 63% → 39%)
- Train/val curves showed overfitting, but magnitude was hidden
- **Always evaluate on test set!**

### **3. Class Imbalance Requires Special Handling**
- Baseline failed completely on 3 rare classes (F1 = 0.00)
- BioBERT handled them better (F1 ≥ 0.42) due to pre-trained knowledge
- **Strategies:** Oversampling, class weights, focal loss, or just use transformers!

### **4. OOV Problem is Real**
- 48% of samples had unknown words in baseline vocab
- Medical terms like "retinopathy" → `<UNK>` → lost information
- **Subword tokenization (BERT) solves this!**

### **5. Proper Evaluation Pipelines Matter**
- Index-based splitting ensures fair comparison
- Per-class metrics reveal hidden weaknesses
- Visualizations make results interpretable
- **Professional ML engineering = trust in results**

### **6. Computational Constraints are Real**
- BioBERT on CPU: 2.5 hours/epoch (vs 5 min for baseline)
- GPU would be 100x faster (5-10 min total for 5 epochs)
- **Workarounds:** Sampling, progress tracking, early stopping
- **For production:** GPU access is essential

### **7. Simple Baselines Have Value**
- Despite terrible performance (39% F1), baseline:
  - ✅ Quantified improvement magnitude
  - ✅ Highlighted vocabulary limitations
  - ✅ Proved transformers aren't overkill
  - ✅ Trained in 5 minutes (vs 2.5 hours)
- **Always start with simple baselines!**

### **8. Documentation is Learning**
- Writing reflections forced me to understand *why* things worked
- README as portfolio showcase
- Visualizations communicate results effectively
- **Good documentation = proof of understanding!**

---

## 🚀 Skills Acquired

### **NLP & Text Processing:**
- ✅ Text cleaning & preprocessing
- ✅ Tokenization (word-level & subword)
- ✅ Vocabulary building
- ✅ Text encoding & padding
- ✅ Handling OOV (Out-Of-Vocabulary) words

### **Unsupervised Learning:**
- ✅ K-means clustering
- ✅ Sentence embeddings (BioBERT)
- ✅ Hyperparameter selection (elbow curve, silhouette score)
- ✅ UMAP dimensionality reduction

### **Supervised Learning (PyTorch):**
- ✅ Building `nn.Module` classes
- ✅ Embedding layers
- ✅ Training loops (5-step mantra)
- ✅ Loss functions (`CrossEntropyLoss`)
- ✅ Optimizers (Adam, AdamW)
- ✅ Custom `Dataset` and `DataLoader` implementation

### **Transfer Learning & Transformers:**
- ✅ Hugging Face `transformers` library
- ✅ Pre-trained model loading (`AutoModel`, `AutoTokenizer`)
- ✅ Fine-tuning BERT models
- ✅ BERT input format (input_ids, attention_mask)

### **Model Evaluation:**
- ✅ Train/val/test splitting (stratified)
- ✅ Metrics: accuracy, precision, recall, F1 (macro/weighted)
- ✅ Per-class performance analysis
- ✅ Overfitting detection (train vs val curves)

### **ML Engineering:**
- ✅ Index-based splitting for fair comparison
- ✅ Model saving & loading (`torch.save`, `load_state_dict`)
- ✅ Progress tracking for long-running jobs
- ✅ CPU optimization strategies
- ✅ Reproducible pipelines (`random_state=42`)

---

## 💭 Personal Reflections

### **What Surprised Me Most:**
The magnitude of improvement from transfer learning! I expected BioBERT to be better, but **2.16x better** (116% relative improvement) was mind-blowing. The fact that 1 epoch of BioBERT (83.73%) crushed 12 epochs of baseline (63.01%) showed that pre-trained knowledge is irreplaceable.

### **Hardest Challenge:**
**CPU training BioBERT for 2.5 hours per epoch!** The "32 minutes, 0 epochs" moment was genuinely scary—I thought my code had crashed!

### **Proudest Moment:**
When test results showed **BioBERT: 83.79% F1 (same as validation!)** while **baseline collapsed to 38.73%**. This validated my entire pipeline and proved this is **production-ready work!** 🎉

### **Most Valuable Lesson:**
**"Always test on held-out data."** Baseline looked okay at 63% validation F1, but test revealed the truth (39% F1). The discipline of proper train/val/test splits saved me from shipping a terrible model!

### **Skills I'm Most Confident In Now:**
- ✅ PyTorch training loops (could write from memory!)
- ✅ Hugging Face transformers (comfortable fine-tuning any model)
- ✅ Proper ML evaluation (train/val/test, stratification, fair comparison)
- ✅ Overfitting detection and mitigation
- ✅ Professional documentation and visualization

---

## 🎯 Future Improvements

### **Short-term:**
1. **Confusion matrix analysis:** Identify which specialty pairs BioBERT confuses most
2. **Error inspection:** Read 10-20 misclassified samples to understand failure modes
3. **Confidence thresholding:** Flag predictions with confidence < 0.7 for human review

### **Medium-term:**
1. **Hyperparameter tuning:** Try different learning rates, dropout rates, batch sizes
2. **Train for more epochs (with GPU):** Current 1 epoch → 3-5 epochs (might reach 85-86%)
3. **Try other medical transformers:** ClinicalBERT, PubMedBERT, Bio_ClinicalBERT

### **Long-term:**
1. **Data augmentation:** Paraphrase with GPT, back-translation, expand to 50K+ samples
2. **Ensemble methods:** Combine BioBERT + ClinicalBERT + domain rules
3. **Deployment:** Build REST API, containerize, deploy to cloud

---

## 🎉 Final Thoughts

**Project 02: Medical Text Classification is COMPLETE!**

From building a taxonomy with k-means, to training a baseline neural network, to fine-tuning a state-of-the-art transformer, this project covered the full spectrum of modern NLP.

**The result:** A production-ready model (BioBERT, 83.79% F1) that can accurately route medical questions to appropriate specialties.

**The learning:** Transfer learning, proper evaluation, and professional ML engineering.

**The pride:** This is portfolio-worthy work that demonstrates real ML expertise! 🏆

---

**Date Completed:** November 5, 2024  
**Signed:** Francisco Teixeira Barbosa (@Tuminha)  
**Achievement Unlocked:** 🏅 Medical NLP Engineer

**This is portfolio-worthy work!** 🎉🚀

