# Reddit Depression Detection: Comparative Statistical & Ensemble Learning Analysis

## Overview
This repository implements an automated statistical machine learning framework to classify Reddit posts into depressive (`y=1`) versus non-depressive (`y=0`) states[cite: 1]. It addresses theoretical trade-offs regarding interpretability vs. capacity in high-dimensional spaces ($p \gg n$)[cite: 1] and sparsity techniques for informal digital texts[cite: 1].

## Team Members & Responsibilities
- **Kanchan Neupane**: Data processing, tokenization, TF-IDF, Sparse PCA
- **Krishiyana Bhakta**: Baseline models (Naive Bayes, LDA, Logistic Regression)
- **Prayog Paudel**: Ensemble methods (SVMs, Random Forest, XGBoost, Stacking)
- **Sulakshyana Ghimire**: Evaluation pipeline, Stratified CV, SHAP interpretability

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
