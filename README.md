# Reddit Depression Detection: Comparative Statistical & Ensemble Learning Analysis

## Overview
This repository implements an automated statistical machine learning framework to classify Reddit posts into depressive (`y=1`) versus non-depressive (`y=0`) states. It addresses fundamental theoretical trade-offs regarding interpretability vs. capacity in high-dimensional spaces ($p \gg n$) and variance reduction techniques across informal digital texts.

## Team Members & Responsibilities
- **Kanchan Neupane**: Data processing, tokenization, TF-IDF extraction, Sparse PCA
- **Krishiyana Bhakta**: Baseline linear & discriminant models (Naive Bayes, LDA, Logistic Regression)
- **Prayog Paudel**: Kernel & ensemble methods (SVMs, Random Forest, XGBoost, Stacking Classifier)
- **Sulakshyana Ghimire**: Evaluation pipeline, Stratified 5-Fold CV, SHAP interpretability analysis

---

## Getting Started

### 1. Prerequisites & Installation

Clone the repository and set up a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Interactive CLI Inference
python predict.py
