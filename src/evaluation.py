import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import config

class ModelEvaluator:
    def __init__(self, n_splits=config.N_FOLDS):
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)

    def evaluate_cv(self, model, X, y):
        """Performs 5-Fold Stratified Cross-Validation."""
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        scores = cross_validate(model, X, y, cv=self.skf, scoring=scoring, n_jobs=-1)
        
        metrics_summary = {
            "Accuracy": f"{np.mean(scores['test_accuracy']):.3f} +/- {np.std(scores['test_accuracy']):.3f}",
            "Precision": np.mean(scores['test_precision']),
            "Recall": np.mean(scores['test_recall']),
            "F1-Score": np.mean(scores['test_f1']),
            "ROC-AUC": np.mean(scores['test_roc_auc'])
        }
        return metrics_summary
