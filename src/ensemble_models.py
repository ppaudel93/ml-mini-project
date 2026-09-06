from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import config

def get_advanced_models():
    """Returns kernelized, tree ensemble, and gradient boosting models."""
    models = {
        "SVC (Linear)": SVC(kernel='linear', probability=True, random_state=config.RANDOM_SEED),
        "SVC (RBF)": SVC(kernel='rbf', probability=True, random_state=config.RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=config.RANDOM_SEED),
        "XGBoost Baseline": XGBClassifier(eval_metric='logloss', random_state=config.RANDOM_SEED)
    }
    return models

def build_stacking_classifier():
    """Phase 2: Builds multi-tier Stacking Classifier using Ridge Logistic Regression meta-learner."""
    base_estimators = [
        ('logistic', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=config.RANDOM_SEED)),
        ('svm_rbf', SVC(kernel='rbf', probability=True, random_state=config.RANDOM_SEED)),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=config.RANDOM_SEED))
    ]
    meta_learner = LogisticRegression(penalty='l2', solver='lbfgs', random_state=config.RANDOM_SEED)
    
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=config.N_FOLDS,
        n_jobs=-1
    )
    return stacking_model
