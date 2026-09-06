from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import config

def get_advanced_models():
    """Returns kernelized, tree ensemble, and gradient boosting models."""
    models = {
        "SVC (Linear)": CalibratedClassifierCV(
            SVC(kernel='linear', random_state=config.RANDOM_SEED), 
            ensemble=False
        ),
        "SVC (RBF)": CalibratedClassifierCV(
            SVC(kernel='rbf', random_state=config.RANDOM_SEED), 
            ensemble=False
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, 
            random_state=config.RANDOM_SEED, 
            n_jobs=-1
        ),
        "XGBoost Baseline": XGBClassifier(
            eval_metric='logloss', 
            verbosity=0, 
            random_state=config.RANDOM_SEED
        )
    }
    return models

def build_stacking_classifier():
    """Phase 2: Builds multi-tier Stacking Classifier using Ridge Logistic Regression meta-learner."""
    base_estimators = [
        (
            'logistic', 
            LogisticRegression(solver='lbfgs', l1_ratio=0, max_iter=2000, random_state=config.RANDOM_SEED)
        ),
        (
            'svm_rbf', 
            CalibratedClassifierCV(SVC(kernel='rbf', random_state=config.RANDOM_SEED), ensemble=False)
        ),
        (
            'xgb', 
            XGBClassifier(eval_metric='logloss', verbosity=0, random_state=config.RANDOM_SEED)
        )
    ]
    meta_learner = LogisticRegression(solver='lbfgs', l1_ratio=0, random_state=config.RANDOM_SEED)
    
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=config.N_FOLDS,
        n_jobs=1
    )
    return stacking_model
