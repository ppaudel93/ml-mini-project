from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import config

def get_baseline_models():
    """Returns baseline linear and discriminant classifiers."""
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "Logistic Regression (L2 Ridge)": LogisticRegression(
            solver='lbfgs', 
            l1_ratio=0,
            random_state=config.RANDOM_SEED, 
            max_iter=2000
        ),
        "Logistic Regression (L1 Lasso)": LogisticRegression(
            solver='saga', 
            l1_ratio=1,
            random_state=config.RANDOM_SEED, 
            max_iter=2000,
            tol=1e-3
        )
    }
    return models
