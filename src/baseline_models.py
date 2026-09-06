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
            penalty='l2', solver='lbfgs', random_state=config.RANDOM_SEED, max_iter=1000
        ),
        "Logistic Regression (L1 Lasso)": LogisticRegression(
            penalty='l1', solver='saga', random_state=config.RANDOM_SEED, max_iter=1000
        )
    }
    return models
