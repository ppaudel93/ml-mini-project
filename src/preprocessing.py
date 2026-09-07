import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import MiniBatchSparsePCA
import config

class DataPreprocessor:
    def __init__(self, max_features=10000, min_df=3, max_df=0.85):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            stop_words='english'  # Filters out common stop words (the, to, from, and, etc.)
        )

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        X = df['clean_text'].fillna('')
        y = df['is_depression'].values
        return X, y

    def fit_transform_tfidf(self, X_train):
        return self.vectorizer.fit_transform(X_train)

    def transform_tfidf(self, X_test):
        return self.vectorizer.transform(X_test)

    def fit_transform_sparse_pca(self, X_sparse):
        """Reduces high-dimensional TF-IDF matrix (10,000 features) to k=50 subspace."""
        return self.sparse_pca.fit_transform(X_sparse.toarray())
