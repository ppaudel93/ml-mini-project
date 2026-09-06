import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import MiniBatchSparsePCA
import config

class DataPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(**config.TFIDF_PARAMS)
        self.sparse_pca = MiniBatchSparsePCA(
            n_components=config.SPARSE_PCA_COMPONENTS, 
            random_state=config.RANDOM_SEED
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