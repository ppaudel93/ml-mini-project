import os

# Paths
DATA_RAW_PATH = os.path.join("data", "raw", "depression_reddit_cleaned.csv")
DATA_PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("models")

# Processing Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5

# TF-IDF Configuration
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "max_features": 10000,
    "min_df": 3,
    "max_df": 0.85
}

# Sparse PCA Parameters
SPARSE_PCA_COMPONENTS = 50

# Hyperparameter Tuning Grids
XGB_PARAM_GRID = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'subsample': [0.7, 0.8, 1.0]
}