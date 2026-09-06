import joblib
import config
from src.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split

def load_pipeline():
    """Loads the trained model and fits the TF-IDF vectorizer on training data."""
    print("Loading preprocessor and trained XGBoost model...")
    preprocessor = DataPreprocessor()
    X_raw, y = preprocessor.load_data(config.DATA_RAW_PATH)
    
    X_train_raw, _, _, _ = train_test_split(
        X_raw, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    
    # Fit vectorizer on training set
    preprocessor.fit_transform_tfidf(X_train_raw)
    
    # Load model artifact
    model = joblib.load(f"{config.MODEL_DIR}/xgb_best_model.pkl")
    return preprocessor, model

def predict_text(text, preprocessor, model):
    """Predicts depressive vs non-depressive label for raw text input."""
    X_tfidf = preprocessor.transform_tfidf([text])
    probabilities = model.predict_proba(X_tfidf)[0]
    prediction = model.predict(X_tfidf)[0]
    
    label = "Depressive (y=1)" if prediction == 1 else "Non-Depressive (y=0)"
    confidence = probabilities[prediction] * 100
    
    print("\n" + "="*50)
    print(f"Input Text: \"{text}\"")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities -> Non-Depressive: {probabilities[0]:.4f} | Depressive: {probabilities[1]:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    preprocessor, model = load_pipeline()
    
    print("\n--- Interactive Model Inference ---")
    print("Type or paste a post below (or type 'exit' to quit):")
    
    while True:
        user_input = input("\nEnter post text: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue
        predict_text(user_input, preprocessor, model)
