from sklearn.model_selection import train_test_split
from src.preprocessing import DataPreprocessor
from src.baseline_models import get_baseline_models
from src.ensemble_models import get_advanced_models, build_stacking_classifier
from src.evaluation import ModelEvaluator
from src.interpretability import InterpretabilityEngine
import config
import joblib

def main():
    print("--- 1. Loading & Preprocessing Data ---")
    preprocessor = DataPreprocessor()
    X_raw, y = preprocessor.load_data(config.DATA_RAW_PATH)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    
    X_train_tfidf = preprocessor.fit_transform_tfidf(X_train_raw)
    X_test_tfidf = preprocessor.transform_tfidf(X_test_raw)
    
    evaluator = ModelEvaluator()
    results = {}

    print("\n--- 2. Evaluating Baseline & Advanced Models ---")
    all_models = {**get_baseline_models(), **get_advanced_models()}
    
    for name, model in all_models.items():
        # Handle dense matrix requirement for LDA
        X_input = X_train_tfidf.toarray() if name == "LDA" else X_train_tfidf
        print(f"Evaluating {name}...")
        metrics = evaluator.evaluate_cv(model, X_input, y_train)
        results[name] = metrics

    print("\n--- 3. Training & Evaluating Stacking Ensemble ---")
    stacking_clf = build_stacking_classifier()
    stacking_clf.fit(X_train_tfidf, y_train)
    results["Stacking Classifier"] = evaluator.evaluate_cv(stacking_clf, X_train_tfidf, y_train)

    print("\n--- Benchmark Results Summary ---")
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print("\n--- 4. SHAP Interpretability Analysis ---")
    best_xgb = all_models["XGBoost Baseline"]
    best_xgb.fit(X_train_tfidf, y_train)
    
    feature_names = preprocessor.vectorizer.get_feature_names_out()
    interpreter = InterpretabilityEngine(best_xgb, feature_names)
    
    # Run global interpretability on a 100-sample test batch
    interpreter.generate_shap_summary(X_test_tfidf[:100].toarray())
    print("SHAP analysis exported successfully.")

    # Save artifact
    joblib.dump(best_xgb, f"{config.MODEL_DIR}/xgb_best_model.pkl")

if __name__ == "__main__":
    main()
