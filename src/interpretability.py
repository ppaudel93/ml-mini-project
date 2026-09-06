import shap
import matplotlib.pyplot as plt

class InterpretabilityEngine:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def generate_shap_summary(self, X_sample):
        """Phase 3: Global SHAP Interpretability Analysis using TreeSHAP."""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.title("Global Feature Importance (TreeSHAP)")
        plt.tight_layout()
        plt.savefig("models/shap_summary.png")
        plt.close()

    def generate_local_force_plot(self, instance, instance_idx):
        """Phase 3: Local SHAP force plots for specific test posts."""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(instance)
        
        shap.plots.force(shap_values[0], show=False, matplotlib=True)
        plt.savefig(f"models/shap_force_post_{instance_idx}.png")
        plt.close()
