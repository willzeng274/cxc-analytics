import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

try:
    import shap
except ImportError:
    print("Warning: shap not installed. Some functionality will be limited.")
    shap = None

runql_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if runql_dir not in sys.path:
    sys.path.append(runql_dir)

from runQL.utils.config import MODEL_PARAMS


class StartupSuccessRF:
    def __init__(self):
        params = MODEL_PARAMS["classification"]["random_forest"]
        self.model = RandomForestClassifier(**params)
        self.feature_importance = None
        self.shap_values = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Random Forest model"""
        self.model.fit(X_train, y_train)
        self.feature_importance = {
            "name": X_train.columns,
            "importance": self.model.feature_importances_,
        }

        if shap is not None:
            try:
                explainer = shap.TreeExplainer(self.model)
                self.shap_values = explainer.shap_values(X_train)
            # pylint: disable=broad-exception-caught
            except Exception as e:
                print(f"Warning: Could not calculate SHAP values: {str(e)}")

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print("\nValidation Metrics:")
            for metric, value in val_metrics.items():
                print(f"{metric}: {value:.4f}")

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get probability predictions"""
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """Evaluate the model"""
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "auc": roc_auc_score(y, y_pred_proba),
        }

        return metrics

    def get_feature_importance(self, importance_type="default"):
        """Get feature importance scores"""
        if self.feature_importance is None:
            raise ValueError("Model hasn't been trained yet")

        if importance_type == "shap" and self.shap_values is not None:
            return {
                "name": self.feature_importance["name"],
                "importance": np.abs(self.shap_values).mean(axis=0),
            }
        return self.feature_importance

    def plot_feature_importance(self, importance_type="default", top_n=20):
        """Plot feature importance"""
        import matplotlib.pyplot as plt

        importance = self.get_feature_importance(importance_type)
        features = importance["name"]
        scores = importance["importance"]

        idx = np.argsort(scores)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(idx)), scores[idx])
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels(features[idx], rotation=45, ha="right")
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance Score")
        ax.set_title("Top Feature Importance")
        plt.tight_layout()

        return fig

    def explain_prediction(self, X_instance):
        """Explain a single prediction using SHAP values"""
        if shap is None:
            raise ImportError("shap package is required for prediction explanation")

        if self.shap_values is None:
            raise ValueError("SHAP values haven't been calculated")

        try:
            explainer = shap.TreeExplainer(self.model)
            instance_shap = explainer.shap_values(X_instance)

            return {
                "features": X_instance.columns.tolist(),
                "contributions": (
                    instance_shap[0]
                    if isinstance(instance_shap, list)
                    else instance_shap
                ),
                "base_value": (
                    explainer.expected_value[0]
                    if isinstance(explainer.expected_value, list)
                    else explainer.expected_value
                ),
            }
        except Exception as e:
            # pylint: disable=raise-missing-from
            raise ValueError(f"Could not generate SHAP explanation: {str(e)}")

    def save_model(self, path=None):
        """Save model state"""
        if path is None:
            path = Path(__file__).parent / "saved_models" / "rf_model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model,
            "feature_importance": self.feature_importance,
            "shap_values": self.shap_values,
        }
        joblib.dump(state, path)

    def load_model(self, path=None):
        """Load model state"""
        if path is None:
            path = Path(__file__).parent / "saved_models" / "rf_model.joblib"

        state = joblib.load(path)
        self.model = state["model"]
        self.feature_importance = state["feature_importance"]
        self.shap_values = state["shap_values"]
