import sys
from pathlib import Path

try:
    import xgboost as xgb
except ImportError:
    # pylint: disable=raise-missing-from
    raise ImportError(
        "XGBoost is required. Please install it with: pip install xgboost"
    )

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from runQL.utils.config import MODEL_PARAMS

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


class StartupSuccessXGBoost:
    def __init__(self):
        params = MODEL_PARAMS["regression"]["gradient_boosting"]
        self.model = xgb.XGBRegressor(**params)
        self.feature_importance = None

    def train(self, X_train, y_train):
        """Train the XGBoost model"""
        self.model.fit(X_train, y_train)
        importance_scores = self.model.feature_importances_
        self.feature_importance = {
            "name": np.array(X_train.columns),
            "importance": importance_scores,
        }

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get probability predictions"""
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """Evaluate the model"""
        y_pred = self.predict(X)
        metrics = {
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
        }
        return metrics

    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.feature_importance is None:
            raise ValueError("Model hasn't been trained yet")
        return self.feature_importance

    def save_model(self, path=None):
        """Save model state"""
        if path is None:
            path = Path(__file__).parent / "saved_models" / "xgboost_model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

        if self.feature_importance is not None:
            importance_path = path.parent / "feature_importance.joblib"
            joblib.dump(self.feature_importance, importance_path)

    def load_model(self, path=None):
        """Load model state"""
        if path is None:
            path = Path(__file__).parent / "saved_models" / "xgboost_model.joblib"
        self.model = joblib.load(path)

        importance_path = path.parent / "feature_importance.joblib"
        if importance_path.exists():
            self.feature_importance = joblib.load(importance_path)

    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        importance = self.get_feature_importance()
        features = importance["name"]
        scores = importance["importance"]

        idx = np.argsort(scores)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(idx)), scores[idx])
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels(features[idx], rotation=45, ha="right")
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance Score")
        ax.set_title("Feature Importance")
        plt.tight_layout()

        return fig
