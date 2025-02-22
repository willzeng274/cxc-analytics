import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from runQL.utils.config import MODEL_PARAMS

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


class MarketSegmentation:
    def __init__(self):
        params = MODEL_PARAMS["clustering"]["kmeans"]
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(**params)),
            ]
        )

        self.cluster_centers_ = None
        self.inertia_ = None
        self.cluster_sizes = None
        self.feature_importance = None
        self.feature_names = None

    def preprocess_data(self, X):
        """Preprocess data by handling missing values and outliers"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.copy()
            X.replace([np.inf, -np.inf], np.nan, inplace=True)

            for column in X.columns:
                if X[column].dtype in ["int64", "float64"]:
                    X[column] = X[column].astype("float64")
                    Q1 = X[column].quantile(0.25)
                    Q3 = X[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    X.loc[X[column] < lower_bound, column] = lower_bound
                    X.loc[X[column] > upper_bound, column] = upper_bound

            return X.values
        return X

    def fit(self, X):
        """Fit the K-Means model"""
        X = self.preprocess_data(X)
        self.pipeline.fit(X)

        kmeans = self.pipeline.named_steps["kmeans"]
        scaler = self.pipeline.named_steps["scaler"]

        self.cluster_centers_ = scaler.inverse_transform(kmeans.cluster_centers_)
        self.inertia_ = kmeans.inertia_

        labels = self.pipeline.predict(X)
        self.cluster_sizes = np.bincount(labels)

        self._calculate_feature_importance(X)

    def predict(self, X):
        """Predict cluster labels for new data"""
        X = self.preprocess_data(X)
        return self.pipeline.predict(X)

    def _calculate_feature_importance(self, X):
        """Calculate feature importance based on variance between clusters"""
        X_transformed = self.pipeline.named_steps["scaler"].transform(
            self.pipeline.named_steps["imputer"].transform(X)
        )
        labels = self.pipeline.named_steps["kmeans"].labels_

        global_mean = np.mean(X_transformed, axis=0)

        between_var = np.zeros(X_transformed.shape[1])
        for i in range(self.pipeline.named_steps["kmeans"].n_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                cluster_mean = np.mean(X_transformed[mask], axis=0)
                between_var += np.sum(mask) * (cluster_mean - global_mean) ** 2

        self.feature_importance = between_var / np.sum(between_var)

    def evaluate(self, X):
        """Evaluate clustering quality using multiple metrics"""
        X = self.preprocess_data(X)
        X_transformed = self.pipeline.transform(X)
        labels = self.pipeline.named_steps["kmeans"].labels_

        metrics = {
            "inertia": self.inertia_,
            "silhouette": silhouette_score(X_transformed, labels),
            "calinski_harabasz": calinski_harabasz_score(X_transformed, labels),
            "davies_bouldin": davies_bouldin_score(X_transformed, labels),
        }

        return metrics

    def get_cluster_profiles(self, X, feature_names=None):
        """Get statistical profiles for each cluster"""
        X = self.preprocess_data(X)
        if feature_names is None and self.feature_names is not None:
            feature_names = self.feature_names
        elif feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        labels = self.predict(X)
        X_transformed = self.pipeline.named_steps["scaler"].transform(
            self.pipeline.named_steps["imputer"].transform(X)
        )

        profiles = []
        for i in range(self.pipeline.named_steps["kmeans"].n_clusters):
            mask = labels == i
            cluster_data = X_transformed[mask]

            profile = {
                "cluster_id": i,
                "size": self.cluster_sizes[i],
                "percentage": self.cluster_sizes[i] / len(X) * 100,
                "center": dict(zip(feature_names, self.cluster_centers_[i])),
                "stats": {
                    name: {
                        "mean": np.mean(cluster_data[:, j]),
                        "std": np.std(cluster_data[:, j]),
                        "min": np.min(cluster_data[:, j]),
                        "max": np.max(cluster_data[:, j]),
                    }
                    for j, name in enumerate(feature_names)
                },
            }
            profiles.append(profile)

        return profiles

    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.feature_importance is None:
            raise ValueError("Model hasn't been fitted yet")

        if self.feature_names is not None:
            return dict(zip(self.feature_names, self.feature_importance))
        return self.feature_importance

    def save_model(self, path=None):
        """Save model state"""
        if path is None:
            path = Path(__file__).parent / "saved_models" / "kmeans_model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "pipeline": self.pipeline,
            "cluster_centers": self.cluster_centers_,
            "inertia": self.inertia_,
            "cluster_sizes": self.cluster_sizes,
            "feature_importance": self.feature_importance,
            "feature_names": self.feature_names,
        }
        joblib.dump(state, path)

    def load_model(self, path=None):
        """Load model state"""
        if path is None:
            path = Path(__file__).parent / "saved_models" / "kmeans_model.joblib"

        state = joblib.load(path)
        self.pipeline = state["pipeline"]
        self.cluster_centers_ = state["cluster_centers"]
        self.inertia_ = state["inertia"]
        self.cluster_sizes = state["cluster_sizes"]
        self.feature_importance = state["feature_importance"]
        self.feature_names = state["feature_names"]
