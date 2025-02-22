from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cached_models"
DATA_DIR = BASE_DIR / "cached_data"
RESULTS_DIR = BASE_DIR / "model_results"
PLOTS_DIR = BASE_DIR / "model_plots"

MODEL_PARAMS = {
    "classification": {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "regression": {
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": 42,
            "n_jobs": -1,
        }
    },
    "time_series": {
        # arima config is too complex to be included here
        "lstm": {
            "input_size": 1,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 25,
            "sequence_length": 12,
        }
    },
    "clustering": {
        "kmeans": {
            "n_clusters": 5,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "tol": 1e-4,
            "random_state": 42,
        }
    },
}

PREPROCESSING_PARAMS = {
    "features": {
        "numerical": ["age", "total_funding", "deal_count", "avg_deal_size"],
        "categorical": ["ecosystem_id", "primary_tag_id"],
    },
    "time_series": {
        "sequence_length": 12,
        "train_test_split": 0.2,
        "validation_split": 0.1,
    },
    "random_state": 42,
}

EVALUATION_METRICS = {
    "classification": ["accuracy", "precision", "recall", "f1", "auc"],
    "regression": ["mse", "rmse", "mae", "r2"],
    "time_series": ["mape", "rmse", "mae"],
    "clustering": ["silhouette", "calinski_harabasz", "davies_bouldin"],
}
