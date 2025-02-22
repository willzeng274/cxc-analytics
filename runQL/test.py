import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from runQL.runql import perform_full_eda
from runQL.models.classification.random_forest_model import StartupSuccessRF
from runQL.models.classification.xgboost_model import StartupSuccessXGBoost
from runQL.models.time_series.arima_model import InvestmentForecast
from runQL.models.time_series.prophet_model import InvestmentProphet
from runQL.models.clustering.kmeans_model import MarketSegmentation
from runQL.train import prepare_data


def convert_to_serializable(obj):
    """Convert numpy and pandas objects to JSON serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        if isinstance(obj.index, pd.DatetimeIndex):
            return dict(
                zip(
                    obj.index.strftime("%Y-%m-%d").tolist(),
                    [convert_to_serializable(v) for v in obj.values],
                )
            )
        return {str(k): convert_to_serializable(v) for k, v in obj.to_dict().items()}
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def test_and_save_predictions():
    """Run all models and save their predictions and metrics"""
    print("Starting model testing and prediction generation...")

    CACHE_DIR = Path(__file__).parent / "cached_models"
    DATA_DIR = Path(__file__).parent / "cached_data"
    RESULTS_DIR = Path(__file__).parent / "model_results"

    for directory in [CACHE_DIR, DATA_DIR, RESULTS_DIR]:
        directory.mkdir(exist_ok=True)

    # Load analysis data and prepare it the same way as in training
    print("Loading and preparing data...")
    analysis = perform_full_eda()
    data = prepare_data(analysis)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_predictions": {},
        "model_metrics": {},
    }

    try:
        if (CACHE_DIR / "random_forest.joblib").exists():
            print("Testing Random Forest model...")
            rf_model = StartupSuccessRF()
            rf_model.load_model(CACHE_DIR / "random_forest.joblib")
            success_target = (
                data["funding_data"]["total_funding"]
                > data["funding_data"]["total_funding"].median()
            )
            predictions = rf_model.predict(data["features"])
            feature_importance = rf_model.get_feature_importance()
            results["model_predictions"]["random_forest"] = {
                "feature_importance": {
                    "names": feature_importance["name"].tolist(),
                    "scores": feature_importance["importance"].tolist(),
                },
                "accuracy": float(np.mean(predictions == success_target)),
            }

        if (CACHE_DIR / "xgboost.joblib").exists():
            print("Testing XGBoost model...")
            xgb_model = StartupSuccessXGBoost()
            xgb_model.load_model(CACHE_DIR / "xgboost.joblib")
            funding_target = (
                data["funding_data"]["total_funding"]
                .reindex(data["features"].index)
                .fillna(0)
            )
            predictions = xgb_model.predict(data["features"])
            feature_importance = xgb_model.get_feature_importance()
            results["model_predictions"]["xgboost"] = {
                "feature_importance": {
                    "names": feature_importance["name"].tolist(),
                    "scores": feature_importance["importance"].tolist(),
                },
                "rmse": float(np.sqrt(np.mean((predictions - funding_target) ** 2))),
            }

        if (CACHE_DIR / "arima.joblib").exists():
            print("Testing ARIMA model...")
            arima_model = InvestmentForecast()
            arima_model.load_model(CACHE_DIR / "arima.joblib")
            forecast = arima_model.predict(12)
            results["model_predictions"]["arima"] = {
                "forecast": convert_to_serializable(forecast),
                "dates": pd.date_range(start=datetime.now(), periods=12, freq="M")
                .strftime("%Y-%m-%d")
                .tolist(),
            }

        if (CACHE_DIR / "prophet.joblib").exists():
            print("Testing Prophet model...")
            prophet_model = InvestmentProphet()
            prophet_model.load_model(CACHE_DIR / "prophet.joblib")
            forecast = prophet_model.predict(periods=12)
            results["model_predictions"]["prophet"] = {
                "forecast": forecast["yhat"].tolist(),
                "dates": forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "lower_bound": forecast["yhat_lower"].tolist(),
                "upper_bound": forecast["yhat_upper"].tolist(),
            }

        if (CACHE_DIR / "segmentation.joblib").exists():
            print("Testing Market Segmentation...")
            segment_model = MarketSegmentation()
            segment_model.load_model(CACHE_DIR / "segmentation.joblib")
            predictions = segment_model.predict(data["features"])
            feature_importance = segment_model.get_feature_importance()
            results["model_predictions"]["segmentation"] = {
                "feature_importance": convert_to_serializable(feature_importance),
                "cluster_distribution": convert_to_serializable(
                    pd.Series(predictions).value_counts()
                ),
            }

        results = convert_to_serializable(results)

        results_file = (
            RESULTS_DIR / f'model_results_{datetime.now().strftime("%Y%m%d")}.json'
        )
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {results_file}")
        return True
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False


if __name__ == "__main__":
    test_and_save_predictions()
