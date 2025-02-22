from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from runQL.runql import perform_full_eda
from runQL.models import (
    StartupSuccessRF,
    StartupSuccessXGBoost,
    InvestmentForecast,
    InvestmentProphet,
    MarketSegmentation,
    InvestmentLSTM,
)


def prepare_data(analysis):
    """Prepare data for different models"""
    companies_df = analysis["raw_data"]["companies"]
    deals_df = analysis["raw_data"]["deals"]

    features = pd.DataFrame(
        {
            "age": companies_df["age"],
            "ecosystem_id": pd.factorize(companies_df["ecosystemName"])[0],
            "primary_tag_id": pd.factorize(companies_df["primaryTag"])[0],
        }
    )
    features.index = companies_df.index

    funding_data = (
        deals_df.groupby("companyId")["amount"].agg(["sum", "count", "mean"]).fillna(0)
    )
    funding_data.columns = ["total_funding", "deal_count", "avg_deal_size"]

    time_series = deals_df.set_index("date")["amount"].resample("M").sum().fillna(0)

    sequence_length = 12
    lstm_data = []
    lstm_targets = []
    values = time_series.values.reshape(-1, 1)
    for i in range(len(values) - sequence_length):
        lstm_data.append(values[i : (i + sequence_length)])
        lstm_targets.append(values[i + sequence_length])
    lstm_data = np.array(lstm_data).reshape((-1, sequence_length, 1))
    lstm_targets = np.array(lstm_targets).reshape((-1, 1))

    monthly_returns = (
        deals_df.pivot_table(
            index=pd.to_datetime(deals_df["date"]).dt.to_period("M"),
            columns="companyId",
            values="amount",
            aggfunc="sum",
        )
        .fillna(0)
        .pct_change()
        .fillna(0)
    )

    return {
        "features": features,
        "funding_data": funding_data,
        "time_series": time_series,
        "lstm_data": (lstm_data, lstm_targets),
        "returns": monthly_returns,
    }


def save_plot(fig, filename, plots_dir):
    """Save a matplotlib figure to file"""
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / filename)
    plt.close(fig)


def train_and_save_models():
    """Train all models and save them along with the analysis data"""
    print("Starting data analysis and model training...")

    CACHE_DIR = Path(__file__).parent / "cached_models"
    DATA_DIR = Path(__file__).parent / "cached_data"
    PLOTS_DIR = Path(__file__).parent / "model_plots"

    for directory in [CACHE_DIR, DATA_DIR, PLOTS_DIR]:
        directory.mkdir(exist_ok=True)

    try:
        print("Performing exploratory data analysis...")
        analysis = perform_full_eda()
        joblib.dump(analysis, DATA_DIR / "analysis_results.joblib")

        print("Preparing data for models...")
        data = prepare_data(analysis)

        print("Training Random Forest model...")
        rf_model = StartupSuccessRF()
        success_target = (
            data["funding_data"]["total_funding"]
            > data["funding_data"]["total_funding"].median()
        )
        rf_model.train(data["features"], success_target)
        rf_model.save_model(CACHE_DIR / "random_forest.joblib")

        fig = rf_model.plot_feature_importance()
        save_plot(fig, "rf_feature_importance.png", PLOTS_DIR)

        print("Training XGBoost model...")
        xgb_model = StartupSuccessXGBoost()
        funding_target = (
            data["funding_data"]["total_funding"]
            .reindex(data["features"].index)
            .fillna(0)
            .astype(float)
        )
        xgb_model.train(data["features"], funding_target)
        xgb_model.save_model(CACHE_DIR / "xgboost.joblib")

        fig = xgb_model.plot_feature_importance()
        save_plot(fig, "xgb_feature_importance.png", PLOTS_DIR)

        print("Training LSTM model...")
        lstm_model = InvestmentLSTM()
        lstm_X, lstm_y = data["lstm_data"]
        lstm_model.train(X_train=lstm_X, y_train=lstm_y)
        lstm_model.save_model(CACHE_DIR / "lstm.pth")

        fig = lstm_model.plot_training_history()
        save_plot(fig, "lstm_training_history.png", PLOTS_DIR)
        fig = lstm_model.plot_predictions(lstm_X, lstm_y, n_samples=100)
        save_plot(fig, "lstm_predictions.png", PLOTS_DIR)

        print("Training ARIMA model...")
        arima_model = InvestmentForecast()
        arima_model.train(data["time_series"])
        arima_model.save_model(CACHE_DIR / "arima.joblib")

        fig = arima_model.plot_diagnostics()
        save_plot(fig, "arima_diagnostics.png", PLOTS_DIR)

        print("Training Prophet model...")
        prophet_model = InvestmentProphet()
        prophet_model.train(data["time_series"].index, data["time_series"].values)
        prophet_model.predict()
        prophet_model.save_model(CACHE_DIR / "prophet.joblib")

        fig = prophet_model.plot_forecast()
        save_plot(fig, "prophet_forecast.png", PLOTS_DIR)
        fig = prophet_model.plot_components()
        save_plot(fig, "prophet_components.png", PLOTS_DIR)

        print("Training Market Segmentation...")
        segment_model = MarketSegmentation()
        segment_model.fit(data["features"])
        segment_model.save_model(CACHE_DIR / "segmentation.joblib")

        print("All models trained and saved successfully!")
        print(f"Model visualizations saved in {PLOTS_DIR}")
        return True
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False


if __name__ == "__main__":
    train_and_save_models()
