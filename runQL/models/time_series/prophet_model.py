try:
    from prophet import Prophet
except ImportError:
    # pylint: disable=raise-missing-from
    raise ImportError(
        "Prophet is required. Please install it with: pip install prophet"
    )

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

runql_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if runql_dir not in sys.path:
    sys.path.append(runql_dir)


class InvestmentProphet:
    def __init__(self, seasonality_mode="multiplicative", changepoint_prior_scale=0.05):
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        self.metrics = None
        self.forecast = None

    def prepare_data(self, dates, values):
        """Prepare data in Prophet format"""
        return pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})

    def train(self, dates, values, regressors=None):
        """Train the Prophet model"""
        df = self.prepare_data(dates, values)

        if regressors is not None:
            for name, values in regressors.items():
                df[name] = values
                self.model.add_regressor(name)

        self.model.fit(df)

        forecast = self.model.predict(df)
        y_pred = forecast["yhat"].values

        self.metrics = {
            "mse": mean_squared_error(values, y_pred),
            "rmse": np.sqrt(mean_squared_error(values, y_pred)),
            "mae": mean_absolute_error(values, y_pred),
            "r2": r2_score(values, y_pred),
        }

        print("\nTraining Metrics:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")

    def predict(self, periods=30, freq="D", future_regressors=None):
        """Make future predictions"""
        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        if future_regressors is not None:
            for name, values in future_regressors.items():
                future[name] = values

        self.forecast = self.model.predict(future)
        return self.forecast

    def get_metrics(self):
        """Get model performance metrics"""
        if self.metrics is None:
            raise ValueError("Model hasn't been trained yet")
        return self.metrics

    def save_model(self, path):
        """Save model state"""
        state = {
            "model": self.model,
            "metrics": self.metrics,
            "forecast": self.forecast,
        }
        joblib.dump(state, path)

    def load_model(self, path):
        """Load model state"""
        state = joblib.load(path)
        self.model = state["model"]
        self.metrics = state["metrics"]
        self.forecast = state["forecast"]

    def plot_components(self, figsize=(15, 10)):
        """Plot forecast components"""
        if self.forecast is None:
            raise ValueError("No forecast available. Run predict() first.")
        return self.model.plot_components(self.forecast, figsize=figsize)

    def plot_forecast(self, figsize=(15, 6)):
        """Plot forecast with uncertainty intervals"""
        if self.forecast is None:
            raise ValueError("No forecast available. Run predict() first.")
        return self.model.plot(self.forecast, figsize=figsize)

    def get_forecast_components(self):
        """Get decomposed forecast components"""
        if self.forecast is None:
            raise ValueError("No forecast available. Run predict() first.")

        components = {
            "trend": self.forecast["trend"],
            "yhat": self.forecast["yhat"],
            "yhat_lower": self.forecast["yhat_lower"],
            "yhat_upper": self.forecast["yhat_upper"],
        }

        seasonal_components = [
            col for col in self.forecast.columns if "seasonal" in col
        ]
        for component in seasonal_components:
            components[component] = self.forecast[component]

        return pd.DataFrame(components, index=self.forecast["ds"])

    def get_changepoints(self):
        """Get detected changepoints"""
        return pd.DataFrame(
            {
                "date": self.model.changepoints,
                "magnitude": self.model.params["delta"].mean(0),
            }
        )
