import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

runql_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if runql_dir not in sys.path:
    sys.path.append(runql_dir)


class InvestmentForecast:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.forecast_history = None
        self.metrics = None

    def find_optimal_params(self, y, max_p=5, max_q=5):
        """Find optimal ARIMA parameters using AIC"""
        best_aic = float("inf")
        best_params = None

        adf_result = adfuller(y)
        d = 0 if adf_result[1] < 0.05 else 1

        max_nlags = min(40, int(len(y) * 0.4))
        acf_values = acf(y, nlags=max_nlags)
        pacf_values = pacf(y, nlags=max_nlags)

        suggested_p = min(len([x for x in pacf_values[1:] if abs(x) > 0.2]), max_p)
        suggested_q = min(len([x for x in acf_values[1:] if abs(x) > 0.2]), max_q)

        for p in range(max(0, suggested_p - 2), min(suggested_p + 2, max_p + 1)):
            for q in range(max(0, suggested_q - 2), min(suggested_q + 2, max_q + 1)):
                try:
                    model = ARIMA(y, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = (p, d, q)
                # pylint: disable=bare-except
                except:
                    continue

        return best_params

    def train(self, y):
        """Train the ARIMA model"""
        best_params = self.find_optimal_params(y)
        self.best_params = {"order": best_params}

        self.model = ARIMA(y, order=best_params)
        self.model = self.model.fit()

        y_pred = self.model.fittedvalues
        self.metrics = self._calculate_metrics(y[1:], y_pred[1:])

        print("\nModel Information:")
        print(f"Order (p,d,q): {best_params}")
        print("\nTraining Metrics:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")

    def predict(self, n_periods, return_conf_int=False, alpha=0.05):
        """Make future predictions"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet")

        forecast = self.model.forecast(steps=n_periods)

        if return_conf_int:
            forecast_obj = self.model.get_forecast(steps=n_periods)
            conf_int = forecast_obj.conf_int(alpha=alpha)
            return forecast, conf_int
        else:
            return forecast

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        }
        return metrics

    def get_diagnostics(self):
        """Get model diagnostics"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet")

        return {
            "aic": self.model.aic,
            "bic": self.model.bic,
            "best_params": self.best_params,
            "training_metrics": self.metrics,
        }

    def save_model(self, path=None):
        """Save model state"""
        if path is None:
            path = Path(__file__).parent / "saved_models" / "arima_model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model,
            "best_params": self.best_params,
            "metrics": self.metrics,
        }
        joblib.dump(state, path)

    def load_model(self, path=None):
        """Load model state"""
        if path is None:
            path = Path(__file__).parent / "saved_models" / "arima_model.joblib"

        state = joblib.load(path)
        self.model = state["model"]
        self.best_params = state["best_params"]
        self.metrics = state["metrics"]

    def plot_diagnostics(self, figsize=(15, 10)):
        """Plot model diagnostics"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet")

        try:
            import matplotlib.pyplot as plt

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

            residuals = pd.Series(self.model.resid)
            residuals.plot(title="Residuals", ax=ax1)
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Residual")

            acf_values = acf(residuals.dropna(), nlags=40)
            ax2.plot(range(len(acf_values)), acf_values)
            ax2.set_title("ACF of Residuals")
            ax2.axhline(y=0, linestyle="--", color="gray")
            ax2.axhline(y=-1.96 / np.sqrt(len(residuals)), linestyle="--", color="gray")
            ax2.axhline(y=1.96 / np.sqrt(len(residuals)), linestyle="--", color="gray")

            import scipy.stats as stats

            stats.probplot(residuals.dropna(), dist="norm", plot=ax3)
            ax3.set_title("Q-Q Plot")

            residuals.hist(ax=ax4, density=True, bins=30)
            ax4.set_title("Residual Distribution")

            xmin, xmax = ax4.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(
                x, np.mean(residuals.dropna()), np.std(residuals.dropna())
            )
            ax4.plot(x, p, "k", linewidth=2)

            plt.tight_layout()
            return fig

        except ImportError:
            print("matplotlib is required for plotting diagnostics")
