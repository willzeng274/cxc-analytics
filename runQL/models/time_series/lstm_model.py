import numpy as np
from pathlib import Path
import sys

runql_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if runql_dir not in sys.path:
    sys.path.append(runql_dir)

from runQL.utils.config import MODEL_PARAMS

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    class LSTMPredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.batch_norm = nn.BatchNorm1d(input_size)

            self.lstm_layers = nn.ModuleList(
                [
                    nn.LSTM(
                        input_size if i == 0 else hidden_size,
                        hidden_size,
                        num_layers=1,
                        dropout=0 if i == num_layers - 1 else dropout,
                        batch_first=True,
                    )
                    for i in range(num_layers)
                ]
            )

            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
                nn.Softmax(dim=1),
            )

            self.fc_layers = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x):
            x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)

            lstm_out = x
            for lstm_layer in self.lstm_layers:
                h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
                new_out, _ = lstm_layer(lstm_out, (h0, c0))
                lstm_out = (
                    new_out + lstm_out if lstm_out.size() == new_out.size() else new_out
                )

            attention_weights = self.attention(lstm_out)
            lstm_out = torch.sum(attention_weights * lstm_out, dim=1)

            out = self.fc_layers(lstm_out)
            return out

    class InvestmentLSTM:
        def __init__(self, input_size=1):
            params = MODEL_PARAMS["time_series"]["lstm"]
            self.device = DEVICE
            self.model = LSTMPredictor(
                input_size=input_size,
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
            ).to(self.device)

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=params["learning_rate"], weight_decay=0.01
            )

            self.mse_criterion = nn.MSELoss()
            self.mae_criterion = nn.L1Loss()

            self.batch_size = params["batch_size"]
            self.epochs = params["num_epochs"]
            self.history = {"train_loss": [], "val_loss": []}

        def prepare_data(self, X, y):
            """Convert numpy arrays to PyTorch tensors and create DataLoader"""

            X_mean = np.mean(X)
            X_std = np.std(X)
            y_mean = np.mean(y)
            y_std = np.std(y)

            X_scaled = (X - X_mean) / (X_std + 1e-8)
            y_scaled = (y - y_mean) / (y_std + 1e-8)

            X = torch.FloatTensor(X_scaled).to(self.device)
            y = torch.FloatTensor(y_scaled).to(self.device)
            dataset = TensorDataset(X, y)

            self.scale_params = {
                "X_mean": X_mean,
                "X_std": X_std,
                "y_mean": y_mean,
                "y_std": y_std,
            }

            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        def train(self, X_train, y_train, X_val=None, y_val=None):
            """Train the LSTM model"""
            train_loader = self.prepare_data(X_train, y_train)
            val_loader = self.prepare_data(X_val, y_val) if X_val is not None else None
            best_val_loss = float("inf")
            patience = 5
            patience_counter = 0

            for epoch in range(self.epochs):
                self.model.train()
                epoch_losses = []

                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()

                    outputs = self.model(batch_X)
                    loss = self.mse_criterion(outputs, batch_y.unsqueeze(1))

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.optimizer.step()

                    epoch_losses.append(loss.item())

                avg_loss = np.mean(epoch_losses)
                self.history["train_loss"].append(avg_loss)

                if val_loader is not None:
                    val_loss = self.evaluate(X_val, y_val)
                    self.history["val_loss"].append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.save_model()
                    else:
                        patience_counter += 1

                    print(
                        f"Epoch [{epoch+1}/{self.epochs}], "
                        f"Loss: {avg_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}"
                    )

                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
                else:
                    print(f"Epoch [{epoch+1}/{self.epochs}], " f"Loss: {avg_loss:.6f}")

                if epoch > 0 and epoch % 10 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] *= 0.9

        def plot_training_history(self):
            """Plot training and validation loss history"""
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(self.history["train_loss"]) + 1)

            ax.plot(epochs, self.history["train_loss"], "b-", label="Training Loss")
            if self.history["val_loss"]:
                ax.plot(epochs, self.history["val_loss"], "r-", label="Validation Loss")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("LSTM Training History")
            ax.grid(True)
            ax.legend()

            return fig

        def plot_predictions(self, X, y_true, n_samples=None):
            """Plot model predictions against true values"""
            import matplotlib.pyplot as plt

            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                predictions = self.model(X_tensor).cpu().numpy()

            if n_samples is not None:
                indices = np.random.choice(
                    len(X), min(n_samples, len(X)), replace=False
                )
                predictions = predictions[indices]
                y_true = y_true[indices]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            ax1.plot(y_true, "b-", label="True Values", alpha=0.7)
            ax1.plot(predictions, "r--", label="Predictions", alpha=0.7)
            ax1.set_title("Time Series Prediction")
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("Value")
            ax1.grid(True)
            ax1.legend()

            ax2.scatter(y_true, predictions, alpha=0.5)
            ax2.plot(
                [y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                "r--",
                alpha=0.5,
            )
            ax2.set_title("Prediction vs True Value")
            ax2.set_xlabel("True Values")
            ax2.set_ylabel("Predictions")
            ax2.grid(True)

            plt.tight_layout()
            return fig

        def predict(self, X):
            """Make predictions"""
            self.model.eval()
            X_scaled = (X - self.scale_params["X_mean"]) / (
                self.scale_params["X_std"] + 1e-8
            )
            X = torch.FloatTensor(X_scaled).to(self.device)

            with torch.no_grad():
                predictions = self.model(X)
                predictions = (
                    predictions.cpu().numpy() * self.scale_params["y_std"]
                    + self.scale_params["y_mean"]
                )

            return predictions

        def evaluate(self, X, y):
            """Evaluate the model"""
            self.model.eval()
            X = torch.FloatTensor(
                (X - self.scale_params["X_mean"]) / (self.scale_params["X_std"] + 1e-8)
            ).to(self.device)
            y = torch.FloatTensor(
                (y - self.scale_params["y_mean"]) / (self.scale_params["y_std"] + 1e-8)
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(X)
                loss = self.mse_criterion(outputs, y.unsqueeze(1))
            return loss.item()

        def save_model(self, path=None):
            """Save model state"""
            if path is None:
                path = Path(__file__).parent / "saved_models" / "lstm_model.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.cpu()
            torch.save(self.model.state_dict(), path)
            self.model.to(self.device)

        def load_model(self, path=None):
            """Load model state"""
            if path is None:
                path = Path(__file__).parent / "saved_models" / "lstm_model.pth"
            state_dict = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

except ImportError:

    class LSTMPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is not available. Please install PyTorch or use ARIMA/Prophet models instead."
            )

    class InvestmentLSTM:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is not available. Please install PyTorch or use ARIMA/Prophet models instead."
            )
