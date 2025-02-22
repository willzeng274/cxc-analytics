from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .config import PREPROCESSING_PARAMS


class DataPreprocessor:
    def __init__(self):
        self.numerical_scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")

    def preprocess_numerical(self, df, columns=None):
        """Preprocess numerical features with scaling and imputation"""
        if columns is None:
            columns = PREPROCESSING_PARAMS["numerical_features"]

        df[columns] = self.numerical_imputer.fit_transform(df[columns])
        df[columns] = self.numerical_scaler.fit_transform(df[columns])

        return df

    def preprocess_categorical(self, df, columns=None):
        """Preprocess categorical features with encoding and imputation"""
        if columns is None:
            columns = PREPROCESSING_PARAMS["categorical_features"]

        df[columns] = self.categorical_imputer.fit_transform(df[columns])

        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])

        return df

    def preprocess_dates(self, df, columns=None):
        """Extract features from date columns"""
        if columns is None:
            columns = PREPROCESSING_PARAMS["date_features"]

        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_quarter"] = df[col].dt.quarter

        return df

    def handle_imbalance(self, X, y, method="smote"):
        """Handle imbalanced datasets"""
        if method == "smote":
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        return X, y

    def create_time_series_features(
        self, df, date_column="date", target_column="amount"
    ):
        """Create features for time series analysis"""
        df = df.sort_values(date_column)

        for lag in [1, 3, 6, 12]:
            df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)

        for window in [3, 6, 12]:
            df[f"{target_column}_rolling_mean_{window}"] = (
                df[target_column].rolling(window=window).mean()
            )

        df[f"{target_column}_yoy_growth"] = df[target_column].pct_change(periods=12)

        return df.dropna()

    def prepare_lstm_sequences(self, data, sequence_length=12):
        """Prepare sequences for LSTM model"""
        sequences = []
        targets = []

        for i in range(len(data) - sequence_length):
            sequences.append(data[i : (i + sequence_length)])
            targets.append(data[i + sequence_length])

        return np.array(sequences), np.array(targets)

    def prepare_data_for_model(self, df, model_type="classification", target=None):
        """Prepare data for specific model type"""
        df = df.copy()

        df = self.preprocess_numerical(df)
        df = self.preprocess_categorical(df)
        df = self.preprocess_dates(df)

        if model_type == "time_series":
            df = self.create_time_series_features(df)
            return df

        elif model_type == "classification":
            if target is None:
                target = PREPROCESSING_PARAMS["target_features"][0]
            X = df.drop(columns=[target])
            y = df[target]
            X, y = self.handle_imbalance(X, y)
            return X, y

        elif model_type == "clustering":
            return df

        return df
