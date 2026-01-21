"""
Ensemble model combining LSTM, Linear Regression, and ARIMA for robust predictions.
Weighted voting reduces overfitting on small datasets.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """Ensemble of LSTM + Linear + ARIMA with learned weights."""
    
    def __init__(self, lstm_model=None, weights=None):
        """
        Args:
            lstm_model: Trained PyTorch LSTM model
            weights: [lstm_weight, linear_weight, arima_weight] - sum to 1.0
                    Default: equal weights [0.33, 0.33, 0.34]
        """
        self.lstm_model = lstm_model
        self.linear_model = None
        self.arima_model = None
        self.scaler = None
        
        if weights is None:
            self.weights = np.array([0.4, 0.3, 0.3])  # LSTM dominant
        else:
            weights = np.array(weights)
            self.weights = weights / weights.sum()  # Normalize
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train linear and ARIMA components (LSTM already trained).
        
        Args:
            X_train: (n_samples, seq_len, n_features) array
            y_train: (n_samples,) target values
            X_val: Validation data (optional, for weight optimization)
            y_val: Validation targets (optional)
        """
        print("Fitting ensemble components...")
        
        # 1. Extract features for linear model (last timestep + simple statistics)
        if len(X_train.shape) == 3:  # Sequence data
            X_linear = np.hstack([
                X_train[:, -1, :],  # Last timestep features
                X_train[:, :, 0].mean(axis=1, keepdims=True),  # Mean over time
                X_train[:, :, 0].std(axis=1, keepdims=True)    # Std over time
            ])
        else:
            X_linear = X_train
        
        # 2. Train Linear Regression
        self.linear_model = Ridge(alpha=1.0)
        self.linear_model.fit(X_linear, y_train)
        print(f"  Linear model R²: {self.linear_model.score(X_linear, y_train):.3f}")
        
        # 3. Train ARIMA on univariate target (if enough data)
        try:
            if len(y_train) >= 50:
                self.arima_model = ARIMA(y_train, order=(1, 0, 1)).fit()
                print(f"  ARIMA model fitted (AIC: {self.arima_model.aic:.1f})")
            else:
                print("  Skipping ARIMA (insufficient data)")
        except Exception as e:
            print(f"  ARIMA fit failed: {e}. Will use linear only.")
        
        # 4. Optimize weights on validation set if provided
        if X_val is not None and y_val is not None:
            self._optimize_weights(X_train, y_train, X_val, y_val)
    
    def _optimize_weights(self, X_train, y_train, X_val, y_val):
        """Optimize ensemble weights on validation set."""
        from scipy.optimize import minimize
        
        def validation_error(w):
            # Constrain weights to be positive and sum to 1
            w_normalized = np.abs(w) / np.abs(w).sum()
            self.weights = w_normalized
            preds = self.predict(X_val)
            mse = np.mean((preds - y_val) ** 2)
            return mse
        
        # Use simplex method which respects bounds naturally
        result = minimize(
            validation_error,
            self.weights,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
        )
        
        self.weights = np.abs(result.x) / np.abs(result.x).sum()
        print(f"  Optimized weights: LSTM={self.weights[0]:.2f}, Linear={self.weights[1]:.2f}, ARIMA={self.weights[2]:.2f}")
    
    def predict(self, X):
        """
        Ensemble prediction combining all models.
        
        Args:
            X: (n_samples, seq_len, n_features) or (n_samples, features)
        
        Returns:
            (n_samples,) predictions
        """
        n_samples = len(X) if isinstance(X, np.ndarray) else X.shape[0]
        predictions = np.zeros((n_samples, 3))  # [lstm, linear, arima]
        
        # 1. LSTM predictions
        if self.lstm_model is not None:
            self.lstm_model.eval()
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X_tensor = torch.FloatTensor(X)
                else:
                    X_tensor = X
                lstm_preds = self.lstm_model(X_tensor).cpu().numpy().flatten()
            predictions[:, 0] = lstm_preds
        
        # 2. Linear predictions
        if self.linear_model is not None:
            if len(X.shape) == 3:
                X_linear = np.hstack([
                    X[:, -1, :],
                    X[:, :, 0].mean(axis=1, keepdims=True),
                    X[:, :, 0].std(axis=1, keepdims=True)
                ])
            else:
                X_linear = X
            predictions[:, 1] = self.linear_model.predict(X_linear)
        
        # 3. ARIMA predictions (use last value as naive forecast)
        if self.arima_model is not None:
            # ARIMA expects single series, so use simple walk-forward
            for i in range(n_samples):
                try:
                    # For each sample, assume it's a continuation of training series
                    pred = self.arima_model.get_forecast(steps=1).predicted_mean.values[0]
                    predictions[i, 2] = pred
                except:
                    predictions[i, 2] = predictions[i, 0]  # Fallback to LSTM
        else:
            predictions[:, 2] = predictions[:, 0]  # Fallback to LSTM
        
        # Weighted ensemble
        ensemble_pred = predictions @ self.weights
        return ensemble_pred
    
    def predict_with_confidence(self, X):
        """Return predictions + uncertainty bounds."""
        predictions = np.zeros((len(X), 3))
        
        # Get all model predictions separately
        self.lstm_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            lstm_preds = self.lstm_model(X_tensor).cpu().numpy().flatten()
        
        if len(X.shape) == 3:
            X_linear = np.hstack([
                X[:, -1, :],
                X[:, :, 0].mean(axis=1, keepdims=True),
                X[:, :, 0].std(axis=1, keepdims=True)
            ])
        else:
            X_linear = X
        linear_preds = self.linear_model.predict(X_linear)
        
        predictions[:, 0] = lstm_preds
        predictions[:, 1] = linear_preds
        predictions[:, 2] = lstm_preds  # Use LSTM as third baseline
        
        # Ensemble mean
        mean = predictions @ self.weights
        
        # Uncertainty = disagreement between models
        std = predictions.std(axis=1)
        
        return mean, std

if __name__ == "__main__":
    # Test ensemble
    X_dummy = np.random.randn(100, 60, 12)
    y_dummy = np.random.randn(100)
    
    ensemble = EnsemblePredictor(weights=[0.4, 0.3, 0.3])
    ensemble.fit(X_dummy, y_dummy)
    preds = ensemble.predict(X_dummy[:10])
    print(f"\nTest predictions shape: {preds.shape}")
    print(f"Sample predictions: {preds[:3]}")
