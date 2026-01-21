"""
Evaluation utilities for model performance metrics.
Contains functions for calculating R², MSE, RMSE, MAE, and other regression metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary containing MSE, RMSE, MAE, and R² score
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def print_metrics(metrics, prefix=""):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: String to prefix before each metric name
    """
    print(f"\n{prefix}Metrics:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
