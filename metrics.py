"""
Evaluation metrics for reservoir computing.
"""

import numpy as np


def nmse(y_true, y_pred):
    """
    Normalised Mean Square Error.

    NMSE = mean((y_true - y_pred)^2) / var(y_true).
    Lower is better; NMSE < 1 beats the mean predictor.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true)
    if var < 1e-30:
        return float("inf")
    return mse / var


def mase(y_true, y_pred):
    """
    Mean Absolute Scaled Error.

    Scaled by the mean absolute difference of the naive (persistence) forecast.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    naive_mae = np.mean(np.abs(np.diff(y_true)))
    if naive_mae < 1e-30:
        return float("inf")
    return mae / naive_mae


def correlation(y_true, y_pred):
    """Pearson correlation coefficient between true and predicted."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cc = np.corrcoef(y_true, y_pred)[0, 1]
    return cc
