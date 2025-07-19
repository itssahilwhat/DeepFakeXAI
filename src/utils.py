# src/utils.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

# --- Classification Metrics ---
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

# --- Segmentation Metrics ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(y_true, y_pred):
    """ Pixel-wise accuracy for segmentation. """
    return np.mean(y_true == y_pred)

def mae(y_true, y_pred):
    """ Mean Absolute Error for segmentation. """
    return mean_absolute_error(y_true.flatten(), y_pred.flatten())

# --- Logging ---
def log_metrics(metrics_dict, step=None, prefix=None):
    msg = f"Step {step}: " if step is not None else ""
    if prefix:
        msg = f"{prefix} | " + msg
    msg += ", ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
    print(msg)