import numpy as np
from sklearn.metrics import roc_curve

def calibrate_threshold_youden(y_true, y_prob):
    """
    Calibrate threshold using Youden's Index on validation set.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold