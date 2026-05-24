import numpy as np


def accuracy(logits, y_true):
    predicted_classes = np.argmax(logits, axis=1)
    return np.mean(predicted_classes == y_true)


def mean_absolute_error(y_pred, y_true):
    y_true = y_true.reshape(-1, 1)
    return np.mean(np.abs(y_pred - y_true))


def root_mean_squared_error(y_pred, y_true):
    y_true = y_true.reshape(-1, 1)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
