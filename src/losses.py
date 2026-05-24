import numpy as np


def mse_loss(y_pred, y_true):
    y_true = y_true.reshape(-1, 1)

    loss = np.mean((y_pred - y_true) ** 2)

    grad = 2 * (y_pred - y_true) / len(y_true)

    return loss, grad


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)

    exp_values = np.exp(logits)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    return probabilities


def cross_entropy_loss(logits, y_true):
    probabilities = softmax(logits)

    n = len(y_true)

    correct_probabilities = probabilities[range(n), y_true]
    loss = -np.mean(np.log(correct_probabilities + 1e-12))

    grad = probabilities.copy()
    grad[range(n), y_true] -= 1
    grad = grad / n

    return loss, grad


