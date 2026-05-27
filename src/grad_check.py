import time
import numpy as np

from data import load_data
from mlp import MLP
from losses import cross_entropy_loss


def get_parameters(model):
    return {
        "W1": model.W1,
        "b1": model.b1,
        "W2": model.W2,
        "b2": model.b2,
        "W3": model.W3,
        "b3": model.b3,
    }


def get_backprop_gradients(model):
    return {
        "W1": model.dW1,
        "b1": model.db1,
        "W2": model.dW2,
        "b2": model.db2,
        "W3": model.dW3,
        "b3": model.db3,
    }


def numerical_gradients(model, X, y, h):
    parameters = get_parameters(model)
    gradients = {}

    for name in parameters:
        parameter = parameters[name]
        gradient = np.zeros_like(parameter)

        # for each param in each layer we calculate numeric gradient
        # so for each param we have to flactuate param and then calculate loss (by forword pass)
        for index in np.ndindex(parameter.shape):
            old_value = parameter[index]

            parameter[index] = old_value + h
            loss_plus, _ = cross_entropy_loss(model.forward(X), y)

            parameter[index] = old_value
            loss_original, _ = cross_entropy_loss(model.forward(X), y)

            gradient[index] = (loss_plus - loss_original) / h

            parameter[index] = old_value

        gradients[name] = gradient

    return gradients


def relative_difference(grad_1, grad_2):
    top = np.linalg.norm(grad_1 - grad_2)
    bottom = np.linalg.norm(grad_1) + np.linalg.norm(grad_2)

    return top / (bottom + 1e-12)


def compare_gradients(h):
    X_train, X_test, y_train, y_test = load_data(task="classification")

    X_batch = X_train[:5]
    y_batch = y_train[:5]

    model = MLP(
        input_size=X_batch.shape[1],
        hidden_size_1=8,
        hidden_size_2=4,
        output_size=3
    )

    start = time.time()

    logits = model.forward(X_batch)
    loss, grad = cross_entropy_loss(logits, y_batch)
    model.backward(grad)

    backprop_time = time.time() - start

    backprop_grads = get_backprop_gradients(model)

    start = time.time()

    numeric_grads = numerical_gradients(model, X_batch, y_batch, h)

    numeric_time = time.time() - start

    print("h:", h)
    print("loss:", round(loss, 6))
    print("backprop time:", round(backprop_time, 6), "seconds")
    print("numerical time:", round(numeric_time, 6), "seconds")

    for name in backprop_grads:
        diff = relative_difference(backprop_grads[name], numeric_grads[name])
        print(name, "relative difference:", diff)

    print()


if __name__ == "__main__":
    compare_gradients(1e-1)
    compare_gradients(1e-3)
    compare_gradients(1e-5)