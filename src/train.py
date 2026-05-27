import numpy as np

from data import load_data
from mlp import MLP
from losses import cross_entropy_loss, mse_loss
from metrics import accuracy, mean_absolute_error, root_mean_squared_error


def train_classification():
    X_train, X_test, y_train, y_test = load_data(task="classification")

    model = MLP(
        input_size=X_train.shape[1],
        hidden_size_1=32,
        hidden_size_2=16,
        output_size=3
    )

    epochs = 300
    learning_rate = 0.05 # constant value, no AdaM
    batch_size = 32

    for epoch in range(1,epochs+1):
        indexes = np.random.permutation(len(X_train))


            # mini batch Stochastic Gradient Descent SGD

        for start in range(0, len(X_train), batch_size):
            batch_indexes = indexes[start:start + batch_size]

            X_batch = X_train[batch_indexes]
            y_batch = y_train[batch_indexes]

            logits = model.forward(X_batch)
            loss, grad = cross_entropy_loss(logits, y_batch)

            model.backward(grad)
            model.update(learning_rate) # updating weight for each batch makes it mini-batch SGD

        if epoch == 1 or epoch  % 50 == 0:
            test_logits = model.forward(X_test)
            test_accuracy = accuracy(test_logits, y_test)

            print("epoch:", epoch, "loss:", round(loss, 4), "test accuracy:", round(test_accuracy, 4))

    return model


def train_regression():
    X_train, X_test, y_train, y_test = load_data(task="regression")


    # normilize data with statistic norm
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_scaled = (y_train - y_mean) / y_std

    model = MLP(
        input_size=X_train.shape[1],
        hidden_size_1=32,
        hidden_size_2=16,
        output_size=1
    )

    # initialize defaults
    epochs = 500
    learning_rate = 0.01 # constant value, no AdaM
    batch_size = 32


    for epoch in range(1,epochs+1):
        # mix train data, so that mini batch SGD wouldn't be biased because of the order
        indexes = np.random.permutation(len(X_train))

        for start in range(0, len(X_train), batch_size):
            batch_indexes = indexes[start:start + batch_size]

            X_batch = X_train[batch_indexes]
            y_batch = y_train_scaled[batch_indexes]

            predictions = model.forward(X_batch)
            loss, grad = mse_loss(predictions, y_batch)

            model.backward(grad)
            model.update(learning_rate)

        if  epoch == 1 or epoch % 100 == 0:
            test_predictions_scaled = model.forward(X_test)
            test_predictions = test_predictions_scaled * y_std + y_mean

            mae = mean_absolute_error(test_predictions, y_test)
            rmse = root_mean_squared_error(test_predictions, y_test)

            print("epoch:", epoch, "loss:", round(loss, 4), "MAE:", round(mae, 2), "RMSE:", round(rmse, 2))

    return model


if __name__ == "__main__":
    print("Classification")
    train_classification()

    print()
    print("Regression")
    train_regression()