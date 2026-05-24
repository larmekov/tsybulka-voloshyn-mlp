import time
import numpy as np
import torch
import torch.nn as nn

from data import load_data
from metrics import accuracy, mean_absolute_error, root_mean_squared_error


class TorchMLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size)
        )

    def forward(self, x):
        return self.layers(x)


def train_torch_classification():
    X_train, X_test, y_train, y_test = load_data(task="classification")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test_np = y_test

    model = TorchMLP(
        input_size=X_train.shape[1],
        hidden_size_1=32,
        hidden_size_2=16,
        output_size=3
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    epochs = 300
    batch_size = 32

    start_time = time.time()

    for epoch in range(epochs):
        indexes = torch.randperm(len(X_train))

        for start in range(0, len(X_train), batch_size):
            batch_indexes = indexes[start:start + batch_size]

            X_batch = X_train[batch_indexes]
            y_batch = y_train[batch_indexes]

            logits = model(X_batch)
            loss = loss_function(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    training_time = time.time() - start_time

    with torch.no_grad():
        test_logits = model(X_test).numpy()

    test_accuracy = accuracy(test_logits, y_test_np)

    print("PyTorch classification")
    print("accuracy:", round(test_accuracy, 4))
    print("training time:", round(training_time, 4), "seconds")
    print()


def train_torch_regression():
    X_train, X_test, y_train, y_test = load_data(task="regression")

    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train_scaled = (y_train - y_mean) / y_std

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32).reshape(-1, 1)

    model = TorchMLP(
        input_size=X_train.shape[1],
        hidden_size_1=32,
        hidden_size_2=16,
        output_size=1
    )

    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 500
    batch_size = 32

    start_time = time.time()

    for epoch in range(epochs):
        indexes = torch.randperm(len(X_train))

        for start in range(0, len(X_train), batch_size):
            batch_indexes = indexes[start:start + batch_size]

            X_batch = X_train[batch_indexes]
            y_batch = y_train_scaled[batch_indexes]

            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    training_time = time.time() - start_time

    with torch.no_grad():
        test_predictions_scaled = model(X_test).numpy()

    test_predictions = test_predictions_scaled * y_std + y_mean

    mae = mean_absolute_error(test_predictions, y_test)
    rmse = root_mean_squared_error(test_predictions, y_test)

    print("PyTorch regression")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("training time:", round(training_time, 4), "seconds")
    print()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    train_torch_classification()
    train_torch_regression()