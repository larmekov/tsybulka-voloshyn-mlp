import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, random_state=42):
        np.random.seed(random_state)

        self.W1 = np.random.randn(input_size, hidden_size_1) * 0.01
        self.b1 = np.zeros((1, hidden_size_1))

        self.W2 = np.random.randn(hidden_size_1, hidden_size_2) * 0.01
        self.b2 = np.zeros((1, hidden_size_2))

        self.W3 = np.random.randn(hidden_size_2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        # saving training data for this batch
        self.X = X

        # calculate first hidden layer output
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        # calculate 2 hidden layer output
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)

        # final output before sofmax
        self.Z3 = self.A2 @ self.W3 + self.b3

        return self.Z3

    def backward(self, grad_output):
        self.dW3 = self.A2.T @ grad_output
        self.db3 = np.sum(grad_output, axis=0, keepdims=True)

        dA2 = grad_output @ self.W3.T
        dZ2 = dA2 * self.relu_derivative(self.Z2)

        self.dW2 = self.A1.T @ dZ2
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)

        self.dW1 = self.X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def update(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

        self.W3 -= learning_rate * self.dW3
        self.b3 -= learning_rate * self.db3
