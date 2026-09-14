import numpy as np


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        # First linear layer
        self.Z1 = X @ self.W1 + self.b1

        # ReLU activation
        self.A1 = self.relu(self.Z1)

        # Second linear layer
        self.Z2 = self.A1 @ self.W2 + self.b2

        # Softmax output
        self.A2 = self.softmax(self.Z2)

        return self.A2

    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-12

        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(
            np.sum(y_true * np.log(y_pred), axis=1)
        )

        return loss
    def backward(self, X, y_true):
        m = X.shape[0]

        # Gradient at the output layer
        dZ2 = (self.A2 - y_true) / m

        # Gradients for W2 and b2
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Gradient flowing into the hidden layer
        dA1 = dZ2 @ self.W2.T

        # Backpropagate through ReLU
        dZ1 = dA1 * (self.Z1 > 0)

        # Gradients for W1 and b1
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2
    def update_parameters(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2