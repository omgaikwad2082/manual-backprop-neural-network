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