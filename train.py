import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Keep only two classes for binary classification
mask = y < 2

X = X[mask]
y = y[mask]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Standardize the input features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of input features:", X_train.shape[1])

from src.neural_network import NeuralNetwork


# Create the neural network
model = NeuralNetwork(
    input_size=4,
    hidden_size=8,
    output_size=2
)


# Run the forward pass
predictions = model.forward(X_train)

print("Prediction shape:", predictions.shape)
print("First prediction:", predictions[0])
print("Sum of probabilities:", np.sum(predictions[0]))

# Convert labels to one-hot encoding
y_train_one_hot = np.zeros((y_train.size, 2))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

# Calculate the loss
loss = model.cross_entropy_loss(
    y_train_one_hot,
    predictions
)

print("Initial loss:", loss)