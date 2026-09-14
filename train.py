import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Split the full dataset into train/validation/test sets
# with the requested distribution: 100 training samples,
# 25 validation samples, and 25 test samples.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=50,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# Standardize the input features using only the training split
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of input features:", X_train.shape[1])

from src.neural_network import NeuralNetwork


# Create the neural network
model = NeuralNetwork(
    input_size=4,
    hidden_size=8,
    output_size=3
)


# Run the forward pass on the training split
predictions = model.forward(X_train)

print("Prediction shape:", predictions.shape)
print("First prediction:", predictions[0])
print("Sum of probabilities:", np.sum(predictions[0]))

# Convert labels to one-hot encoding
y_train_one_hot = np.zeros((y_train.size, 3))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

y_val_one_hot = np.zeros((y_val.size, 3))
y_val_one_hot[np.arange(y_val.size), y_val] = 1

# Calculate the loss
loss = model.cross_entropy_loss(
    y_train_one_hot,
    predictions
)

print("Initial loss:", loss)
dW1, db1, dW2, db2 = model.backward(
    X_train,
    y_train_one_hot
)

print("dW1 shape:", dW1.shape)
print("db1 shape:", db1.shape)
print("dW2 shape:", dW2.shape)
print("db2 shape:", db2.shape)

# Training
learning_rate = 0.1
epochs = 1000

loss_history = []
validation_loss_history = []

for epoch in range(epochs):

    # Forward pass
    predictions = model.forward(X_train)

    # Calculate training loss
    loss = model.cross_entropy_loss(
        y_train_one_hot,
        predictions
    )
    loss_history.append(loss)

    # Backward pass
    dW1, db1, dW2, db2 = model.backward(
        X_train,
        y_train_one_hot
    )

    # Update parameters
    model.update_parameters(
        dW1, db1, dW2, db2, learning_rate
    )

    # Evaluate on the validation split to detect overfitting
    val_predictions = model.forward(X_val)
    val_loss = model.cross_entropy_loss(
        y_val_one_hot,
        val_predictions
    )
    validation_loss_history.append(val_loss)

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")

# Evaluate on the validation split and test set
val_predictions = model.forward(X_val)
val_predicted_classes = np.argmax(val_predictions, axis=1)
val_accuracy = np.mean(val_predicted_classes == y_val)

print("Validation accuracy:", val_accuracy)
print("Validation accuracy (%):", val_accuracy * 100)

test_predictions = model.forward(X_test)

test_predicted_classes = np.argmax(test_predictions, axis=1)

accuracy = np.mean(test_predicted_classes == y_test)

print("Test accuracy:", accuracy)
print("Test accuracy (%):", accuracy * 100)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(loss_history, label="Training Loss")
plt.plot(validation_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("results/loss_curve.png")
plt.show()
