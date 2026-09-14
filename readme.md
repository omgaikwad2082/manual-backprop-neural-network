# Manual Backpropagation Neural Network

A small feedforward neural network implemented from scratch using NumPy, with the backpropagation algorithm manually derived and implemented using the chain rule.
The project uses a binary classification version of the Iris dataset and independently verifies the manually calculated gradients against PyTorch autograd.

## Features
- Manual neural network implementation using NumPy
- Forward propagation
- ReLU activation
- Softmax output
- Cross-entropy loss
- Manual backpropagation
- Gradient descent training
- Training loss visualization
- Independent gradient verification using PyTorch
- Test-set evaluation

## Network Architecture

Input Layer (4)
       ↓
Hidden Layer (8)
       ↓
     ReLU
       ↓
Output Layer (2)
       ↓
    Softmax


## Dataset
The Iris dataset from scikit-learn is used.
Only the first two classes are selected, creating a binary classification problem.

- Total samples: 100
- Training samples: 80
- Testing samples: 20
- Input features: 4

The four input features are:

- Sepal length
- Sepal width
- Petal length
- Petal width

The features are standardized using `StandardScaler`.

## Training

The model uses full-batch gradient descent.

| Parameter      | Value           |
| Hidden neurons | 8               |
| Learning rate  | 0.1             |
| Epochs         | 1000            |
| Optimizer      | Gradient Descent|

The training loss decreased from approximately `0.6933` to `0.0009`.

The model achieved:

**Test Accuracy: 100.0% (20/20)**

Because the dataset is small and the two selected classes are relatively separable, this result should not be interpreted as evidence of general performance on more difficult datasets.

## Manual Backpropagation

The gradients are calculated manually using the chain rule.

For the output layer:


dZ2 = (A2 - Y) / m
dW2 = A1.T @ dZ2
db2 = sum(dZ2)


The gradient is propagated through ReLU:


dZ1 = dA1 * (Z1 > 0)


Then:

dW1 = X.T @ dZ1
db1 = sum(dZ1)


The parameters are updated using gradient descent:


W = W - learning_rate * dW
b = b - learning_rate * db


No automatic differentiation is used in the actual neural network implementation.

## Gradient Verification

The manually calculated gradients were compared with an equivalent PyTorch implementation.

| Parameter | Maximum Difference | Result |
|-----------|-------------------:|--------|
| W1        | 4.34 × 10⁻¹⁹       | PASS   |
| b1        | 2.17 × 10⁻¹⁹       | PASS   |
| W2        | 8.67 × 10⁻¹⁹       | PASS   |
| b2        | 2.78 × 10⁻¹⁷       | PASS   |

All gradients passed the verification test.



PyTorch autograd is used only as an independent verification reference and is not used for training the NumPy network.

## Results

The training loss curve is generated during training and saved at:
results/loss_curve.png


The loss decreases consistently during training, demonstrating that the manually implemented network is learning the classification task.

## Project Structure


manual-backprop-neural-network/
│
├── README.md
├── WRITEUP.md
├── requirements.txt
├── train.py
│
├── src/
│   └── neural_network.py
│
├── test/
│   └── test_gradients.py
│
└── results/
    └── loss_curve.png


## Installation

Install the required Python packages using:

```bash
python -m pip install -r requirements.txt
```

## Run Training

Run the training program from the project root:

```bash
python train.py
```

This will:

1. Load and preprocess the Iris dataset.
2. Train the neural network for 1000 epochs.
3. Display the training loss.
4. Evaluate the model on the test set.
5. Generate the training loss graph.

The loss graph is saved to:


results/loss_curve.png


## Run Gradient Verification

Run the independent gradient verification test:

```bash
python test/test_gradients.py
```

A successful verification produces:


ALL GRADIENT TESTS PASSED

## Technologies

- Python
- NumPy
- scikit-learn
- Matplotlib
- PyTorch (gradient verification only)

## Future Improvements

Possible extensions include:

- Additional activation functions such as Sigmoid or Tanh
- Mini-batch training
- Manual implementation of Momentum or Adam
- Numerical gradient checking
- Multi-class classification using all three Iris classes
- Additional hidden layers
- Experimenting with different learning rates and network architectures