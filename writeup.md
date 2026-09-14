# Manual Backpropagation Neural Network

## 1. Objective

The objective of this project is to build a small artificial neural network and implement the backpropagation algorithm manually using NumPy.

The implementation does not use automatic differentiation or built-in backpropagation functions. All gradients are derived using the chain rule and calculated manually.

The network is trained on a binary classification task using the Iris dataset. The manually calculated gradients are independently verified against PyTorch autograd to check the correctness of the implementation.

## 2. Network Architecture

The neural network used in this project has the following architecture:

Input Layer → Hidden Layer → Output Layer

- Input layer: 4 features
- Hidden layer: 8 neurons
- Hidden activation: ReLU
- Output layer: 2 neurons
- Output activation: Softmax
- Loss function: Cross-entropy

The model can therefore be represented as:

4 → 8 → 2

The Iris dataset contains three classes, but this project uses only the first two classes to create a binary classification problem.

## 3. Dataset and Preprocessing

The Iris dataset was obtained using scikit-learn.

Only the first two Iris classes were selected, resulting in 100 total samples. The dataset contains four input features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The data was divided into:
- 80 training samples
- 20 testing samples

The training and testing split was performed using stratified sampling with a fixed random state.

The input features were standardized using `StandardScaler`. The scaler was fitted only on the training data and then applied to the test data.

The class labels were converted into one-hot encoded vectors for use with the cross-entropy loss function.

## 4. Forward Pass

The forward pass calculates the network's prediction from the input data.

The first linear layer is:
Z1 = XW1 + b1

The result is passed through the ReLU activation function:
A1 = ReLU(Z1)
where:
ReLU(x) = max(0, x)
The hidden-layer output is then passed to the second linear layer:
Z2 = A1W2 + b2

Finally, the output is converted into class probabilities using Softmax:
A2 = Softmax(Z2)
The Softmax function converts the two output values into probabilities whose sum is 1.
The complete forward pass is therefore:
X → Linear Layer → ReLU → Linear Layer → Softmax → Prediction

For numerical stability, the maximum value in each row of Z2 is subtracted before calculating the exponential in the Softmax function.
The implementation was tested by checking that the output probabilities had the expected shape and that the probabilities for each sample summed to approximately 1.

## 5. Cross-Entropy Loss
The network uses categorical cross-entropy to measure the difference between the predicted probabilities and the true class labels.

For one sample, the loss is:
L = -Σ y_i log(p_i)

where:
- y_i is the true one-hot encoded label
- p_i is the predicted probability for class i

The final loss is the mean loss over all training samples.
A small epsilon value is used when calculating the logarithm to prevent numerical problems caused by log(0).
The initial loss was approximately 0.693, which is close to the expected value for a two-class classifier making nearly random predictions.
As training progressed, the loss decreased significantly, demonstrating that the network was learning from the data.

## 6. Manual Backpropagation

Backpropagation calculates how much each parameter contributed to the final loss.
The gradients are derived using the chain rule and are calculated manually using NumPy.
The forward pass is:
X → Z1 → A1 → Z2 → A2 → Loss
During backpropagation, the gradients flow in the reverse direction:
Loss → A2 → Z2 → A1 → Z1 → W1

### 6.1 Gradient at the Output Layer

Because the output layer uses Softmax together with cross-entropy loss, their derivatives simplify to:
dZ2 = (A2 - Y) / m
where:
- A2 is the predicted probability matrix
- Y is the one-hot encoded target matrix
- m is the number of training samples
This gives the gradient of the loss with respect to the output layer's pre-activation values.

### 6.2 Gradients for the Second Layer

The second layer is:
Z2 = A1W2 + b2
Using the chain rule:
dW2 = A1ᵀ dZ2
The bias gradient is
db2 = Σ dZ2
The gradient that needs to be propagated back into the hidden layer is:
dA1 = dZ2 W2ᵀ

### 6.3 Backpropagation Through ReLU

The hidden layer uses the ReLU activation:
A1 = ReLU(Z1)
The derivative of ReLU is:
dReLU/dZ1 = 1, if Z1 > 0
            0, if Z1 ≤ 0
Therefore:
dZ1 = dA1 * (Z1 > 0)
The expression `(Z1 > 0)` creates a mask that allows gradients to pass through neurons where the ReLU was active and sets the gradient to zero for inactive neurons.

### 6.4 Gradients for the First Layer

The first layer is:Z1 = XW1 + b1
Therefore:
dW1 = Xᵀ dZ1
and:
db1 = Σ dZ1
These gradients are then used to update the parameters.

### 6.5 Parameter Update

Gradient descent is used to update the weights and biases:
W = W - η dW
b = b - η db
where η is the learning rate.
In this project, the learning rate was set to 0.1.
No automatic differentiation or built-in backpropagation function is used in the neural network implementation. The gradients are calculated directly from the mathematical derivatives above.

## 7. Training Process

The network was trained using gradient descent.

For each training epoch, the following steps were performed:

1. Perform a forward pass to calculate predictions.
2. Calculate the cross-entropy loss.
3. Perform the manual backward pass to calculate gradients.
4. Update all weights and biases using gradient descent.
5. Store the loss value for visualization.

The training configuration was:

- Learning rate: 0.1
- Number of epochs: 1000
- Hidden neurons: 8
- Optimizer: Gradient Descent
- Batch: Full training dataset

The loss was recorded after every epoch so that the learning behavior of the network could be visualized.

The training loss decreased from approximately 0.693 at the beginning to less than 0.001 by the end of training.

This large decrease in loss indicates that the manually implemented network successfully learned to distinguish between the two Iris classes.

## 8. Results

The network successfully learned the binary classification task.

The training loss decreased consistently during training:

| Epoch | Loss |
|------:|-----:|
| 0 | 0.6933 |
| 100 | 0.0404 |
| 200 | 0.0081 |
| 300 | 0.0041 |
| 400 | 0.0027 |
| 500 | 0.0019 |
| 600 | 0.0015 |
| 700 | 0.0012 |
| 800 | 0.0010 |
| 900 | 0.0009 |

The loss curve is saved in:

`results/loss_curve.png`

The graph shows a rapid reduction in loss during the early stages of training, followed by a slower decrease as the model approaches a low-loss solution.

The model was also evaluated on the held-out test set after training.
The model achieved 100.0% test accuracy (20/20 correct predictions) on the held-out test set. However, this result should be interpreted cautiously because the dataset is small and the task uses only two relatively separable Iris classes

## 9. Gradient Verification

To verify the correctness of the manually implemented backpropagation, the gradients were compared against gradients calculated independently using PyTorch autograd.

PyTorch was used only as a reference for verification. It is not used anywhere in the actual neural network implementation or training process.

The verification process was:

1. Create a small test dataset.
2. Initialize the NumPy model.
3. Calculate gradients using the manually implemented `backward()` function.
4. Create an equivalent computation in PyTorch using the same inputs and parameters.
5. Use PyTorch autograd to calculate reference gradients.
6. Compare the manual gradients with the reference gradients.
7. Check whether the maximum absolute difference is below a specified tolerance.

The tolerance used was:

`1 × 10⁻⁷`

### Verification Results

The comparison produced the following results:

| Parameter | Maximum Difference | Result |
|-----------|-------------------:|--------|
| W1 | 4.34 × 10⁻¹⁹ | PASS |
| b1 | 2.17 × 10⁻¹⁹ | PASS |
| W2 | 8.67 × 10⁻¹⁹ | PASS |
| b2 | 2.78 × 10⁻¹⁷ | PASS |

All differences were far below the chosen tolerance.

The final output of the verification program was:

`ALL GRADIENT TESTS PASSED`

This provides strong evidence that the manually derived gradients match the gradients produced by an independent automatic differentiation system.

### Important Implementation Constraint

The neural network itself does not use:

- `torch.autograd`
- `.backward()`
- PyTorch tensors for training
- Any automatic differentiation library

The actual forward pass, loss calculation, backpropagation, and parameter updates are implemented using NumPy.

PyTorch autograd is used only in the separate gradient verification test.

## 10. Gradient Debugging and Common Mistakes

Implementing backpropagation manually requires careful handling of matrix dimensions, activation derivatives, and averaging over the training samples.

Several potential sources of gradient errors were considered during implementation.

### 10.1 ReLU Derivative

A common mistake when implementing backpropagation is to propagate the gradient through the ReLU activation without applying its derivative.

The correct calculation is:

dZ1 = dA1 * (Z1 > 0)

The `(Z1 > 0)` term acts as a mask. Without this mask, gradients would incorrectly pass through neurons whose ReLU activation was zero.

### 10.2 Averaging the Output Gradient

Another important detail is the division by the number of samples:

dZ2 = (A2 - Y) / m

The loss function calculates the mean loss over all samples, so the corresponding gradient must also be averaged over the batch.

### 10.3 Matrix Dimensions

The dimensions of every gradient were checked to ensure that the chain rule was being applied correctly.

For the training data, the main dimensions are:

- X: (80, 4)
- W1: (4, 8)
- b1: (1, 8)
- W2: (8, 2)
- b2: (1, 2)
- dW1: (4, 8)
- db1: (1, 8)
- dW2: (8, 2)
- db2: (1, 2)

Checking these dimensions helped ensure that the matrix multiplications represented the intended mathematical operations.

### 10.4 Gradient Verification as a Debugging Tool

Instead of relying only on the training loss, the implementation was tested using an independent gradient verification program.

The manually calculated gradients were compared with PyTorch autograd gradients using the same parameters and input data.

All four gradients passed the verification test with differences far below the selected tolerance.

This verification made it possible to distinguish between problems in the gradient calculations and other possible training issues such as the learning rate or data preprocessing.

## 11. Limitations and Future Improvements

Although the network successfully performs the classification task, the implementation is intentionally small and educational.

Some limitations are:

- The network contains only one hidden layer.
- The model is trained using full-batch gradient descent.
- Only the first two Iris classes are used.
- The dataset is relatively small and simple.
- The implementation does not include advanced optimizers such as Adam.
- There is no regularization such as dropout or L2 regularization.

Possible future improvements include:

- Adding more hidden layers.
- Supporting additional activation functions such as Sigmoid or Tanh.
- Implementing Momentum or Adam manually.
- Supporting mini-batch training.
- Extending the model to all three Iris classes.
- Adding numerical gradient checking in addition to the PyTorch comparison.
- Experimenting with different learning rates and network sizes.

These improvements would make the implementation more flexible while still allowing the underlying mathematics of backpropagation to remain visible.

## 12. Conclusion

This project demonstrates a complete neural network implemented from the ground up using NumPy.

The implementation includes:

- Data preprocessing
- Forward propagation
- ReLU activation
- Softmax output
- Cross-entropy loss
- Manual backpropagation
- Gradient descent
- Model evaluation
- Training loss visualization
- Independent gradient verification

The most important part of the project was implementing backpropagation manually using the chain rule rather than relying on an automatic differentiation framework.

The gradient verification results showed that the manually calculated gradients closely matched PyTorch's independently calculated gradients, providing strong evidence that the implementation is mathematically correct.

The project therefore demonstrates both the practical implementation and the underlying mathematics of neural network training.