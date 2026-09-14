# Manual Backpropagation Neural Network

## 1. Objective

The objective of this project is to implement and train a small feedforward neural network while manually deriving and calculating the backpropagation gradients using the chain rule.

The neural network is implemented using NumPy and does not use automatic differentiation for training. The gradients are calculated explicitly from the mathematical derivatives of each layer.

The model is trained on the Iris dataset for a 3-class classification task. A separate validation set is used to monitor generalization and identify possible overfitting. The manually calculated gradients are independently verified against PyTorch autograd.

---

## 2. Network Architecture and Dataset

The network has the following architecture:

**Input (4) → Hidden Layer (8) → ReLU → Output (3) → Softmax**

### Network Configuration

- Input features: 4
- Hidden neurons: 8
- Output neurons: 3
- Hidden activation: ReLU
- Output activation: Softmax
- Loss function: Cross-entropy
- Optimizer: Full-batch Gradient Descent
- Learning rate: 0.1
- Epochs: 1000

The Iris dataset from scikit-learn is used. Unlike the earlier binary version of the project, all three Iris classes are now included.

The dataset contains 150 samples and is divided into:

- 100 training samples
- 25 validation samples
- 25 test samples

The four input features are:

- Sepal length
- Sepal width
- Petal length
- Petal width

The three classes are:

- Iris setosa
- Iris versicolor
- Iris virginica

Stratified sampling is used when splitting the dataset so that the class distribution is maintained across the different subsets.

---

## 3. Data Preprocessing

The input features are standardized using `StandardScaler`.

The scaler is fitted only on the training data:

\[
X_{train}' = StandardScaler.fit(X_{train})
\]

The fitted scaler is then applied to the validation and test sets.

This is important because fitting the scaler using validation or test data could introduce data leakage.

The class labels are converted into one-hot encoded vectors because the network uses categorical cross-entropy loss.

The final data distribution is:

| Dataset | Number of Samples | Purpose |
|---|---:|---|
| Training | 100 | Used for learning model parameters |
| Validation | 25 | Used to monitor generalization |
| Test | 25 | Used for final evaluation |

---

## 4. Forward Pass

The forward pass calculates predictions from the input data.

The first linear layer is:

\[
Z_1 = XW_1+b_1
\]

The ReLU activation is then applied:

\[
A_1 = ReLU(Z_1)
\]

where:

\[
ReLU(x)=\max(0,x)
\]

The second linear layer is:

\[
Z_2=A_1W_2+b_2
\]

The final output is calculated using Softmax:

\[
A_2=Softmax(Z_2)
\]

The complete forward pass is:

**Input → Linear → ReLU → Linear → Softmax → Class Probabilities**

A numerically stable implementation of Softmax is used by subtracting the maximum value in each row before calculating the exponential.

For the current implementation, the output shape is:

\[
(100,3)
\]

because there are 100 training samples and 3 output classes.

The first prediction from the training run was:

```text
[0.33331243 0.3333818  0.33330577]
```

The probabilities sum to:

```text
1.0
```

This confirms that the Softmax output represents a valid probability distribution.

---

## 5. Cross-Entropy Loss

The network uses categorical cross-entropy loss.

For one-hot encoded targets:

\[
L=-\frac{1}{m}\sum_{j=1}^{m}\sum_i y_{ji}\log(p_{ji})
\]

where:

- \(m\) is the number of samples
- \(y\) is the one-hot encoded target
- \(p\) is the predicted probability

A small epsilon value is used before taking the logarithm to avoid numerical problems caused by `log(0)`.

The initial loss was:

\[
\boxed{1.09865}
\]

For a three-class classifier making nearly uniform initial predictions, the expected cross-entropy is approximately:

\[
\ln(3)\approx1.0986
\]

which agrees with the observed initial loss.

---

## 6. Manual Backpropagation

Backpropagation calculates the gradient of the loss with respect to every trainable parameter.

The forward computation is:

**X → Z1 → A1 → Z2 → A2 → Loss**

During backpropagation, the gradients are calculated in the reverse direction using the chain rule.

### Output Layer

For Softmax combined with cross-entropy loss, the derivative simplifies to:

\[
dZ_2=\frac{A_2-Y}{m}
\]

For:

\[
Z_2=A_1W_2+b_2
\]

the gradients are:

\[
dW_2=A_1^TdZ_2
\]

\[
db_2=\sum dZ_2
\]

The gradient passed to the hidden layer is:

\[
dA_1=dZ_2W_2^T
\]

### ReLU Derivative

Since:

\[
A_1=ReLU(Z_1)
\]

the derivative is:

\[
ReLU'(Z_1)=
\begin{cases}
1 & Z_1>0\\
0 & Z_1\leq0
\end{cases}
\]

Therefore:

\[
dZ_1=dA_1*(Z_1>0)
\]

The expression `(Z1 > 0)` acts as a mask that allows gradients to pass through active ReLU neurons.

### First Layer

For:

\[
Z_1=XW_1+b_1
\]

the gradients are:

\[
dW_1=X^TdZ_1
\]

\[
db_1=\sum dZ_1
\]

The parameters are updated using gradient descent:

\[
W=W-\eta dW
\]

\[
b=b-\eta db
\]

where \(\eta\) is the learning rate.

The learning rate used is:

\[
\eta=0.1
\]

The resulting gradient dimensions from the current implementation are:

| Gradient | Shape |
|---|---|
| dW1 | (4, 8) |
| db1 | (1, 8) |
| dW2 | (8, 3) |
| db2 | (1, 3) |

These dimensions are consistent with the network architecture.

---

## 7. Training

The network is trained using full-batch gradient descent for 1000 epochs.

For every epoch:

1. Perform a forward pass on the training data.
2. Calculate the training loss.
3. Perform manual backpropagation.
4. Update the weights and biases.
5. Perform a forward pass on the validation data.
6. Calculate and store the validation loss.

The training and validation losses were recorded throughout the process.

### Training Results

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 0 | 1.0987 | 1.0986 |
| 100 | 0.5069 | 0.5090 |
| 200 | 0.2492 | 0.2291 |
| 300 | 0.1494 | 0.1503 |
| 400 | 0.1024 | 0.1170 |
| 500 | 0.0796 | 0.1005 |
| 600 | 0.0665 | 0.0923 |
| 700 | 0.0579 | 0.0890 |
| 800 | 0.0517 | 0.0882 |
| 900 | 0.0470 | 0.0891 |

The training loss decreased consistently from approximately **1.0987** to **0.0470**.

The validation loss also decreased substantially, reaching a minimum of approximately **0.0882 around epoch 800**, before increasing slightly to **0.0891** at epoch 900.

This small increase near the end suggests that the model may begin to overfit if training continues indefinitely, although the validation performance remains strong.

---

## 8. Validation and Test Results

The model was evaluated separately on the validation and test sets after training.

### Validation Accuracy

The final validation accuracy was:

\[
\boxed{96.0\%}
\]

This corresponds to:

\[
24/25
\]

correct validation predictions.

### Test Accuracy

The final test accuracy was:

\[
\boxed{88.0\%}
\]

This corresponds to:

\[
22/25
\]

correct test predictions.

### Summary

| Metric | Result |
|---|---:|
| Training samples | 100 |
| Validation samples | 25 |
| Test samples | 25 |
| Final validation accuracy | **96.0%** |
| Final test accuracy | **88.0%** |
| Final training loss at epoch 900 | **0.0470** |
| Validation loss at epoch 900 | **0.0891** |
| Lowest recorded validation loss | **0.0882** |

The difference between validation accuracy and test accuracy also shows why a separate test set is useful. Performance on a small dataset can vary depending on which samples happen to be included in each split.

---

## 9. Mistakes, Overfitting and Improvements

During the development of the project, an important issue was identified with the earlier dataset setup.

### Initial Problem: Overfitting

The original version of the project used only two Iris classes, giving a total of **100 samples**. The data was split into:

- 80 training samples
- 20 test samples

The model achieved **100% test accuracy** on that setup.

However, this result was potentially misleading because the dataset was very small and the selected two classes were relatively easy to separate. The model was also being trained on a large proportion of the available data, making it difficult to properly monitor generalization during training.

This raised a concern that the model could be fitting the training data too closely rather than demonstrating robust generalization.

### Solution: Larger Dataset and Validation Split

To address this issue, the dataset was changed to use the **complete Iris dataset**, including all three classes.

The new split is:

```text
150 total samples
       │
       ├── 100 Training
       │
       └── 50 Remaining
              │
              ├── 25 Validation
              │
              └── 25 Test
```

This provides a dedicated validation set for monitoring generalization during training.

The validation loss is calculated after each training update. By comparing training and validation loss, we can identify behavior such as:

```text
Training loss ↓
Validation loss ↓
        → Good generalization

Training loss ↓
Validation loss ↑
        → Possible overfitting
```

In the current run, the validation loss decreased from **1.0986** to a minimum of approximately **0.0882**, while the training loss continued decreasing. The validation loss then increased slightly to **0.0891** by epoch 900.

Therefore, the current run does not show severe overfitting, but the small increase in validation loss near the end indicates that continued training could eventually lead to overfitting.

The final test accuracy of **88.0%** is also more conservative than the previous 100% result, giving a more realistic indication of generalization on unseen data.

---

## 10. Gradient Verification

Manual backpropagation is susceptible to mistakes such as:

- Incorrect matrix dimensions
- Missing activation derivatives
- Incorrect batch averaging
- Incorrect transposes
- Incorrect signs in parameter updates

A separate gradient verification program is used to check the manually calculated gradients.

The NumPy network is compared with an equivalent PyTorch implementation using the same parameters and inputs.

PyTorch autograd is used only as an independent reference and is not used for training the actual NumPy neural network.

The gradients checked are:

- \(W_1\)
- \(b_1\)
- \(W_2\)
- \(b_2\)

The comparison uses a tolerance of:

\[
10^{-7}
\]

The expected successful output is:

```text
ALL GRADIENT TESTS PASSED
```

This provides strong evidence that the manually derived gradients have been implemented correctly.

---

## 11. Loss Visualization

The training program plots both training and validation loss:

```text
Training Loss
Validation Loss
```

The resulting graph is saved to:

```text
results/loss_curve.png
```

The two curves provide a visual way to inspect learning and generalization.

The training loss continues to decrease throughout the run, while the validation loss decreases rapidly at first and then levels off.

This behavior is consistent with a model that learns the classification task effectively while showing a small amount of possible overfitting near the later epochs.

---

## 12. Limitations

The implementation is intentionally small so that the mathematics of neural network training remains visible.

The main limitations are:

- Only one hidden layer is used.
- The hidden layer contains 8 neurons.
- Full-batch gradient descent is used.
- The Iris dataset is small.
- No regularization is implemented.
- No early stopping is implemented.
- The learning rate is fixed at 0.1.
- Validation and test sets contain only 25 samples each, so their accuracy can vary significantly with the particular split.

Because of the small dataset size, the reported accuracy should not be interpreted as evidence of performance on larger or more difficult datasets.

---

## 13. Conclusion

This project demonstrates a complete feedforward neural network implemented from the ground up using NumPy.

The implementation includes:

- Data preprocessing
- Stratified train/validation/test splitting
- Forward propagation
- ReLU activation
- Softmax
- Cross-entropy loss
- Manual backpropagation
- Gradient descent
- Validation monitoring
- Test evaluation
- Loss visualization
- Independent gradient verification

The major improvement during development was recognizing that the earlier binary Iris setup could give an overly optimistic evaluation. The project was therefore changed to use the complete **150-sample Iris dataset**, divided into **100 training, 25 validation, and 25 test samples**.

The final run achieved **96.0% validation accuracy** and **88.0% test accuracy**. The validation loss decreased substantially and remained close to the training loss, with only a small increase near the end of training.

The project demonstrates not only how to implement backpropagation manually, but also why separating training, validation, and test data is important when evaluating whether a neural network is actually generalizing to unseen data.
