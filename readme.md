# Iris Neural Network

A simple neural network that classifies the Iris dataset into 3 classes.

## Requirements

- Python 3.9+
- NumPy
- scikit-learn
- Matplotlib

Install dependencies:

```bash
pip install numpy scikit-learn matplotlib
```

## Project Structure

```text
project-folder/
├── src/
│   └── neural_network.py
├── results/
├── main.py
└── README.md
```

Make sure `src/neural_network.py` contains the `NeuralNetwork` class with:

```python
forward()
cross_entropy_loss()
backward()
update_parameters()
```

## Run the Code

Open a terminal in the project folder and run:

```bash
python main.py
```

If the `results` folder does not exist, create it first:

```bash
mkdir results
```

## What the Code Does

The program:

1. Loads the Iris dataset.
2. Splits it into:
   - 100 training samples
   - 25 validation samples
   - 25 test samples
3. Standardizes the features using the training data.
4. Creates a neural network:
   - Input: 4 features
   - Hidden layer: 8 neurons
   - Output: 3 classes
5. Trains the model for 1000 epochs.
6. Calculates validation and test accuracy.
7. Saves the training/validation loss graph.

## Expected Output

The terminal will display information such as:

```text
Training samples: 100
Validation samples: 25
Testing samples: 25
Number of input features: 4

Initial loss: ...
Epoch 0, Loss: ..., Validation Loss: ...
...
Validation accuracy: ...
Test accuracy: ...
```

The prediction output should have shape:

```text
(100, 3)
```

## Results

After training, the loss graph is saved as:

```text
results/loss_curve.png
```

The graph shows:

- Training Loss
- Validation Loss

The final validation and test accuracy are printed in the terminal.

## Quick Start

```bash
pip install numpy scikit-learn matplotlib
mkdir results
python main.py
```

