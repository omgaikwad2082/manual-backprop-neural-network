import sys
from pathlib import Path

import numpy as np
import torch


sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.neural_network import NeuralNetwork



np.random.seed(42)


X = np.random.randn(5, 4)

y = np.array([0, 1, 0, 1, 0])

y_one_hot = np.zeros((5, 2))
y_one_hot[np.arange(5), y] = 1


# Create our NumPy model
model = NeuralNetwork(
    input_size=4,
    hidden_size=8,
    output_size=2
)

# Forward pass
predictions = model.forward(X)

# Manual gradients
manual_dW1, manual_db1, manual_dW2, manual_db2 = model.backward(
    X,
    y_one_hot
)

# PyTorch reference implementation

X_t = torch.tensor(X, dtype=torch.float64)

W1_t = torch.tensor(
    model.W1,
    dtype=torch.float64,
    requires_grad=True
)

b1_t = torch.tensor(
    model.b1,
    dtype=torch.float64,
    requires_grad=True
)

W2_t = torch.tensor(
    model.W2,
    dtype=torch.float64,
    requires_grad=True
)

b2_t = torch.tensor(
    model.b2,
    dtype=torch.float64,
    requires_grad=True
)

Y_t = torch.tensor(
    y_one_hot,
    dtype=torch.float64
)


# Forward pass in PyTorch
Z1_t = X_t @ W1_t + b1_t
A1_t = torch.relu(Z1_t)

Z2_t = A1_t @ W2_t + b2_t
A2_t = torch.softmax(Z2_t, dim=1)

epsilon = 1e-12

loss_t = -torch.mean(
    torch.sum(
        Y_t * torch.log(torch.clamp(A2_t, min=epsilon)),
        dim=1
    )
)

# PyTorch calculates reference gradients
loss_t.backward()


# Convert PyTorch gradients to NumPy
torch_dW1 = W1_t.grad.numpy()
torch_db1 = b1_t.grad.numpy()
torch_dW2 = W2_t.grad.numpy()
torch_db2 = b2_t.grad.numpy()



# Compare gradients


tolerance = 1e-7

tests = {
    "W1": (manual_dW1, torch_dW1),
    "b1": (manual_db1, torch_db1),
    "W2": (manual_dW2, torch_dW2),
    "b2": (manual_db2, torch_db2)
}

all_passed = True

print("Gradient Verification")
print("---------------------")

for name, (manual, reference) in tests.items():

    max_difference = np.max(
        np.abs(manual - reference)
    )

    passed = max_difference < tolerance

    if passed:
        print(
            f"{name}: PASS "
            f"(max difference = {max_difference:.2e})"
        )
    else:
        print(
            f"{name}: FAIL "
            f"(max difference = {max_difference:.2e})"
        )

        all_passed = False


print("---------------------")

if all_passed:
    print("ALL GRADIENT TESTS PASSED")
else:
    print("GRADIENT TESTS FAILED")
    sys.exit(1)