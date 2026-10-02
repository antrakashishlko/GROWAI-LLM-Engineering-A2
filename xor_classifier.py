"""
Neural Network from Scratch: XOR Classifier
This project builds a simple 2-layer neural network in PyTorch to learn the XOR function.
The network uses 2 input neurons, 4 hidden neurons with ReLU and 1 Sigmoid output neuron.
It is trained using BCELoss and Adam optimization for 5000 epochs.
The final predictions are compared with the expected XOR outputs to verify correctness.
"""

import torch
import torch.nn as nn
import torch
import torch.nn as nn

# -----------------------------
# XOR Dataset
# -----------------------------

X = torch.tensor(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=torch.float32,
)

y = torch.tensor(
    [
        [0],
        [1],
        [1],
        [0],
    ],
    dtype=torch.float32,
)

# -----------------------------
# Edge Case Handling
# -----------------------------

if len(X) != 4 or len(y) != 4:
    raise ValueError(
        "XOR dataset must contain exactly 4 input-output pairs."
    )

# -----------------------------
# Neural Network Definition
# -----------------------------

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

# -----------------------------
# Model Initialization
# -----------------------------

model = XORNet()

print("\nModel:")
print(model)

# -----------------------------
# Loss Function & Optimizer
# -----------------------------

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)

# -----------------------------
# Training Configuration
# -----------------------------

epochs = 5000

# -----------------------------
# Training Loop
# -----------------------------

for epoch in range(epochs):
    # Forward pass
    predictions = model(X)

    # Calculate loss
    loss = criterion(predictions, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Display loss every 500 epochs
    if (epoch + 1) % 500 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.6f}"
        )

# -----------------------------
# Final Predictions
# -----------------------------

print("\nFinal Predictions:")

model.eval()

with torch.no_grad():
    predictions = model(X)

    for i in range(len(X)):
        probability = predictions[i].item()
        predicted_class = 1 if probability >= 0.5 else 0
        expected_class = int(y[i].item())

        print(
            f"Input: {X[i].tolist()} | "
            f"Predicted: {predicted_class} | "
            f"Expected: {expected_class} | "
            f"Probability: {probability:.4f}"
        )
