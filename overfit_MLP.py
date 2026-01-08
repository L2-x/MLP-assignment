"""
A simple example of overfitting using a Multi-Layer Perceptron (MLP) to fit noisy sine wave data.
Author: Dongyang Kuang

NOTE:
    [] Multiple aspects can be investigated:
"""

# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch


def sin_2pi_on_grid(x):
    """
    Computes y = sin(2pi*x) on a uniform grid from 0 to 1.

    Parameters:
    x (int or array): input for evaluation.

    Returns:
    y (numpy.ndarray): The computed sine values at the grid points.
    """

    y = np.sin(2 * np.pi * x)  # what if include more periods in [0,1]
    return y


# %%
# Example usage:
num_points = 100  # Are there any sampling method that is more efficient?
x = np.linspace(0, 1, num_points)  # what if non-uniform grid?
y = sin_2pi_on_grid(x)

# Add white noise to y
noise_intensity = 0.4
noise = np.random.normal(0, noise_intensity, len(y))
y_noise = y + noise

# %%
import matplotlib.pyplot as plt

# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot y vs x on the left subplot
axs[0].plot(x, y, label='y = sin(2πx)')
axs[0].set_title('y vs x')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()

# Plot y_noise vs x on the right subplot
axs[1].plot(x, y_noise, label='y_noise = sin(2πx) + noise', color='orange')
axs[1].set_title('y_noise vs x')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y_noise')
axs[1].legend()

# Display the plots
plt.tight_layout()
plt.show()

# %%
import torch.nn as nn
import torch.optim as optim


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, hidden_units=32):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1, hidden_units)  # what if I used different initialization?
        self.hidden2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x


# %%
# Prepare the data
USE_NOISE = True
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
if USE_NOISE:
    y_tensor = torch.tensor(y_noise, dtype=torch.float32).view(-1, 1)
else:
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# %%
# Initialize the model, loss function, and optimizer
model = MLP(hidden_units=32)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Training loop
loss_history = []
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    predicted = model(x_tensor).numpy()

# %%
# Plot the true values and the predicted values
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='True y = sin(2πx)')
plt.plot(x, predicted, label='Predicted y', linestyle='--')
plt.title('True vs Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# %%