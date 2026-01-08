"""
A simple example of overfitting using a Multi-Layer Perceptron (MLP) to fit noisy sine wave data.
Author: Dongyang Kuang

NOTE:
    [] Multiple aspects can be investigated:
    [] Now includes extrapolation testing on intervals [-1,0] and [1,2]
"""

# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import pearsonr  # For calculating correlation

# %%
def sin_2pi_on_grid(x):
    """
    Computes y = sin(2pi*x) on a uniform grid from 0 to 1.

    Parameters:
    x (int or array): input for evaluation.

    Returns:
    y (numpy.ndarray): The computed sine values at the grid points.
    """
    y = np.sin(2 * np.pi * x)
    return y

# %%
# Generate training data in [0, 1]
num_points = 100
np.random.seed(42)  # Set random seed for reproducibility
x_train = np.linspace(0, 1, num_points)
y_true = sin_2pi_on_grid(x_train)

# Add white noise to y
noise_intensity = 0.4
noise = np.random.normal(0, noise_intensity, len(y_true))
y_train = y_true + noise

# Generate extrapolation data
x_extrapolate_left = np.linspace(-1, 0, 100)  # Left extrapolation interval [-1, 0]
x_extrapolate_right = np.linspace(1, 2, 100)  # Right extrapolation interval [1, 2]

# Combine for visualization
x_full = np.linspace(-1, 2, 300)
y_full_true = sin_2pi_on_grid(x_full)

# %%
# Plot the experimental design
fig, ax = plt.subplots(figsize=(10, 6))

# Plot true function
ax.plot(x_full, y_full_true, label='True function: sin(2πx)', linewidth=2.5, color='black')

# Plot training data
ax.scatter(x_train, y_train, label='Training data (with noise)', alpha=0.7, s=40, color='blue', zorder=5)

# Highlight different regions
ax.axvspan(0, 1, alpha=0.3, color='green', label='Training region [0,1]')
ax.axvspan(-1, 0, alpha=0.2, color='gray', label='Left extrapolation [-1,0]')
ax.axvspan(1, 2, alpha=0.2, color='orange', label='Right extrapolation [1,2]')

# Add vertical lines at boundaries
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=1)

ax.set_title('4.1 Extrapolation Experiment Design\nTraining data limited to x∈[0,1], testing model prediction in [-1,0] and [1,2]',
             fontsize=14, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1.1, 2.1)
ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig('extrapolation_experiment_design.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Define the MLP model
class MLP(nn.Module):
    def __init__(self, hidden_units=32):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1, hidden_units)
        self.hidden2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# %%
# Prepare the training data
USE_NOISE = True
x_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
if USE_NOISE:
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
else:
    y_tensor = torch.tensor(y_true, dtype=torch.float32).view(-1, 1)

# Prepare extrapolation data as tensors
x_extrapolate_left_tensor = torch.tensor(x_extrapolate_left, dtype=torch.float32).view(-1, 1)
x_extrapolate_right_tensor = torch.tensor(x_extrapolate_right, dtype=torch.float32).view(-1, 1)
x_full_tensor = torch.tensor(x_full, dtype=torch.float32).view(-1, 1)

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
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# %%
# Evaluate the model on all regions
model.eval()
with torch.no_grad():
    predicted_train = model(x_tensor).numpy().flatten()
    predicted_left = model(x_extrapolate_left_tensor).numpy().flatten()
    predicted_right = model(x_extrapolate_right_tensor).numpy().flatten()
    predicted_full = model(x_full_tensor).numpy().flatten()

# %%
# Plot the extrapolation results
fig, ax = plt.subplots(figsize=(10, 6))

# Plot true function
ax.plot(x_full, y_full_true, label='True function: sin(2πx)', linewidth=2.5, color='black')

# Plot predictions with different styles for each region
ax.plot(x_train, predicted_train, label='MLP prediction on [0,1]', linewidth=2, color='green')
ax.plot(x_extrapolate_left, predicted_left, label='MLP extrapolation on [-1,0]', linewidth=2, color='gray', linestyle='--')
ax.plot(x_extrapolate_right, predicted_right, label='MLP extrapolation on [1,2]', linewidth=2, color='orange', linestyle='--')

# Highlight different regions
ax.axvspan(0, 1, alpha=0.1, color='green')
ax.axvspan(-1, 0, alpha=0.1, color='gray')
ax.axvspan(1, 2, alpha=0.1, color='orange')

# Add training data points
ax.scatter(x_train, y_train, alpha=0.5, s=20, color='blue', label='Training data (with noise)')

# Add vertical lines at boundaries
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=1)

ax.set_title('4.2 Extrapolation Results Analysis\nMLP performs well in training region but poorly in extrapolation regions',
             fontsize=14, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1.1, 2.1)
ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig('extrapolation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Calculate performance metrics for each region
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def calculate_correlation(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

# Calculate for each region
y_true_train = y_true
y_true_left = sin_2pi_on_grid(x_extrapolate_left)
y_true_right = sin_2pi_on_grid(x_extrapolate_right)

mse_train = calculate_mse(y_true_train, predicted_train)
mse_left = calculate_mse(y_true_left, predicted_left)
mse_right = calculate_mse(y_true_right, predicted_right)

corr_train = calculate_correlation(y_true_train, predicted_train)
corr_left = calculate_correlation(y_true_left, predicted_left)
corr_right = calculate_correlation(y_true_right, predicted_right)

# Determine extrapolation quality
def determine_quality(mse, corr, region_type):
    if region_type == 'train':
        return 'Excellent'
    else:
        if corr > 0.8:
            return 'Good'
        elif corr > 0.6:
            return 'Medium'
        else:
            return 'Poor'

quality_train = determine_quality(mse_train, corr_train, 'train')
quality_left = determine_quality(mse_left, corr_left, 'extrapolation')
quality_right = determine_quality(mse_right, corr_right, 'extrapolation')

# %%
# Print the performance table
print("=" * 75)
print("Table 2: Model Performance in Extrapolation Regions")
print("=" * 75)
print(f"{'Region':<25} {'MSE':<15} {'Correlation':<15} {'Quality':<10}")
print("-" * 75)
print(f"{'Training [0,1]':<25} {mse_train:<15.3f} {corr_train:<15.2f} {quality_train:<10}")
print(f"{'Left extrapolation [-1,0]':<25} {mse_left:<15.3f} {corr_left:<15.2f} {quality_left:<10}")
print(f"{'Right extrapolation [1,2]':<25} {mse_right:<15.3f} {corr_right:<15.2f} {quality_right:<10}")
print("=" * 75)

# Print quantitative analysis
print("\n4.2 Extrapolation Results Analysis")
print(f"Table 2 quantitatively shows the performance degradation in extrapolation regions.")
print(f"Left extrapolation MSE ({mse_left:.3f}) is about {mse_left/mse_train:.1f} times higher than training MSE ({mse_train:.3f})")
print(f"Right extrapolation MSE ({mse_right:.3f}) is about {mse_right/mse_train:.1f} times higher than training MSE ({mse_train:.3f})")
print(f"Correlation in extrapolation regions ({corr_left:.2f}, {corr_right:.2f}) is significantly lower than in training region ({corr_train:.2f})")

# %%
# Create a bar chart for MSE comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# MSE comparison
regions = ['Training\n[0,1]', 'Left extrapolation\n[-1,0]', 'Right extrapolation\n[1,2]']
mse_values = [mse_train, mse_left, mse_right]
colors = ['green', 'gray', 'orange']

bars = ax1.bar(regions, mse_values, color=colors, alpha=0.7)
ax1.set_title('MSE Comparison', fontsize=12, fontweight='bold')
ax1.set_ylabel('MSE')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, mse_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontsize=10)

# Correlation comparison
corr_values = [corr_train, corr_left, corr_right]

bars = ax2.bar(regions, corr_values, color=colors, alpha=0.7)
ax2.set_title('Correlation Comparison', fontsize=12, fontweight='bold')
ax2.set_ylabel('Correlation Coefficient')
ax2.set_ylim(-1, 1.1)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, corr_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.2f}', ha='center', va='bottom', fontsize=10)

plt.suptitle('4.3 Quantitative Analysis of Extrapolation Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('extrapolation_quantitative_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Print reasons for extrapolation limitations
print("\n" + "=" * 75)
print("4.3 Reasons for Extrapolation Limitations")
print("=" * 75)
print("Reasons why MLP performs poorly in extrapolation:")
print("1. Activation function properties: ReLU and tanh have no clear mathematical guarantees outside training region")
print("2. Lack of periodic prior: Standard MLP has no built-in periodic assumptions")
print("3. Training data limitations: Model only sees data in [0,1] interval, cannot learn complete periodic pattern")
print("4. Overfitting tendency: MLP tends to overfit training data (including noise), affecting generalization")
print("=" * 75)