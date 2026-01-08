"""
A simple example of overfitting using a Multi-Layer Perceptron (MLP) to fit noisy sine wave data.
Author: Dongyang Kuang

NOTE:
    [] Multiple aspects can be investigated:
    [] Exploring effects of different noise types on MLP training and prediction
"""

# %%
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


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
# Define the 4 specified noise types
def add_gaussian_noise_sigma02(y):
    """Add Gaussian noise with σ=0.2"""
    noise = np.random.normal(0, 0.2, len(y))
    return y + noise


def add_gaussian_noise_sigma04(y):
    """Add Gaussian noise with σ=0.4"""
    noise = np.random.normal(0, 0.4, len(y))
    return y + noise


def add_uniform_noise(y):
    """Add uniform noise [-0.4, 0.4] to match σ=0.4 intensity"""
    # For uniform distribution U[-a,a], variance = a²/3
    # To get similar variance as σ=0.4, we set a = 0.4*sqrt(3) ≈ 0.6928
    intensity = 0.4 * np.sqrt(3)
    noise = np.random.uniform(-intensity, intensity, len(y))
    return y + noise


def add_impulse_noise(y, probability=0.1):
    """Add impulse noise (10%) with intensity ±0.8"""
    y_noise = y.copy()
    mask = np.random.random(len(y)) < probability
    impulse = np.random.choice([-1, 1], size=np.sum(mask)) * 0.8
    y_noise[mask] += impulse
    return y_noise


# %%
# Generate clean data
num_points = 100
x = np.linspace(0, 1, num_points)
y_clean = sin_2pi_on_grid(x)

# Add different types of noise
np.random.seed(42)  # For reproducibility
noise_types = {
    'Gaussian (σ=0.2)': add_gaussian_noise_sigma02(y_clean),
    'Gaussian (σ=0.4)': add_gaussian_noise_sigma04(y_clean),
    'Uniform': add_uniform_noise(y_clean),
    'Impulse (10%)': add_impulse_noise(y_clean, probability=0.1)
}

# %%
# 1. 噪声类型可视化
print("Generating Figure 1: Noise Types Visualization...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (name, y_noisy) in enumerate(noise_types.items()):
    ax = axes[idx]
    ax.plot(x, y_clean, 'b-', label='Clean', alpha=0.7, linewidth=2)
    ax.plot(x, y_noisy, 'r-', label=name, alpha=0.7)
    ax.set_title(f'{name}', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noise_types_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# %%
# Define MLP model
class MLP(nn.Module):
    def __init__(self, hidden_units=32):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1, hidden_units)
        self.hidden2 = nn.Linear(hidden_units, hidden_units)
        self.hidden3 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, 1)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.dropout(x)
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x


# %%
# Train MLP on different noise types
def train_mlp_on_noise(x_tensor, y_tensor, noise_name, num_epochs=1000, hidden_units=32):
    """Train MLP on specific noise type and return results"""
    print(f"\nTraining on {noise_name}...")

    model = MLP(hidden_units=hidden_units)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    loss_history = []
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        loss_history.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate
    model.eval()
    with torch.no_grad():
        predicted = model(x_tensor).numpy()

    return model, loss_history, predicted, best_loss


# %%
# Prepare data for all noise types
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
results = {}

for noise_name, y_noisy in noise_types.items():
    y_tensor = torch.tensor(y_noisy, dtype=torch.float32).view(-1, 1)

    model, loss_history, predicted, final_loss = train_mlp_on_noise(
        x_tensor, y_tensor, noise_name, num_epochs=1000, hidden_units=32
    )

    results[noise_name] = {
        'y_noisy': y_noisy,
        'predicted': predicted.flatten(),
        'loss_history': loss_history,
        'final_loss': final_loss,
        'model': model
    }

# %%
# 2. 训练损失对比
print("\nGenerating Figure 2: Training Loss Comparison...")
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot loss histories
ax1 = axes[0]
colors = ['blue', 'green', 'red', 'purple']
for idx, (noise_name, result) in enumerate(results.items()):
    ax1.plot(result['loss_history'], label=f'{noise_name} (final: {result["final_loss"]:.4f})',
             color=colors[idx], alpha=0.8)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss History for Different Noise Types')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')  # Log scale for better visualization

# Plot final losses comparison
ax2 = axes[1]
noise_names = list(results.keys())
final_losses = [results[name]['final_loss'] for name in noise_names]

bars = ax2.bar(noise_names, final_losses, color=colors[:len(noise_names)], alpha=0.7)
ax2.set_xlabel('Noise Type')
ax2.set_ylabel('Final Loss')
ax2.set_title('Final Training Loss Comparison')
ax2.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, loss in zip(bars, final_losses):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.0001,
             f'{loss:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('training_loss_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# 3. 预测结果对比
print("\nGenerating Figure 3: Prediction Results Comparison...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (noise_name, result) in enumerate(results.items()):
    ax = axes[idx]
    y_noisy = result['y_noisy']
    predicted = result['predicted']

    # Calculate metrics
    mse = np.mean((y_clean - predicted) ** 2)
    mae = np.mean(np.abs(y_clean - predicted))

    ax.plot(x, y_clean, 'b-', label='True', alpha=0.7, linewidth=2)
    ax.plot(x, y_noisy, 'g-', label='Noisy', alpha=0.3, linewidth=1)
    ax.plot(x, predicted, 'r--', label='Predicted', alpha=0.9, linewidth=2)

    ax.set_title(f'{noise_name}\nMSE: {mse:.4f}, MAE: {mae:.4f}', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Fill area between true and predicted
    ax.fill_between(x, y_clean, predicted, alpha=0.1, color='red')

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# 4. 误差分布分析
print("\nGenerating Figure 4: Error Distribution Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (noise_name, result) in enumerate(results.items()):
    ax = axes[idx]
    predicted = result['predicted']
    errors = y_clean - predicted

    # Histogram of errors
    ax.hist(errors, bins=25, alpha=0.7, color=colors[idx], edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_title(f'{noise_name} Error Distribution', fontsize=11)
    ax.set_xlabel('Error (True - Predicted)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: {np.mean(errors):.4f}\nStd: {np.std(errors):.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('error_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# 打印性能指标总结
print("\n" + "=" * 60)
print("OVERALL PERFORMANCE METRICS COMPARISON")
print("=" * 60)

print(f"{'Noise Type':<20} {'MSE':<10} {'MAE':<10} {'Max Error':<12} {'Std Error':<10} {'Train Loss':<10}")
print("-" * 72)

for noise_name, result in results.items():
    predicted = result['predicted']
    errors = y_clean - predicted

    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))
    std_error = np.std(errors)

    print(f"{noise_name:<20} {mse:<10.6f} {mae:<10.6f} "
          f"{max_error:<12.6f} {std_error:<10.6f} {result['final_loss']:<10.6f}")

print("\nAnalysis complete! Check saved images:")
print("1. noise_types_comparison.png - Noise types visualization")
print("2. training_loss_comparison.png - Training loss comparison")
print("3. predictions_comparison.png - Prediction results comparison")
print("4. error_distributions.png - Error distribution analysis")