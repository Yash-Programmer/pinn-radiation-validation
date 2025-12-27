"""
KAVACH BL4S - Train Baseline Neural Network
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Change to script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.data_loader import PINNDataLoader
from src.baseline_nn import BaselineNN

# Set random seeds
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 60)
print("BASELINE NEURAL NETWORK TRAINING")
print("=" * 60)

# Load data
loader = PINNDataLoader()
X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data()

# Create model
baseline = BaselineNN(
    input_dim=5,
    hidden_layers=[64, 64, 64, 32],
    activation='relu',
    learning_rate=0.001
)

# Build and show architecture
baseline.build_model()
print("\nModel Architecture:")
baseline.get_summary()

# Train
print("\nStarting training...")
history = baseline.train(
    X_train, y_train,
    X_val, y_val,
    epochs=200,
    batch_size=32,
    verbose=1
)

# Evaluate
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

test_loss, test_mae = baseline.evaluate(X_test, y_test)
print(f"Test MSE: {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# Save model
baseline.save('models/baseline_nn.keras')

# Save history
np.save('models/baseline_history.npy', history.history)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training History - Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

ax2.plot(history.history['mae'], label='Train MAE')
ax2.plot(history.history['val_mae'], label='Val MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.set_title('Training History - MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/baseline_training_history.png', dpi=150)
print("\nTraining plot saved to plots/baseline_training_history.png")

print("\n" + "=" * 60)
print("BASELINE TRAINING COMPLETE!")
print("=" * 60)
