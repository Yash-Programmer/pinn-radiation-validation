"""
KAVACH BL4S - Train Advanced PINN with Multiple Physics Constraints
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.data_loader import PINNDataLoader
from src.advanced_pinn import AdvancedPINN

np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("ADVANCED PINN TRAINING (Multiple Physics Constraints)")
print("=" * 60)

# Load data with physics reference
loader = PINNDataLoader()
(X_train, y_train, physics_train,
 X_val, y_val, physics_val,
 X_test, y_test) = loader.load_with_physics()

# Create Advanced PINN
pinn = AdvancedPINN(
    input_dim=5,
    hidden_layers=[64, 64, 64, 32],
    activation='tanh',
    learning_rate=0.001,
    lambda_smooth=0.1,
    lambda_positive=0.01,
    lambda_gradient=0.05
)

pinn.build_model()
print("\nModel Architecture:")
pinn.get_summary()

print("\nStarting Advanced PINN training...")
print("Physics constraints: Smoothness + Positivity + Gradient Correlation\n")

history = pinn.train(
    X_train, y_train,
    X_val, y_val,
    physics_train=physics_train,
    epochs=200,
    batch_size=32,
    verbose=1
)

# Evaluate
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

test_mse, test_mae = pinn.evaluate(X_test, y_test)
print(f"Test MSE: {test_mse:.6f}")
print(f"Test MAE: {test_mae:.6f}")

pinn.save('models/advanced_pinn.keras')
np.save('models/advanced_pinn_history.npy', history)

# Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(history['loss'], 'b-', alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Total Loss')
axes[0, 0].set_title('Total Loss')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

axes[0, 1].plot(history['data_loss'], label='Train', alpha=0.7)
axes[0, 1].plot(history['val_loss'], label='Val', alpha=0.7)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Data Loss')
axes[0, 1].set_title('Data Loss: Train vs Val')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

axes[0, 2].plot(history['smooth_loss'], 'g-', alpha=0.7)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Smoothness Loss')
axes[0, 2].set_title('Physics: Smoothness')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_yscale('log')

axes[1, 0].plot(history['positive_loss'], 'r-', alpha=0.7)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Positivity Loss')
axes[1, 0].set_title('Physics: Positivity')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

axes[1, 1].plot(history['gradient_loss'], 'm-', alpha=0.7)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Gradient Loss')
axes[1, 1].set_title('Physics: Gradient Correlation')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yscale('log')

axes[1, 2].plot(history['data_loss'], label='Data', alpha=0.7)
axes[1, 2].plot(history['smooth_loss'], label='Smooth', alpha=0.7)
axes[1, 2].plot(history['gradient_loss'], label='Gradient', alpha=0.7)
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Loss')
axes[1, 2].set_title('All Loss Components')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_yscale('log')

plt.tight_layout()
plt.savefig('plots/advanced_pinn_training_history.png', dpi=150)
print("\nTraining plot saved to plots/advanced_pinn_training_history.png")

print("\n" + "=" * 60)
print("ADVANCED PINN TRAINING COMPLETE!")
print("=" * 60)
