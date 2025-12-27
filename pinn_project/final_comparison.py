"""
KAVACH BL4S - Final Model Comparison
Compare Baseline NN vs Simple PINN vs Advanced PINN
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.data_loader import PINNDataLoader
from tensorflow import keras

os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

# Load data
loader = PINNDataLoader()
X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data()
test_df = pd.read_csv('../pinn_data/test_split_v2.csv')

# Load all models
print("\nLoading models...")
try:
    baseline = keras.models.load_model('models/baseline_nn.keras')
    print("  ✓ Baseline NN loaded")
except:
    print("  ✗ Baseline NN not found - run train_baseline.py first")
    baseline = None

try:
    simple_pinn = keras.models.load_model('models/simple_pinn.keras')
    print("  ✓ Simple PINN loaded")
except:
    print("  ✗ Simple PINN not found - run train_simple_pinn.py first")
    simple_pinn = None

try:
    advanced_pinn = keras.models.load_model('models/advanced_pinn.keras')
    print("  ✓ Advanced PINN loaded")
except:
    print("  ✗ Advanced PINN not found - run train_advanced_pinn.py first")
    advanced_pinn = None

# Check if models exist
models = {'Baseline NN': baseline, 'Simple PINN': simple_pinn, 'Advanced PINN': advanced_pinn}
models = {k: v for k, v in models.items() if v is not None}

if len(models) == 0:
    print("\nNo trained models found! Run training scripts first.")
    sys.exit(1)

# Make predictions
predictions = {}
for name, model in models.items():
    y_pred_scaled = model.predict(X_test, verbose=0)
    predictions[name] = loader.inverse_transform_target(y_pred_scaled)

y_true = loader.inverse_transform_target(y_test)

# Calculate metrics
print("\n" + "=" * 60)
print("TEST SET PERFORMANCE")
print("=" * 60)
print(f"{'Model':<20} {'MSE':<15} {'MAE':<15} {'MAPE (%)':<10}")
print("-" * 60)

results = []
for name, y_pred in predictions.items():
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-15))) * 100
    
    print(f"{name:<20} {mse:<15.2e} {mae:<15.2e} {mape:<10.2f}")
    results.append({'Model': name, 'MSE': mse, 'MAE': mae, 'MAPE (%)': mape})

print("=" * 60)

# Calculate improvements
if len(results) > 1:
    baseline_mse = results[0]['MSE']
    print("\nIMPROVEMENT OVER BASELINE:")
    for r in results[1:]:
        improvement = (baseline_mse - r['MSE']) / baseline_mse * 100
        print(f"  {r['Model']}: {improvement:+.2f}%")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('results/final_comparison.csv', index=False)
print(f"\nResults saved to results/final_comparison.csv")

# Create comparison figure
test_energies = sorted(test_df['energy_MeV'].unique())
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = {'Baseline NN': 'blue', 'Simple PINN': 'green', 'Advanced PINN': 'red'}

for i, energy in enumerate(test_energies):
    ax = axes[i]
    
    mask = test_df['energy_MeV'] == energy
    depths = test_df.loc[mask, 'depth_mm'].values
    
    true_dose = y_true[test_df['energy_MeV'] == energy]
    ax.plot(depths, true_dose, 'k-', linewidth=3, label='TOPAS (Truth)', alpha=0.9)
    
    for name, y_pred in predictions.items():
        pred_dose = y_pred[test_df['energy_MeV'] == energy]
        ax.plot(depths, pred_dose, '--', linewidth=2, label=name, 
                color=colors.get(name, 'gray'), alpha=0.7)
    
    ax.set_xlabel('Depth (mm)', fontsize=12)
    ax.set_ylabel('Dose (Gy)', fontsize=12)
    ax.set_title(f'{energy:.0f} MeV (Unseen Test Energy)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/final_model_comparison.png', dpi=300)
print(f"Comparison plot saved to plots/final_model_comparison.png")

print("\n" + "=" * 60)
print("COMPARISON COMPLETE!")
print("=" * 60)
