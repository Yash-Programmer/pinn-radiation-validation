"""
Hyperparameter Optimization for Physics-Informed Neural Networks
KAVACH BL4S - Validation Framework

Uses grid search to systematically find optimal λ values
for physics constraints. Test set withheld from all optimization.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import tensorflow as tf
from src.data_loader import PINNDataLoader
from src.advanced_pinn import AdvancedPINN

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameter search space (logarithmic grid)
LAMBDA_SMOOTH_VALUES = [0.01, 0.05, 0.1, 0.5]
LAMBDA_POS_VALUES = [0.001, 0.005, 0.01, 0.05]
LAMBDA_GRAD_VALUES = [0.01, 0.05, 0.1, 0.5]


def run_hyperparameter_search():
    """Run systematic hyperparameter optimization."""
    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION: Physics Constraint Weights")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    print("Loading data...")
    data_loader = PINNDataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_data()
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print()
    
    # Generate all combinations
    combinations = list(product(LAMBDA_SMOOTH_VALUES, LAMBDA_POS_VALUES, LAMBDA_GRAD_VALUES))
    print(f"Total configurations to test: {len(combinations)}")
    print()
    
    results = []
    best_val_loss = float('inf')
    best_config = None
    
    for i, (smooth, pos, grad) in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] λ_s={smooth}, λ_p={pos}, λ_g={grad}", end=" ")
        
        # Create and train model
        model = AdvancedPINN(
            input_dim=5,
            lambda_smooth=smooth,
            lambda_positive=pos,
            lambda_gradient=grad
        )
        
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=50,  # Reduced for speed
            verbose=0
        )
        
        # Get final validation loss
        final_val_loss = history['val_loss'][-1]
        
        # Evaluate on test set (only for reporting, not selection)
        y_pred = model.predict(X_test)
        test_mse = np.mean((y_test - y_pred) ** 2)
        
        result = {
            'lambda_smooth': smooth,
            'lambda_pos': pos,
            'lambda_grad': grad,
            'val_loss': final_val_loss,
            'test_mse': test_mse,
            'epochs': len(history['loss']),
            'final_smooth_loss': history['smooth_loss'][-1],
            'final_pos_loss': history['positive_loss'][-1]
        }
        results.append(result)
        
        print(f"→ val_loss={final_val_loss:.6f}")
        
        # Track best
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_config = result
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/hyperparameter_search.csv', index=False)
    
    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print()
    
    print("Best configuration (by validation loss):")
    print(f"  λ_smooth = {best_config['lambda_smooth']}")
    print(f"  λ_pos    = {best_config['lambda_pos']}")
    print(f"  λ_grad   = {best_config['lambda_grad']}")
    print(f"  Val Loss = {best_config['val_loss']:.6f}")
    print(f"  Test MSE = {best_config['test_mse']:.6f}")
    
    # Sensitivity analysis
    print()
    print("Sensitivity Analysis:")
    print("-" * 50)
    
    # Effect of each parameter
    for param in ['lambda_smooth', 'lambda_pos', 'lambda_grad']:
        grouped = df.groupby(param)['val_loss'].mean()
        best_val = grouped.idxmin()
        print(f"  {param}: best value = {best_val} (mean val_loss = {grouped[best_val]:.6f})")
    
    # Create visualization
    create_hyperparameter_plots(df)
    
    print()
    print(f"Results saved to: results/hyperparameter_search.csv")
    print(f"Plot saved to: plots/hyperparameter_sensitivity.png")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df, best_config


def create_hyperparameter_plots(df):
    """Create sensitivity analysis visualizations."""
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    params = ['lambda_smooth', 'lambda_pos', 'lambda_grad']
    titles = ['Smoothness (λ_smooth)', 'Positivity (λ_pos)', 'Gradient (λ_grad)']
    
    for ax, param, title in zip(axes, params, titles):
        # Group by this parameter, average over others
        grouped = df.groupby(param)['val_loss'].agg(['mean', 'std'])
        
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   fmt='o-', capsize=5, markersize=8)
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title(f'Sensitivity to {title}', fontsize=13)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/hyperparameter_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    results, best = run_hyperparameter_search()
