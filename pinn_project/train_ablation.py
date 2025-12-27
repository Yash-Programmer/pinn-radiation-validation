"""
Ablation Study: Evaluating Physics Constraint Contributions
KAVACH BL4S - Validation Framework

Tests 7 configurations to determine which physics constraints matter most:
1. Baseline (no constraints)
2. Smoothness only
3. Positivity only  
4. Gradient only
5. Smoothness + Positivity
6. Smoothness + Gradient
7. Full PINN (all three)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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

# Configuration space for ablation
ABLATION_CONFIGS = [
    {'name': 'baseline', 'smooth': 0.0, 'pos': 0.0, 'grad': 0.0, 'description': 'No physics constraints'},
    {'name': 'smooth_only', 'smooth': 0.1, 'pos': 0.0, 'grad': 0.0, 'description': 'Smoothness constraint only'},
    {'name': 'pos_only', 'smooth': 0.0, 'pos': 0.01, 'grad': 0.0, 'description': 'Positivity constraint only'},
    {'name': 'grad_only', 'smooth': 0.0, 'pos': 0.0, 'grad': 0.05, 'description': 'Gradient constraint only'},
    {'name': 'smooth_pos', 'smooth': 0.1, 'pos': 0.01, 'grad': 0.0, 'description': 'Smoothness + Positivity'},
    {'name': 'smooth_grad', 'smooth': 0.1, 'pos': 0.0, 'grad': 0.05, 'description': 'Smoothness + Gradient'},
    {'name': 'full_pinn', 'smooth': 0.1, 'pos': 0.01, 'grad': 0.05, 'description': 'All three constraints'},
]


def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics."""
    y_pred = model.predict(X_test)
    
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    
    return {'mse': mse, 'mae': mae, 'mape': mape}


def run_ablation_study():
    """Run complete ablation study."""
    print("=" * 60)
    print("ABLATION STUDY: Physics Constraint Contributions")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    print("Loading data...")
    data_loader = PINNDataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_data()
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print()
    
    # Run each configuration
    results = []
    
    for config in ABLATION_CONFIGS:
        print("-" * 60)
        print(f"Configuration: {config['name']}")
        print(f"  λ_smooth = {config['smooth']}, λ_pos = {config['pos']}, λ_grad = {config['grad']}")
        print(f"  {config['description']}")
        
        # Create and train model
        model = AdvancedPINN(
            input_dim=5,
            lambda_smooth=config['smooth'],
            lambda_positive=config['pos'],
            lambda_gradient=config['grad']
        )
        
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=30,
            verbose=0
        )
        
        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test)
        
        # Record results
        result = {
            'config': config['name'],
            'description': config['description'],
            'lambda_smooth': config['smooth'],
            'lambda_pos': config['pos'],
            'lambda_grad': config['grad'],
            'test_mse': metrics['mse'],
            'test_mae': metrics['mae'],
            'test_mape': metrics['mape'],
            'epochs': len(history['loss']),
            'final_loss': history['loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
        results.append(result)
        
        print(f"  Test MSE: {metrics['mse']:.6f}")
        print(f"  Test MAPE: {metrics['mape']:.2f}%")
        print(f"  Epochs: {len(history['loss'])}")
        print()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate improvements relative to baseline
    baseline_mse = df[df['config'] == 'baseline']['test_mse'].values[0]
    df['improvement_pct'] = (baseline_mse - df['test_mse']) / baseline_mse * 100
    
    # Save results
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/ablation_study.csv', index=False)
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()
    
    # Print summary table
    print(f"{'Configuration':<20} {'MSE':<12} {'MAPE %':<10} {'Improvement':<12}")
    print("-" * 54)
    for _, row in df.iterrows():
        sign = '+' if row['improvement_pct'] > 0 else ''
        print(f"{row['config']:<20} {row['test_mse']:.6f}     {row['test_mape']:.2f}%     {sign}{row['improvement_pct']:.1f}%")
    
    print()
    print("-" * 54)
    
    # Key findings
    best_config = df.loc[df['test_mse'].idxmin()]
    print(f"\nBest configuration: {best_config['config']}")
    print(f"  MSE: {best_config['test_mse']:.6f}")
    print(f"  Improvement over baseline: +{best_config['improvement_pct']:.1f}%")
    
    # Individual constraint contributions
    smooth_contrib = df[df['config'] == 'smooth_only']['improvement_pct'].values[0]
    pos_contrib = df[df['config'] == 'pos_only']['improvement_pct'].values[0]
    grad_contrib = df[df['config'] == 'grad_only']['improvement_pct'].values[0]
    
    print(f"\nIndividual constraint contributions:")
    print(f"  Smoothness: +{smooth_contrib:.1f}%")
    print(f"  Positivity: +{pos_contrib:.1f}%")
    print(f"  Gradient:   +{grad_contrib:.1f}%")
    
    # Create visualization
    create_ablation_plot(df)
    
    print(f"\nResults saved to: results/ablation_study.csv")
    print(f"Plot saved to: plots/ablation_study.png")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df


def create_ablation_plot(df):
    """Create visualization of ablation results."""
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sort by MSE for visualization
    df_sorted = df.sort_values('test_mse', ascending=False)
    
    # Plot 1: MSE comparison (bar chart)
    ax1 = axes[0]
    colors = ['#e74c3c' if c == 'baseline' else '#3498db' if c == 'full_pinn' else '#95a5a6' 
              for c in df_sorted['config']]
    bars = ax1.barh(df_sorted['config'], df_sorted['test_mse'], color=colors)
    ax1.set_xlabel('Test MSE', fontsize=12)
    ax1.set_title('MSE by Configuration (Lower is Better)', fontsize=14)
    ax1.axvline(x=df_sorted[df_sorted['config'] == 'baseline']['test_mse'].values[0], 
                color='red', linestyle='--', alpha=0.7, label='Baseline')
    ax1.legend()
    
    # Add value labels
    for bar, val in zip(bars, df_sorted['test_mse']):
        ax1.text(val, bar.get_y() + bar.get_height()/2, f' {val:.5f}', 
                 va='center', fontsize=9)
    
    # Plot 2: Improvement percentage
    ax2 = axes[1]
    df_sorted2 = df.sort_values('improvement_pct', ascending=True)
    colors2 = ['#27ae60' if x > 0 else '#e74c3c' for x in df_sorted2['improvement_pct']]
    bars2 = ax2.barh(df_sorted2['config'], df_sorted2['improvement_pct'], color=colors2)
    ax2.set_xlabel('Improvement over Baseline (%)', fontsize=12)
    ax2.set_title('Improvement by Configuration', fontsize=14)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, df_sorted2['improvement_pct']):
        sign = '+' if val > 0 else ''
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{sign}{val:.1f}%', 
                 va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/ablation_study.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    results = run_ablation_study()
