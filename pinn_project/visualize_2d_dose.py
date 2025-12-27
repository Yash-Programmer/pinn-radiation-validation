"""
2D Dose Distribution Visualization
KAVACH BL4S - 2D Dose Validation Framework

Creates publication-quality 2D dose maps and profile comparisons.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys

# Setup path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)


def plot_2d_dose_map(
    depth_grid: np.ndarray,
    radial_grid: np.ndarray,
    dose_grid: np.ndarray,
    energy: float,
    title: str = "2D Dose Distribution",
    save_path: str = None
):
    """
    Create 2D dose distribution heatmap.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: 2D heatmap
    ax1 = axes[0]
    im = ax1.pcolormesh(radial_grid, depth_grid, dose_grid, 
                        shading='auto', cmap='hot')
    ax1.set_xlabel('Radial Distance (mm)', fontsize=12)
    ax1.set_ylabel('Depth (mm)', fontsize=12)
    ax1.set_title(f'{title} - {energy} MeV', fontsize=14)
    ax1.invert_yaxis()  # Depth increases downward
    plt.colorbar(im, ax=ax1, label='Dose (a.u.)')
    
    # Panel 2: Depth and lateral profiles
    ax2 = axes[1]
    
    # Central axis depth profile (r=0)
    r0_idx = dose_grid.shape[1] // 2
    depth_profile = dose_grid[:, r0_idx]
    depths = depth_grid[:, 0]
    ax2.plot(depths, depth_profile / depth_profile.max(), 'b-', 
             linewidth=2, label='Depth Profile (r=0)')
    
    # Lateral profile at peak depth
    peak_depth_idx = np.argmax(depth_profile)
    lateral_profile = dose_grid[peak_depth_idx, :]
    radials = radial_grid[0, :]
    ax2.plot(radials, lateral_profile / lateral_profile.max(), 'r--',
             linewidth=2, label=f'Lateral Profile (z={depths[peak_depth_idx]:.1f}mm)')
    
    ax2.set_xlabel('Distance (mm)', fontsize=12)
    ax2.set_ylabel('Normalized Dose', fontsize=12)
    ax2.set_title('Dose Profiles', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def compare_2d_dose_maps(
    dose_maps: dict,
    depth_grid: np.ndarray,
    radial_grid: np.ndarray,
    save_path: str = None
):
    """
    Compare multiple 2D dose distributions (e.g., TOPAS vs PINN).
    """
    n_maps = len(dose_maps)
    fig, axes = plt.subplots(1, n_maps, figsize=(5*n_maps, 4))
    
    if n_maps == 1:
        axes = [axes]
    
    for ax, (name, dose) in zip(axes, dose_maps.items()):
        im = ax.pcolormesh(radial_grid, depth_grid, dose,
                          shading='auto', cmap='hot')
        ax.set_xlabel('Radial Distance (mm)', fontsize=10)
        ax.set_ylabel('Depth (mm)', fontsize=10)
        ax.set_title(name, fontsize=12)
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, label='Dose')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def create_demo_2d_dose_plots():
    """Create demonstration 2D dose plots using synthetic data."""
    
    from src.pinn_2d import create_synthetic_2d_data, PINN2D
    
    os.makedirs('plots', exist_ok=True)
    
    print("Creating 2D dose visualization demo...")
    print("=" * 50)
    
    # Generate synthetic data
    X, y = create_synthetic_2d_data(n_samples=5000)
    
    # Split and normalize
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    
    # Train 2D PINN
    print("Training 2D PINN...")
    model = PINN2D(
        input_dim=3,
        hidden_layers=[128, 128, 64, 32],
        lambda_smooth=0.1,
        lambda_positive=0.01,
        lambda_radial=0.05
    )
    
    model.train(
        X_train_norm, y_train_norm,
        X_val_norm, y_val_norm,
        epochs=50,
        verbose=0
    )
    
    # Generate 2D dose map for visualization
    energy = 250  # MeV
    n_depth, n_radial = 50, 30
    
    depth_range = (0, 20)
    radial_range = (0, 25)
    
    depths = np.linspace(*depth_range, n_depth)
    radials = np.linspace(*radial_range, n_radial)
    depth_grid, radial_grid = np.meshgrid(depths, radials, indexing='ij')
    
    # Create input for prediction
    X_grid = np.zeros((n_depth * n_radial, 3))
    X_grid[:, 0] = energy
    X_grid[:, 1] = depth_grid.flatten()
    X_grid[:, 2] = radial_grid.flatten()
    
    # Normalize
    X_grid_norm = (X_grid - X_mean) / X_std
    
    # Predict
    dose_pred_norm = model.predict(X_grid_norm)
    dose_pred = dose_pred_norm * y_std + y_mean
    dose_grid = dose_pred.reshape(n_depth, n_radial)
    dose_grid = np.maximum(dose_grid, 0)  # Ensure non-negative
    
    # Create plot
    print("Generating 2D dose map...")
    plot_2d_dose_map(
        depth_grid, radial_grid, dose_grid,
        energy=energy,
        title="PINN 2D Dose Prediction",
        save_path="plots/dose_2d_map.png"
    )
    
    # Multi-energy comparison
    energies = [70, 250, 1000]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, E in zip(axes, energies):
        X_grid[:, 0] = E
        X_grid_norm = (X_grid - X_mean) / X_std
        dose_pred_norm = model.predict(X_grid_norm)
        dose_pred = dose_pred_norm * y_std + y_mean
        dose_grid = dose_pred.reshape(n_depth, n_radial)
        dose_grid = np.maximum(dose_grid, 0)
        
        im = ax.pcolormesh(radial_grid, depth_grid, dose_grid,
                          shading='auto', cmap='hot')
        ax.set_xlabel('Radial Distance (mm)')
        ax.set_ylabel('Depth (mm)')
        ax.set_title(f'{E} MeV')
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, label='Dose')
    
    plt.suptitle('2D Dose Distribution: Energy Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/dose_2d_energy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("=" * 50)
    print("Saved:")
    print("  - plots/dose_2d_map.png")
    print("  - plots/dose_2d_energy_comparison.png")
    print("\n✓ 2D dose visualization complete!")


if __name__ == "__main__":
    create_demo_2d_dose_plots()
