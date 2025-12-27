#!/usr/bin/env python3
"""
KAVACH BL4S - PINN Data Validation Script
Runs all critical quality checks before neural network training.

Checks:
1. Missing values (NaN/NULL)
2. Duplicates
3. Physical data ranges
4. Row counts per energy
5. Bethe-Bloch agreement
6. Energy conservation
7. Range-energy monotonicity
8. Train/val/test split integrity
9. Feature scaling (creates scalers)
10. Visualization plots
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import optional dependencies
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Configuration
BASE_DIR = Path(__file__).parent
PINN_DIR = BASE_DIR / 'pinn_data'
PLOTS_DIR = PINN_DIR / 'plots'

# Physical constants
SCINT_THICKNESS = 10.0  # mm
N_BINS = 100
N_HISTORIES = 10000
EXPECTED_ENERGIES = [70, 100, 150, 200, 250, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000]

# Create plots directory
PLOTS_DIR.mkdir(exist_ok=True)


class ValidationResult:
    """Store validation results."""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, check, message=""):
        self.passed.append(f"✓ {check}: {message}" if message else f"✓ {check}")
    
    def add_fail(self, check, message=""):
        self.failed.append(f"✗ {check}: {message}" if message else f"✗ {check}")
    
    def add_warning(self, check, message=""):
        self.warnings.append(f"⚠ {check}: {message}" if message else f"⚠ {check}")
    
    def summary(self):
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"\nPassed: {len(self.passed)}")
        for p in self.passed:
            print(f"  {p}")
        
        if self.warnings:
            print(f"\nWarnings: {len(self.warnings)}")
            for w in self.warnings:
                print(f"  {w}")
        
        if self.failed:
            print(f"\nFailed: {len(self.failed)}")
            for f in self.failed:
                print(f"  {f}")
        
        print("\n" + "=" * 60)
        if self.failed:
            print("STATUS: ❌ VALIDATION FAILED - Fix issues before training")
        elif self.warnings:
            print("STATUS: ⚠️ PASSED WITH WARNINGS - Review before training")
        else:
            print("STATUS: ✅ ALL CHECKS PASSED - Ready for PINN training!")
        print("=" * 60)
        
        return len(self.failed) == 0


def check_missing_values(df, results):
    """Check for NaN/NULL values."""
    print("\n[1/10] Checking for missing values...")
    
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls == 0:
        results.add_pass("Missing values", "No NaN/NULL found")
    else:
        results.add_fail("Missing values", f"Found {total_nulls} null values")
        print(null_counts[null_counts > 0])


def check_duplicates(df, results):
    """Check for duplicate rows."""
    print("\n[2/10] Checking for duplicates...")
    
    duplicates = df.duplicated().sum()
    
    if duplicates == 0:
        results.add_pass("Duplicates", "No duplicate rows found")
    else:
        results.add_warning("Duplicates", f"Found {duplicates} duplicate rows")


def check_physical_ranges(df, results):
    """Verify data is within physical ranges."""
    print("\n[3/10] Checking physical ranges...")
    
    checks_passed = True
    
    # Energy range
    if df['energy_MeV'].min() >= 70 and df['energy_MeV'].max() <= 6000:
        print(f"  Energy: {df['energy_MeV'].min()}-{df['energy_MeV'].max()} MeV ✓")
    else:
        results.add_fail("Physical ranges", "Energy out of expected range")
        checks_passed = False
    
    # Depth range
    if df['depth_mm'].min() >= 0 and df['depth_mm'].max() <= SCINT_THICKNESS:
        print(f"  Depth: {df['depth_mm'].min():.2f}-{df['depth_mm'].max():.2f} mm ✓")
    else:
        results.add_warning("Physical ranges", "Depth outside scintillator")
    
    # Beta (0 < β < 1)
    if (df['beta'] > 0).all() and (df['beta'] < 1).all():
        print(f"  Beta: {df['beta'].min():.4f}-{df['beta'].max():.4f} ✓")
    else:
        results.add_fail("Physical ranges", "Beta outside (0,1)")
        checks_passed = False
    
    # Gamma (γ ≥ 1)
    if (df['gamma'] >= 1).all():
        print(f"  Gamma: {df['gamma'].min():.4f}-{df['gamma'].max():.4f} ✓")
    else:
        results.add_fail("Physical ranges", "Gamma < 1 (unphysical)")
        checks_passed = False
    
    # dE/dx (positive)
    if (df['dEdx_MeV_per_mm'] >= 0).all():
        print(f"  dE/dx: {df['dEdx_MeV_per_mm'].min():.4f}-{df['dEdx_MeV_per_mm'].max():.4f} MeV/mm ✓")
    else:
        results.add_warning("Physical ranges", "Negative dE/dx values")
    
    if checks_passed:
        results.add_pass("Physical ranges", "All values within expected bounds")


def check_row_counts(df, results):
    """Each energy should have exactly 100 depth bins."""
    print("\n[4/10] Checking row counts per energy...")
    
    counts = df.groupby('energy_MeV').size()
    
    if (counts == N_BINS).all():
        results.add_pass("Row counts", f"All {len(counts)} energies have {N_BINS} bins")
    else:
        bad_counts = counts[counts != N_BINS]
        results.add_fail("Row counts", f"Energies with wrong bin count: {bad_counts.to_dict()}")


def check_bethe_bloch_agreement(df, results):
    """Verify simulated dE/dx agrees with Bethe-Bloch within reasonable margin."""
    print("\n[5/10] Checking Bethe-Bloch agreement...")
    
    # Calculate ratio of simulated to analytical
    # Use mean dE/dx per energy
    summary = df.groupby('energy_MeV').agg({
        'dEdx_MeV_per_mm': 'mean',
        'bethe_dEdx_MeV_per_mm': 'first'
    }).reset_index()
    
    # Filter out low dE/dx where ratio is noisy
    valid = summary['bethe_dEdx_MeV_per_mm'] > 0.1
    summary = summary[valid]
    
    summary['ratio'] = summary['dEdx_MeV_per_mm'] / summary['bethe_dEdx_MeV_per_mm']
    
    mean_ratio = summary['ratio'].mean()
    std_ratio = summary['ratio'].std()
    
    print(f"  Mean ratio (sim/analytical): {mean_ratio:.3f}")
    print(f"  Std ratio: {std_ratio:.3f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(summary['energy_MeV'], summary['ratio'], 'bo-', markersize=8)
    plt.axhline(1.0, color='r', linestyle='--', label='Perfect agreement')
    plt.axhline(0.9, color='orange', linestyle=':', alpha=0.7)
    plt.axhline(1.1, color='orange', linestyle=':', alpha=0.7)
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Simulated / Analytical dE/dx')
    plt.title('Bethe-Bloch Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.savefig(PLOTS_DIR / 'bethe_bloch_validation.png', dpi=150)
    plt.close()
    
    if 0.5 < mean_ratio < 2.0 and std_ratio < 0.5:
        results.add_pass("Bethe-Bloch", f"Mean ratio = {mean_ratio:.3f} ± {std_ratio:.3f}")
    elif 0.3 < mean_ratio < 3.0:
        results.add_warning("Bethe-Bloch", f"Mean ratio = {mean_ratio:.3f} ± {std_ratio:.3f} (slightly off)")
    else:
        results.add_fail("Bethe-Bloch", f"Mean ratio = {mean_ratio:.3f} ± {std_ratio:.3f} (too far off)")


def check_energy_conservation(df, results):
    """Check if deposited energy is reasonable."""
    print("\n[6/10] Checking energy conservation...")
    
    conservation_ok = True
    
    for energy in sorted(df['energy_MeV'].unique()):
        subset = df[df['energy_MeV'] == energy]
        total_deposited = subset['energy_deposit_MeV'].sum() / N_HISTORIES
        
        # For high energy particles, most pass through
        # For low energy, most stop in scintillator
        expected_max = energy  # Can't deposit more than incident energy
        
        print(f"  {energy:5} MeV: Deposited ~{total_deposited:.2f} MeV/proton", end="")
        
        if total_deposited > expected_max * 1.1:
            print(" ⚠️ (>110% of incident)")
            conservation_ok = False
        elif total_deposited > expected_max * 0.5 and energy > 300:
            print(" ⚠️ (high for this energy)")
        else:
            print(" ✓")
    
    if conservation_ok:
        results.add_pass("Energy conservation", "Deposited energy within bounds")
    else:
        results.add_warning("Energy conservation", "Some energies show unexpected deposition")


def check_range_energy_monotonicity(df, results):
    """Range should increase with energy."""
    print("\n[7/10] Checking range-energy relationship...")
    
    range_summary = df.groupby('energy_MeV')['total_range_mm'].first().sort_index()
    
    is_monotonic = range_summary.is_monotonic_increasing
    
    print(f"  Range values: {range_summary.values[:5]}... {range_summary.values[-3:]}")
    
    if is_monotonic:
        results.add_pass("Range-energy", "Range monotonically increases with energy")
    else:
        results.add_fail("Range-energy", "Range does NOT increase monotonically!")


def check_split_integrity(results):
    """Verify train/val/test have no energy overlap."""
    print("\n[8/10] Checking train/val/test splits...")
    
    try:
        train = pd.read_csv(PINN_DIR / 'train_split.csv')
        val = pd.read_csv(PINN_DIR / 'val_split.csv')
        test = pd.read_csv(PINN_DIR / 'test_split.csv')
        
        train_E = set(train['energy_MeV'].unique())
        val_E = set(val['energy_MeV'].unique())
        test_E = set(test['energy_MeV'].unique())
        
        print(f"  Train energies ({len(train_E)}): {sorted(train_E)}")
        print(f"  Val energies ({len(val_E)}): {sorted(val_E)}")
        print(f"  Test energies ({len(test_E)}): {sorted(test_E)}")
        
        overlap_tv = train_E & val_E
        overlap_tt = train_E & test_E
        overlap_vt = val_E & test_E
        
        if overlap_tv or overlap_tt or overlap_vt:
            results.add_fail("Split integrity", f"Overlaps: train-val={overlap_tv}, train-test={overlap_tt}, val-test={overlap_vt}")
        else:
            results.add_pass("Split integrity", "No energy overlap between splits")
            
    except FileNotFoundError as e:
        results.add_fail("Split integrity", f"Split file not found: {e}")


def create_feature_scaling(df, results):
    """Create and save feature scalers."""
    print("\n[9/10] Creating feature scaling...")
    
    if not HAS_SKLEARN:
        results.add_warning("Feature scaling", "sklearn not installed - skipping")
        return
    
    # Features to scale
    features = ['energy_MeV', 'depth_mm', 'beta', 'gamma', 'momentum_MeV_c']
    target = 'dEdx_MeV_per_mm'
    
    # StandardScaler for features
    scaler_features = StandardScaler()
    scaler_features.fit(df[features])
    
    # MinMaxScaler for target (to keep positive)
    scaler_target = MinMaxScaler()
    scaler_target.fit(df[[target]])
    
    # Save scalers
    joblib.dump(scaler_features, PINN_DIR / 'scaler_features.pkl')
    joblib.dump(scaler_target, PINN_DIR / 'scaler_target.pkl')
    
    # Save parameters as JSON for portability
    scaling_params = {
        'feature_names': features,
        'feature_means': scaler_features.mean_.tolist(),
        'feature_stds': scaler_features.scale_.tolist(),
        'target_name': target,
        'target_min': scaler_target.data_min_.tolist(),
        'target_max': scaler_target.data_max_.tolist()
    }
    
    with open(PINN_DIR / 'scaling_params.json', 'w') as f:
        json.dump(scaling_params, f, indent=2)
    
    results.add_pass("Feature scaling", f"Scalers saved for {len(features)} features")


def create_visualizations(df, results):
    """Create visualization plots."""
    print("\n[10/10] Creating visualizations...")
    
    # 1. All dose-depth profiles
    fig, ax = plt.subplots(figsize=(12, 8))
    for energy in sorted(df['energy_MeV'].unique()):
        subset = df[df['energy_MeV'] == energy]
        ax.plot(subset['depth_mm'], subset['energy_deposit_MeV'] / N_HISTORIES, 
                label=f'{energy} MeV', alpha=0.8)
    ax.set_xlabel('Depth (mm)')
    ax.set_ylabel('Energy Deposit (MeV/proton)')
    ax.set_title('Dose-Depth Profiles: 70-6000 MeV')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / 'all_dose_profiles.png', dpi=150)
    plt.close()
    print("  Saved: all_dose_profiles.png")
    
    # 2. Bragg peaks (clinical energies)
    clinical = [70, 100, 150, 200, 250]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, energy in enumerate(clinical):
        ax = axes.flat[i]
        subset = df[df['energy_MeV'] == energy]
        ax.plot(subset['depth_mm'], subset['energy_deposit_MeV'] / N_HISTORIES, 
                'b-', linewidth=2)
        ax.set_title(f'{energy} MeV')
        ax.set_xlabel('Depth (mm)')
        ax.set_ylabel('Energy Deposit (MeV/proton)')
        ax.grid(True, alpha=0.3)
    axes.flat[-1].axis('off')  # Hide empty subplot
    plt.suptitle('Clinical Energy Bragg Peaks', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'bragg_peaks.png', dpi=150)
    plt.close()
    print("  Saved: bragg_peaks.png")
    
    # 3. Stopping power vs energy
    summary = df.groupby('energy_MeV').agg({
        'dEdx_MeV_per_mm': 'mean',
        'bethe_dEdx_MeV_per_mm': 'first'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(summary['energy_MeV'], summary['bethe_dEdx_MeV_per_mm'], 
             'r-', linewidth=2, label='Bethe-Bloch (analytical)')
    plt.plot(summary['energy_MeV'], summary['dEdx_MeV_per_mm'], 
             'bo', markersize=8, label='TOPAS simulation')
    plt.xscale('log')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('dE/dx (MeV/mm)')
    plt.title('Stopping Power vs Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / 'stopping_power.png', dpi=150)
    plt.close()
    print("  Saved: stopping_power.png")
    
    # 4. Correlation heatmap
    if HAS_SEABORN:
        features = ['energy_MeV', 'depth_mm', 'beta', 'gamma', 
                    'momentum_MeV_c', 'dEdx_MeV_per_mm']
        corr = df[features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'correlation_heatmap.png', dpi=150)
        plt.close()
        print("  Saved: correlation_heatmap.png")
    
    results.add_pass("Visualizations", f"Saved to {PLOTS_DIR}")


def create_readme(results):
    """Create README for pinn_data directory."""
    readme_content = """# PINN Training Data Documentation

## Data Source
- **Software:** OpenTOPAS v4.2 with Geant4 v11.3.2
- **Energies:** 14 points from 70-6000 MeV
- **Histories:** 10,000 protons per energy (140,000 total)
- **Generated:** December 27, 2025

## File Descriptions

| File | Description |
|------|-------------|
| `pinn_training_data.csv` | Master dataset (1,400 rows) |
| `train_split.csv` | Training set (8 energies) |
| `val_split.csv` | Validation set (4 energies) |
| `test_split.csv` | Test set (2 energies) |
| `energy_summary.csv` | Physics summary per energy |
| `material_properties.json` | EJ-200 scintillator properties |
| `scaling_params.json` | Feature normalization parameters |
| `scaler_features.pkl` | Sklearn StandardScaler (features) |
| `scaler_target.pkl` | Sklearn MinMaxScaler (target) |

## Column Definitions

| Column | Units | Description |
|--------|-------|-------------|
| `energy_MeV` | MeV | Incident proton kinetic energy |
| `z_bin` | - | Bin number (0-99) |
| `depth_mm` | mm | Depth in scintillator |
| `energy_deposit_MeV` | MeV | Total energy deposited in bin |
| `dEdx_MeV_per_mm` | MeV/mm | Stopping power (per proton) |
| `dose_Gy` | Gy | Physical dose |
| `beta` | - | Relativistic velocity v/c |
| `gamma` | - | Lorentz factor |
| `momentum_MeV_c` | MeV/c | Particle momentum |
| `bethe_dEdx_MeV_per_mm` | MeV/mm | Analytical Bethe-Bloch |
| `total_range_mm` | mm | CSDA range in material |
| `residual_range_mm` | mm | Remaining range from depth |

## Train/Val/Test Split

- **Train (8):** 70, 150, 250, 500, 1000, 2000, 4500, 6000 MeV
- **Validate (4):** 100, 300, 1500, 3000 MeV
- **Test (2):** 200, 750 MeV (held out for final evaluation)

## Validation Status

Run `python validate_pinn_data.py` to verify data quality.

## Plots

See `plots/` directory for:
- `all_dose_profiles.png` - Dose-depth curves for all energies
- `bragg_peaks.png` - Clinical energy Bragg peaks
- `stopping_power.png` - dE/dx vs energy
- `correlation_heatmap.png` - Feature correlations
- `bethe_bloch_validation.png` - Simulation vs analytical
"""
    
    with open(PINN_DIR / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print("\nCreated: pinn_data/README.md")


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("KAVACH BL4S - PINN Data Validation")
    print("=" * 60)
    
    results = ValidationResult()
    
    # Load data
    try:
        df = pd.read_csv(PINN_DIR / 'pinn_training_data.csv')
        print(f"\nLoaded {len(df)} rows from pinn_training_data.csv")
    except FileNotFoundError:
        print("ERROR: pinn_training_data.csv not found!")
        print("Run extract_pinn_data.py first.")
        return False
    
    # Run all checks
    check_missing_values(df, results)
    check_duplicates(df, results)
    check_physical_ranges(df, results)
    check_row_counts(df, results)
    check_bethe_bloch_agreement(df, results)
    check_energy_conservation(df, results)
    check_range_energy_monotonicity(df, results)
    check_split_integrity(results)
    create_feature_scaling(df, results)
    create_visualizations(df, results)
    
    # Create documentation
    create_readme(results)
    
    # Print summary
    return results.summary()


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
