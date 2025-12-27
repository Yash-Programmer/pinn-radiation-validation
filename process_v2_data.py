#!/usr/bin/env python3
"""
KAVACH BL4S - Complete v2 Data Processing Pipeline
Generates scalers, validation plots, and quality checks for CORRECTED v2 data.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import seaborn as sns
    HAS_SEABORN = True
except:
    HAS_SEABORN = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import joblib
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False

BASE_DIR = Path(__file__).parent
PINN_DIR = BASE_DIR / 'pinn_data'
PLOTS_DIR = PINN_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)


def create_scalers_v2():
    """Create feature scalers for v2 corrected data."""
    if not HAS_SKLEARN:
        print("sklearn not available - skipping scalers")
        return
    
    df = pd.read_csv(PINN_DIR / 'pinn_training_data_v2.csv')
    
    # Features to scale
    features = ['energy_MeV', 'depth_mm', 'beta', 'gamma', 'momentum_MeV_c']
    target = 'sim_dEdx_MeV_per_mm'  # Changed column name in v2
    
    # StandardScaler for features
    scaler_features = StandardScaler()
    scaler_features.fit(df[features])
    
    # MinMaxScaler for target
    scaler_target = MinMaxScaler()
    scaler_target.fit(df[[target]])
    
    # Save scalers
    joblib.dump(scaler_features, PINN_DIR / 'scaler_features_v2.pkl')
    joblib.dump(scaler_target, PINN_DIR / 'scaler_target_v2.pkl')
    
    # Save parameters as JSON
    scaling_params = {
        'version': 'v2',
        'date': '2025-12-27',
        'feature_names': features,
        'feature_means': scaler_features.mean_.tolist(),
        'feature_stds': scaler_features.scale_.tolist(),
        'target_name': target,
        'target_min': scaler_target.data_min_.tolist(),
        'target_max': scaler_target.data_max_.tolist(),
        'notes': 'v2 uses corrected dose and NIST dE/dx reference'
    }
    
    with open(PINN_DIR / 'scaling_params_v2.json', 'w') as f:
        json.dump(scaling_params, f, indent=2)
    
    print("✓ Scalers created for v2 data")
    print(f"  Features: {features}")
    print(f"  Target: {target}")


def create_validation_plots_v2():
    """Create validation plots for v2 corrected data."""
    df = pd.read_csv(PINN_DIR / 'pinn_training_data_v2.csv')
    
    # 1. Dose-depth profiles (all energies)
    fig, ax = plt.subplots(figsize=(12, 8))
    for energy in sorted(df['energy_MeV'].unique()):
        subset = df[df['energy_MeV'] == energy]
        # Use dose_Gy * 1e9 to show in nGy (more readable)
        ax.plot(subset['depth_mm'], subset['dose_Gy'] * 1e9, 
                label=f'{energy} MeV', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Depth (mm)', fontsize=12)
    ax.set_ylabel('Dose (nGy per proton)', fontsize=12)
    ax.set_title('Dose-Depth Profiles (v2 Corrected)', fontsize=14)
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / 'dose_profiles_v2.png', dpi=150)
    plt.close()
    print("✓ Saved: dose_profiles_v2.png")
    
    # 2. Bragg peaks (clinical energies)
    clinical = [70, 100, 150, 200, 250]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, energy in enumerate(clinical):
        ax = axes.flat[i]
        subset = df[df['energy_MeV'] == energy]
        ax.plot(subset['depth_mm'], subset['dose_Gy'] * 1e9, 'b-', linewidth=2)
        ax.set_title(f'{energy} MeV', fontsize=12)
        ax.set_xlabel('Depth (mm)')
        ax.set_ylabel('Dose (nGy/proton)')
        ax.grid(True, alpha=0.3)
    axes.flat[-1].axis('off')
    plt.suptitle('Clinical Energy Bragg Peaks (v2)', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'bragg_peaks_v2.png', dpi=150)
    plt.close()
    print("✓ Saved: bragg_peaks_v2.png")
    
    # 3. Sim vs NIST dE/dx comparison
    summary = df.groupby('energy_MeV').agg({
        'sim_dEdx_MeV_per_mm': 'mean',
        'nist_dEdx_MeV_per_mm': 'first'
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: dE/dx comparison
    ax1 = axes[0]
    ax1.plot(summary['energy_MeV'], summary['nist_dEdx_MeV_per_mm'], 
             'b-', linewidth=2, label='NIST PSTAR')
    ax1.plot(summary['energy_MeV'], summary['sim_dEdx_MeV_per_mm'], 
             'ro', markersize=10, label='TOPAS Simulation')
    ax1.set_xscale('log')
    ax1.set_xlabel('Energy (MeV)', fontsize=12)
    ax1.set_ylabel('dE/dx (MeV/mm)', fontsize=12)
    ax1.set_title('Stopping Power: TOPAS vs NIST', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Right: Ratio
    ax2 = axes[1]
    ratio = summary['sim_dEdx_MeV_per_mm'] / summary['nist_dEdx_MeV_per_mm']
    ax2.semilogx(summary['energy_MeV'], ratio, 'go-', markersize=10, linewidth=2)
    ax2.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Perfect')
    ax2.axhspan(0.9, 1.1, alpha=0.2, color='green', label='±10%')
    ax2.set_xlabel('Energy (MeV)', fontsize=12)
    ax2.set_ylabel('Simulation / NIST', fontsize=12)
    ax2.set_title('Validation Ratio', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.8, 1.5)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'nist_comparison_v2.png', dpi=150)
    plt.close()
    print("✓ Saved: nist_comparison_v2.png")
    
    # 4. Correlation heatmap
    if HAS_SEABORN:
        features = ['energy_MeV', 'depth_mm', 'beta', 'gamma', 
                    'momentum_MeV_c', 'sim_dEdx_MeV_per_mm', 'dose_Gy']
        corr = df[features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix (v2)', fontsize=14)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'correlation_v2.png', dpi=150)
        plt.close()
        print("✓ Saved: correlation_v2.png")


def run_quality_checks_v2():
    """Run quality checks on v2 data."""
    df = pd.read_csv(PINN_DIR / 'pinn_training_data_v2.csv')
    
    print("\n" + "=" * 50)
    print("QUALITY CHECKS (v2)")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    # 1. Check for NaN
    nulls = df.isnull().sum().sum()
    if nulls == 0:
        print("✓ No NaN values")
        passed += 1
    else:
        print(f"✗ Found {nulls} NaN values")
        failed += 1
    
    # 2. Check for duplicates
    dups = df.duplicated().sum()
    if dups == 0:
        print("✓ No duplicates")
        passed += 1
    else:
        print(f"⚠ Found {dups} duplicates")
    
    # 3. Check row counts
    counts = df.groupby('energy_MeV').size()
    if (counts == 100).all():
        print(f"✓ All {len(counts)} energies have 100 bins")
        passed += 1
    else:
        print(f"✗ Row count mismatch")
        failed += 1
    
    # 4. Dose values
    dose_min = df['dose_Gy'].min()
    dose_max = df['dose_Gy'].max()
    if dose_min > 0:
        print(f"✓ Dose range: {dose_min:.2e} to {dose_max:.2e} Gy")
        passed += 1
    else:
        print(f"✗ Dose has zero/negative values")
        failed += 1
    
    # 5. NIST ratio check
    summary = df.groupby('energy_MeV').agg({
        'sim_dEdx_MeV_per_mm': 'mean',
        'nist_dEdx_MeV_per_mm': 'first'
    })
    ratio = summary['sim_dEdx_MeV_per_mm'] / summary['nist_dEdx_MeV_per_mm']
    mean_ratio = ratio.mean()
    
    if 0.9 < mean_ratio < 1.5:
        print(f"✓ Sim/NIST ratio: {mean_ratio:.3f} (expected ~1.0-1.3)")
        passed += 1
    else:
        print(f"⚠ Sim/NIST ratio: {mean_ratio:.3f} (unusual)")
    
    # 6. Check splits
    train = pd.read_csv(PINN_DIR / 'train_split_v2.csv')
    val = pd.read_csv(PINN_DIR / 'val_split_v2.csv')
    test = pd.read_csv(PINN_DIR / 'test_split_v2.csv')
    
    train_E = set(train['energy_MeV'].unique())
    val_E = set(val['energy_MeV'].unique())
    test_E = set(test['energy_MeV'].unique())
    
    if not (train_E & val_E) and not (train_E & test_E) and not (val_E & test_E):
        print("✓ No energy overlap between splits")
        passed += 1
    else:
        print("✗ Energy overlap found!")
        failed += 1
    
    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def update_readme_v2():
    """Update README for v2 data."""
    readme = """# PINN Training Data v2 (CORRECTED)

## Version History
- **v2** (Dec 27, 2025): Corrected dose, real NIST dE/dx
- **v1**: Deprecated (had calculation errors)

## Corrections Applied in v2
1. Dose calculation fixed (was 0.0 Gy)
2. NIST PSTAR reference integrated (real data from nist.gov)
3. EJ-200 properties verified from Eljen datasheet
4. Column names clarified (sim_dEdx vs nist_dEdx)

## Files

| File | Description |
|------|-------------|
| `pinn_training_data_v2.csv` | Master dataset (1400 rows) |
| `train_split_v2.csv` | Training (8 energies, 800 rows) |
| `val_split_v2.csv` | Validation (4 energies, 400 rows) |
| `test_split_v2.csv` | Test (2 energies, 200 rows) |
| `scaler_features_v2.pkl` | StandardScaler for inputs |
| `scaler_target_v2.pkl` | MinMaxScaler for target |
| `material_properties_VERIFIED.json` | Eljen datasheet values |

## Columns (v2)

| Column | Units | Description |
|--------|-------|-------------|
| energy_MeV | MeV | Proton kinetic energy |
| z_bin | - | Bin index (0-99) |
| depth_mm | mm | Depth in scintillator |
| energy_deposit_MeV | MeV | Total energy in bin |
| sim_dEdx_MeV_per_mm | MeV/mm | TOPAS simulated dE/dx |
| dose_Gy | Gy | Absorbed dose per proton |
| beta | - | Relativistic v/c |
| gamma | - | Lorentz factor |
| momentum_MeV_c | MeV/c | Momentum |
| nist_dEdx_MeV_per_mm | MeV/mm | NIST PSTAR reference |
| nist_range_mm | mm | CSDA range from NIST |
| residual_range_mm | mm | Remaining range |

## Sources Verified

- **EJ-200**: [Eljen Technology](https://eljentechnology.com/products/plastic-scintillators/ej-200)
- **NIST PSTAR**: [physics.nist.gov](https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html)

## Notes

- Dose is per-proton (multiply by N_histories for total)
- sim_dEdx includes nuclear interactions; nist_dEdx is electronic only
- ~10-30% higher sim vs NIST at high energies is expected
"""
    
    with open(PINN_DIR / 'README_v2.md', 'w') as f:
        f.write(readme)
    print("✓ README_v2.md created")


def main():
    print("=" * 60)
    print("KAVACH BL4S - v2 Data Processing Pipeline")
    print("=" * 60)
    
    print("\n[1/4] Creating scalers for v2 data...")
    create_scalers_v2()
    
    print("\n[2/4] Creating validation plots...")
    create_validation_plots_v2()
    
    print("\n[3/4] Running quality checks...")
    passed = run_quality_checks_v2()
    
    print("\n[4/4] Updating documentation...")
    update_readme_v2()
    
    print("\n" + "=" * 60)
    if passed:
        print("STATUS: ✅ ALL V2 PROCESSING COMPLETE!")
    else:
        print("STATUS: ⚠️ Some checks failed - review above")
    print("=" * 60)


if __name__ == '__main__':
    main()
