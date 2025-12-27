#!/usr/bin/env python3
"""
KAVACH BL4S - PINN Data Extraction and Processing
Extracts TOPAS simulation data and prepares it for Physics-Informed Neural Network training.

Features:
- Combines all EnergyDepZ files into master dataset
- Calculates physics features (β, γ, momentum)
- Computes Bethe-Bloch reference values
- Creates train/val/test splits
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# Physical constants
M_PROTON = 938.272  # MeV/c²
C = 1.0  # Natural units, c=1
K_BETHE = 0.307075  # MeV·cm²/g

# EJ-200 material properties
DENSITY = 1.023  # g/cm³
MEAN_EXCITATION = 64.7e-6  # MeV (64.7 eV)
Z_EFF = 3.29  # Effective Z for PVT (H:8.47%, C:91.53%)
A_EFF = 5.74  # Effective A
BIRKS_CONSTANT = 0.1232  # mm/MeV

# Scintillator geometry
THICKNESS_MM = 10.0
N_BINS = 100
BIN_SIZE_MM = THICKNESS_MM / N_BINS  # 0.1 mm


def calculate_beta(energy_MeV):
    """Calculate relativistic β = v/c from kinetic energy."""
    gamma = (energy_MeV + M_PROTON) / M_PROTON
    beta = np.sqrt(1 - 1/gamma**2)
    return beta


def calculate_gamma(energy_MeV):
    """Calculate Lorentz factor γ."""
    return (energy_MeV + M_PROTON) / M_PROTON


def calculate_momentum(energy_MeV):
    """Calculate momentum in MeV/c."""
    total_energy = energy_MeV + M_PROTON
    momentum = np.sqrt(total_energy**2 - M_PROTON**2)
    return momentum


def bethe_bloch_dEdx(energy_MeV):
    """
    Calculate stopping power dE/dx using Bethe-Bloch formula.
    Returns dE/dx in MeV/mm for EJ-200 scintillator.
    """
    beta = calculate_beta(energy_MeV)
    gamma = calculate_gamma(energy_MeV)
    
    if beta < 0.001:  # Avoid division by zero at very low energy
        return np.nan
    
    # Wmax - maximum energy transfer
    me = 0.511  # electron mass in MeV
    Wmax = 2 * me * beta**2 * gamma**2 / (1 + 2*gamma*me/M_PROTON + (me/M_PROTON)**2)
    
    # Bethe-Bloch formula
    term1 = np.log(2 * me * beta**2 * gamma**2 * Wmax / MEAN_EXCITATION**2)
    term2 = 2 * beta**2
    
    # Density correction (simplified)
    delta = 0  # Negligible for most energies
    
    dEdx = K_BETHE * DENSITY * (Z_EFF / A_EFF) * (1/beta**2) * (0.5 * term1 - term2 - delta/2)
    
    # Convert from MeV/cm to MeV/mm
    return dEdx / 10.0


def calculate_range_CSDA(energy_MeV, step=1.0):
    """
    Calculate CSDA range (Continuous Slowing Down Approximation).
    Integration of 1/(dE/dx) from 0 to E.
    Returns range in mm.
    """
    if energy_MeV < 1:
        return 0.0
    
    energies = np.arange(step, energy_MeV, step)
    if len(energies) == 0:
        return 0.0
    
    dEdx_values = np.array([bethe_bloch_dEdx(E) for E in energies])
    # Filter out invalid values
    valid = ~np.isnan(dEdx_values) & (dEdx_values > 0)
    
    if np.sum(valid) == 0:
        return 0.0
    
    # Trapezoidal integration
    range_mm = np.trapz(1.0 / dEdx_values[valid], energies[valid])
    return range_mm


def extract_energy_from_filename(filename):
    """Extract energy value from filename like 'EnergyDepZ_70MeV_Res.csv'."""
    basename = os.path.basename(filename)
    # Pattern: EnergyDepZ_XXXMeV_Res.csv
    parts = basename.replace('EnergyDepZ_', '').replace('MeV_Res.csv', '')
    try:
        return int(parts)
    except ValueError:
        return None


def load_energydepz_file(filepath):
    """Load and parse an EnergyDepZ CSV file from TOPAS."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    x_bin = int(parts[0].strip())
                    y_bin = int(parts[1].strip())
                    z_bin = int(parts[2].strip())
                    energy_dep = float(parts[3].strip())
                    data.append({
                        'z_bin': z_bin,
                        'energy_deposit_MeV': energy_dep
                    })
                except ValueError:
                    continue
    return pd.DataFrame(data)


def process_all_files(runs_dir):
    """Process all EnergyDepZ files and create master dataset."""
    pattern = os.path.join(runs_dir, 'EnergyDepZ_*MeV_Res.csv')
    files = glob.glob(pattern)
    
    all_data = []
    
    for filepath in sorted(files):
        energy = extract_energy_from_filename(filepath)
        if energy is None:
            continue
            
        print(f"Processing {energy} MeV...")
        df = load_energydepz_file(filepath)
        
        if df.empty:
            print(f"  Warning: Empty data for {energy} MeV")
            continue
        
        # Add energy column
        df['energy_MeV'] = energy
        
        # Calculate depth in mm (z_bin * bin_size)
        df['depth_mm'] = (df['z_bin'] + 0.5) * BIN_SIZE_MM  # Center of bin
        
        # Calculate physics features
        df['beta'] = calculate_beta(energy)
        df['gamma'] = calculate_gamma(energy)
        df['momentum_MeV_c'] = calculate_momentum(energy)
        
        # Calculate dE/dx (MeV/mm per proton per bin)
        n_histories = 10000
        df['dEdx_MeV_per_mm'] = df['energy_deposit_MeV'] / (n_histories * BIN_SIZE_MM)
        
        # Calculate dose (simplified - MeV to Gy conversion)
        # Volume of each bin: 50mm × 50mm × 0.1mm = 250 mm³ = 0.25 cm³
        # Mass = 0.25 cm³ × 1.023 g/cm³ = 0.256 g
        bin_mass_g = (50 * 50 * BIN_SIZE_MM) / 1000 * DENSITY  # in grams
        df['dose_Gy'] = df['energy_deposit_MeV'] / (bin_mass_g * 1e6) * 1.602e-13 * 1e9 / n_histories
        
        # Bethe-Bloch reference
        df['bethe_dEdx_MeV_per_mm'] = bethe_bloch_dEdx(energy)
        
        # Calculate residual range
        total_range = calculate_range_CSDA(energy)
        df['total_range_mm'] = total_range
        df['residual_range_mm'] = total_range - df['depth_mm']
        df['residual_range_mm'] = df['residual_range_mm'].clip(lower=0)
        
        all_data.append(df)
    
    # Combine all data
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by energy and depth
    master_df = master_df.sort_values(['energy_MeV', 'depth_mm'])
    
    # Reorder columns
    columns_order = [
        'energy_MeV', 'z_bin', 'depth_mm', 
        'energy_deposit_MeV', 'dEdx_MeV_per_mm', 'dose_Gy',
        'beta', 'gamma', 'momentum_MeV_c',
        'bethe_dEdx_MeV_per_mm', 'total_range_mm', 'residual_range_mm'
    ]
    master_df = master_df[columns_order]
    
    return master_df


def create_train_val_test_splits(df, output_dir):
    """Create train/validation/test splits."""
    energies = sorted(df['energy_MeV'].unique())
    
    # Split strategy:
    # Train: 70, 150, 250, 500, 1000, 2000, 4500, 6000 MeV (8 energies)
    # Validate: 100, 300, 1500, 3000 MeV (4 energies)
    # Test: 200, 750 MeV (2 energies - held completely out)
    
    train_energies = [70, 150, 250, 500, 1000, 2000, 4500, 6000]
    val_energies = [100, 300, 1500, 3000]
    test_energies = [200, 750]
    
    train_df = df[df['energy_MeV'].isin(train_energies)]
    val_df = df[df['energy_MeV'].isin(val_energies)]
    test_df = df[df['energy_MeV'].isin(test_energies)]
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
    
    print(f"\nData splits created:")
    print(f"  Train: {len(train_df)} rows ({len(train_energies)} energies)")
    print(f"  Validate: {len(val_df)} rows ({len(val_energies)} energies)")
    print(f"  Test: {len(test_df)} rows ({len(test_energies)} energies)")
    
    return train_df, val_df, test_df


def create_material_properties_json(output_dir):
    """Create JSON file with material properties."""
    import json
    
    properties = {
        "material": "EJ-200",
        "type": "plastic_scintillator",
        "base": "polyvinyltoluene",
        "properties": {
            "density_g_per_cm3": DENSITY,
            "composition": {
                "hydrogen_fraction": 0.0847,
                "carbon_fraction": 0.9153
            },
            "mean_excitation_energy_eV": 64.7,
            "effective_Z": Z_EFF,
            "effective_A": A_EFF,
            "birks_constant_mm_per_MeV": BIRKS_CONSTANT,
            "birks_constant_g_per_MeV_cm2": 0.0126
        },
        "optical_properties": {
            "light_yield_photons_per_MeV": 10000,
            "peak_emission_wavelength_nm": 425,
            "decay_time_ns": 2.1,
            "refractive_index": 1.58,
            "attenuation_length_cm": 380
        },
        "geometry": {
            "thickness_mm": THICKNESS_MM,
            "width_mm": 50.0,
            "height_mm": 50.0
        }
    }
    
    with open(os.path.join(output_dir, 'material_properties.json'), 'w') as f:
        json.dump(properties, f, indent=2)
    
    print(f"Material properties saved to {output_dir}/material_properties.json")


def main():
    """Main execution."""
    # Define paths
    base_dir = Path(__file__).parent
    runs_dir = base_dir / 'runs'
    output_dir = base_dir / 'pinn_data'
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("KAVACH BL4S - PINN Data Extraction")
    print("=" * 60)
    
    # Process all files
    print("\n[1/4] Extracting data from TOPAS outputs...")
    master_df = process_all_files(runs_dir)
    
    # Save master dataset
    master_path = output_dir / 'pinn_training_data.csv'
    master_df.to_csv(master_path, index=False)
    print(f"\nMaster dataset saved: {master_path}")
    print(f"  Total rows: {len(master_df)}")
    print(f"  Energies: {sorted(master_df['energy_MeV'].unique())}")
    
    # Create train/val/test splits
    print("\n[2/4] Creating train/val/test splits...")
    create_train_val_test_splits(master_df, output_dir)
    
    # Create material properties JSON
    print("\n[3/4] Creating material properties file...")
    create_material_properties_json(output_dir)
    
    # Summary statistics
    print("\n[4/4] Summary Statistics")
    print("=" * 60)
    
    summary = master_df.groupby('energy_MeV').agg({
        'energy_deposit_MeV': 'sum',
        'dEdx_MeV_per_mm': 'mean',
        'dose_Gy': 'mean',
        'beta': 'first',
        'gamma': 'first',
        'momentum_MeV_c': 'first',
        'bethe_dEdx_MeV_per_mm': 'first',
        'total_range_mm': 'first'
    }).round(4)
    
    summary.columns = ['Total_EdepMeV', 'Mean_dEdx', 'Mean_DoseGy', 
                       'Beta', 'Gamma', 'Momentum', 'Bethe_dEdx', 'CSDA_Range']
    
    print(summary.to_string())
    
    # Save summary
    summary.to_csv(output_dir / 'energy_summary.csv')
    
    print("\n" + "=" * 60)
    print("PINN data extraction complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
