#!/usr/bin/env python3
"""
KAVACH BL4S - CORRECTED Data Extraction
Fixes all identified issues from deep analysis.

FIXES APPLIED:
1. Dose calculation formula corrected
2. Bethe-Bloch uses NIST interpolation (not custom formula)
3. dE/dx properly labeled (simulation vs reference)
4. CSDA range from NIST data
5. Material properties verified against Eljen datasheet
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

# ============================================================================
# VERIFIED PHYSICAL CONSTANTS (from official sources)
# ============================================================================
M_PROTON = 938.272046  # MeV/c² (PDG 2022)

# EJ-200 Properties - VERIFIED from Eljen Technology Datasheet
# Source: https://eljentechnology.com/products/plastic-scintillators/ej-200
EJ200_DENSITY = 1.023  # g/cm³ (VERIFIED)
EJ200_LIGHT_YIELD = 10000  # photons/MeV (VERIFIED)
EJ200_DECAY_TIME = 2.1  # ns (VERIFIED)
EJ200_REFRACTIVE_INDEX = 1.58  # (VERIFIED)
EJ200_ATTENUATION_LENGTH = 380  # cm (VERIFIED)
EJ200_PEAK_WAVELENGTH = 425  # nm (VERIFIED)

# Atomic composition from datasheet:
# H: 5.17e22 atoms/cm³, C: 4.69e22 atoms/cm³
H_DENSITY = 5.17e22  # atoms/cm³
C_DENSITY = 4.69e22  # atoms/cm³
H_FRACTION_BY_WEIGHT = 0.0847  # Calculated from atomic densities
C_FRACTION_BY_WEIGHT = 0.9153

# Mean excitation energy for polyvinyltoluene (from NIST)
MEAN_EXCITATION_EV = 64.7  # eV

# Scintillator geometry
THICKNESS_MM = 10.0
N_BINS = 100
BIN_SIZE_MM = THICKNESS_MM / N_BINS  # 0.1 mm
WIDTH_MM = 50.0
HEIGHT_MM = 50.0
N_HISTORIES = 10000

# ============================================================================
# NIST PSTAR DATA (Real data from physics.nist.gov)
# ============================================================================
NIST_ENERGIES = np.array([70, 100, 200, 500, 1000, 2000, 5000])  # MeV
NIST_STOPPING_POWER = np.array([9.369, 7.140, 4.397, 2.683, 2.153, 1.960, 1.970])  # MeV·cm²/g

# Convert NIST to MeV/mm in EJ-200
# dE/dx [MeV/mm] = S [MeV·cm²/g] × ρ [g/cm³] × 0.1 [cm/mm]
NIST_DEDX_MEV_PER_MM = NIST_STOPPING_POWER * EJ200_DENSITY * 0.1

# Create interpolation function for any energy
_nist_interp = interp1d(NIST_ENERGIES, NIST_DEDX_MEV_PER_MM, 
                        kind='linear', fill_value='extrapolate')


def get_nist_dEdx(energy_MeV):
    """Get NIST-based dE/dx for given energy (MeV/mm)."""
    return float(_nist_interp(energy_MeV))


def get_nist_range(energy_MeV):
    """Estimate CSDA range from NIST dE/dx by integration."""
    if energy_MeV < 1:
        return 0.0
    
    # Simple integration using trapezoidal rule
    energies = np.linspace(1, energy_MeV, 1000)
    dEdx_values = _nist_interp(energies)
    
    # Range = ∫(1/dE/dx) dE
    # Filter out invalid values
    valid = (dEdx_values > 0.001)
    if np.sum(valid) < 2:
        return 0.0
    
    range_mm = np.trapz(1.0 / dEdx_values[valid], energies[valid])
    return range_mm


# ============================================================================
# PHYSICS CALCULATIONS (Verified formulas)
# ============================================================================

def calculate_beta(energy_MeV):
    """Calculate relativistic β = v/c from kinetic energy."""
    gamma = (energy_MeV + M_PROTON) / M_PROTON
    beta = np.sqrt(1 - 1/gamma**2)
    return beta


def calculate_gamma(energy_MeV):
    """Calculate Lorentz factor γ = E_total/m_0."""
    return (energy_MeV + M_PROTON) / M_PROTON


def calculate_momentum(energy_MeV):
    """Calculate momentum p = √(E² + 2Em) in MeV/c."""
    total_energy = energy_MeV + M_PROTON
    momentum = np.sqrt(total_energy**2 - M_PROTON**2)
    return momentum


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_energy_from_filename(filename):
    """Extract energy value from filename like 'EnergyDepZ_70MeV_Res.csv'."""
    basename = os.path.basename(filename)
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
                    z_bin = int(parts[2].strip())
                    energy_dep = float(parts[3].strip())
                    data.append({
                        'z_bin': z_bin,
                        'energy_deposit_MeV': energy_dep
                    })
                except ValueError:
                    continue
    return pd.DataFrame(data)


def calculate_dose_Gy(energy_deposit_MeV, volume_mm3, density_g_cm3, n_histories):
    """
    Calculate absorbed dose in Gray.
    
    Dose [Gy] = Energy [J] / Mass [kg]
    
    1 MeV = 1.602e-13 J
    Mass = Volume × Density
    """
    # Convert MeV to Joules
    energy_J = energy_deposit_MeV * 1.602e-13
    
    # Convert volume from mm³ to cm³
    volume_cm3 = volume_mm3 / 1000.0
    
    # Mass in kg
    mass_kg = volume_cm3 * density_g_cm3 / 1000.0
    
    # Dose per history
    dose_per_history = energy_J / mass_kg if mass_kg > 0 else 0.0
    
    # Average dose per particle
    return dose_per_history / n_histories


def process_all_files(runs_dir):
    """Process all EnergyDepZ files and create master dataset."""
    pattern = os.path.join(runs_dir, 'EnergyDepZ_*MeV_Res.csv')
    files = glob.glob(pattern)
    
    all_data = []
    
    # Bin volume: width × height × depth = 50mm × 50mm × 0.1mm = 250 mm³
    bin_volume_mm3 = WIDTH_MM * HEIGHT_MM * BIN_SIZE_MM
    
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
        
        # Calculate depth in mm (center of bin)
        df['depth_mm'] = (df['z_bin'] + 0.5) * BIN_SIZE_MM
        
        # Calculate physics features
        df['beta'] = calculate_beta(energy)
        df['gamma'] = calculate_gamma(energy)
        df['momentum_MeV_c'] = calculate_momentum(energy)
        
        # Calculate simulated dE/dx (per proton, per mm)
        # This is TOTAL energy deposit (including secondaries)
        df['sim_dEdx_MeV_per_mm'] = df['energy_deposit_MeV'] / (N_HISTORIES * BIN_SIZE_MM)
        
        # NIST reference dE/dx (primary proton only)
        df['nist_dEdx_MeV_per_mm'] = get_nist_dEdx(energy)
        
        # CORRECTED dose calculation
        df['dose_Gy'] = df['energy_deposit_MeV'].apply(
            lambda edep: calculate_dose_Gy(edep, bin_volume_mm3, EJ200_DENSITY, N_HISTORIES)
        )
        
        # NIST-based range
        total_range = get_nist_range(energy)
        df['nist_range_mm'] = total_range
        df['residual_range_mm'] = (total_range - df['depth_mm']).clip(lower=0)
        
        all_data.append(df)
    
    # Combine all data
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by energy and depth
    master_df = master_df.sort_values(['energy_MeV', 'depth_mm'])
    
    # Reorder columns
    columns_order = [
        'energy_MeV', 'z_bin', 'depth_mm', 
        'energy_deposit_MeV', 'sim_dEdx_MeV_per_mm', 'dose_Gy',
        'beta', 'gamma', 'momentum_MeV_c',
        'nist_dEdx_MeV_per_mm', 'nist_range_mm', 'residual_range_mm'
    ]
    master_df = master_df[columns_order]
    
    return master_df


def create_verified_material_properties(output_dir):
    """Create JSON with VERIFIED material properties from Eljen datasheet."""
    properties = {
        "material": "EJ-200",
        "manufacturer": "Eljen Technology",
        "datasheet_url": "https://eljentechnology.com/products/plastic-scintillators/ej-200",
        "verification_date": "2025-12-27",
        "properties": {
            "density_g_per_cm3": EJ200_DENSITY,
            "density_source": "Eljen Technology Datasheet",
            "composition": {
                "base": "polyvinyltoluene (PVT)",
                "hydrogen_atoms_per_cm3": 5.17e22,
                "carbon_atoms_per_cm3": 4.69e22,
                "hydrogen_weight_fraction": H_FRACTION_BY_WEIGHT,
                "carbon_weight_fraction": C_FRACTION_BY_WEIGHT,
                "H_to_C_ratio": 5.17 / 4.69
            },
            "mean_excitation_energy_eV": MEAN_EXCITATION_EV,
            "mean_excitation_source": "NIST (polyvinyltoluene)"
        },
        "optical_properties": {
            "light_yield_photons_per_MeV": EJ200_LIGHT_YIELD,
            "light_output_percent_anthracene": 64,
            "peak_emission_wavelength_nm": EJ200_PEAK_WAVELENGTH,
            "decay_time_ns": EJ200_DECAY_TIME,
            "refractive_index": EJ200_REFRACTIVE_INDEX,
            "attenuation_length_cm": EJ200_ATTENUATION_LENGTH
        },
        "geometry": {
            "thickness_mm": THICKNESS_MM,
            "width_mm": WIDTH_MM,
            "height_mm": HEIGHT_MM
        },
        "nist_reference": {
            "source": "NIST PSTAR Database",
            "url": "https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html",
            "material": "Polystyrene (closest match)",
            "energies_MeV": NIST_ENERGIES.tolist(),
            "stopping_power_MeV_cm2_per_g": NIST_STOPPING_POWER.tolist()
        }
    }
    
    with open(os.path.join(output_dir, 'material_properties_VERIFIED.json'), 'w') as f:
        json.dump(properties, f, indent=2)
    
    print(f"Verified material properties saved")


def create_train_val_test_splits(df, output_dir):
    """Create train/validation/test splits."""
    train_energies = [70, 150, 250, 500, 1000, 2000, 4500, 6000]
    val_energies = [100, 300, 1500, 3000]
    test_energies = [200, 750]
    
    train_df = df[df['energy_MeV'].isin(train_energies)]
    val_df = df[df['energy_MeV'].isin(val_energies)]
    test_df = df[df['energy_MeV'].isin(test_energies)]
    
    train_df.to_csv(os.path.join(output_dir, 'train_split_v2.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_split_v2.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split_v2.csv'), index=False)
    
    print(f"Splits created: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")


def main():
    """Main execution with all fixes applied."""
    base_dir = Path(__file__).parent
    runs_dir = base_dir / 'runs'
    output_dir = base_dir / 'pinn_data'
    
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("KAVACH BL4S - CORRECTED Data Extraction v2")
    print("=" * 60)
    print("\nFixes applied:")
    print("  1. Dose calculation corrected (was showing 0.0)")
    print("  2. Using NIST dE/dx reference (not custom Bethe-Bloch)")
    print("  3. Column names clarified (sim vs nist)")
    print("  4. Material properties VERIFIED from Eljen datasheet")
    print("  5. CSDA range from NIST integration")
    print()
    
    # Process all files
    print("[1/4] Extracting data with corrections...")
    master_df = process_all_files(runs_dir)
    
    # Save master dataset
    master_path = output_dir / 'pinn_training_data_v2.csv'
    master_df.to_csv(master_path, index=False)
    print(f"\nMaster dataset saved: {master_path}")
    print(f"  Total rows: {len(master_df)}")
    
    # Create splits
    print("\n[2/4] Creating train/val/test splits...")
    create_train_val_test_splits(master_df, output_dir)
    
    # Create verified material properties
    print("\n[3/4] Creating verified material properties...")
    create_verified_material_properties(output_dir)
    
    # Summary statistics
    print("\n[4/4] Summary Statistics")
    print("=" * 60)
    
    summary = master_df.groupby('energy_MeV').agg({
        'energy_deposit_MeV': 'sum',
        'sim_dEdx_MeV_per_mm': 'mean',
        'dose_Gy': 'mean',
        'beta': 'first',
        'gamma': 'first',
        'nist_dEdx_MeV_per_mm': 'first',
        'nist_range_mm': 'first'
    }).round(6)
    
    summary.columns = ['Total_Edep', 'Sim_dEdx', 'Dose_Gy', 
                       'Beta', 'Gamma', 'NIST_dEdx', 'NIST_Range']
    
    print(summary.to_string())
    
    # Save summary
    summary.to_csv(output_dir / 'energy_summary_v2.csv')
    
    # Verify dose is not zero
    print(f"\n--- Dose Check ---")
    print(f"Min dose: {master_df['dose_Gy'].min():.2e} Gy")
    print(f"Max dose: {master_df['dose_Gy'].max():.2e} Gy")
    print(f"Mean dose: {master_df['dose_Gy'].mean():.2e} Gy")
    
    if master_df['dose_Gy'].min() > 0:
        print("✓ Dose calculation FIXED (no longer zero)")
    else:
        print("✗ WARNING: Some dose values are still zero")
    
    print("\n" + "=" * 60)
    print("CORRECTED data extraction complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
