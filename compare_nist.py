#!/usr/bin/env python3
"""
Compare simulation dE/dx with NIST PSTAR reference data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

BASE_DIR = Path(__file__).parent
PINN_DIR = BASE_DIR / 'pinn_data'
REF_DIR = PINN_DIR / 'reference'
PLOTS_DIR = PINN_DIR / 'plots'

# EJ-200 density
DENSITY_EJ200 = 1.023  # g/cm³
DENSITY_PS = 1.060     # g/cm³ (polystyrene)

def load_nist_data():
    """Load NIST PSTAR data (real data from NIST website)."""
    nist = pd.read_csv(REF_DIR / 'nist_pstar_polystyrene.csv', comment='#')
    
    # Convert stopping power to MeV/mm for EJ-200
    # NIST gives MeV·cm²/g, convert to MeV/mm:
    # dE/dx [MeV/mm] = S [MeV·cm²/g] × ρ [g/cm³] × 0.1 [cm/mm]
    nist['dEdx_MeV_per_mm'] = nist['Total_Stopping_Power_MeV_cm2_per_g'] * DENSITY_EJ200 * 0.1
    
    return nist

def load_simulation_data():
    """Load simulation summary."""
    summary = pd.read_csv(PINN_DIR / 'energy_summary.csv')
    return summary

def compare_stopping_power():
    """Compare simulation with NIST."""
    nist = load_nist_data()
    sim = load_simulation_data()
    
    # Interpolate NIST to simulation energies
    nist_interp = interp1d(nist['Energy_MeV'], nist['dEdx_MeV_per_mm'], 
                           kind='linear', fill_value='extrapolate')
    
    sim['nist_dEdx'] = nist_interp(sim['energy_MeV'])
    sim['ratio'] = sim['Mean_dEdx'] / sim['nist_dEdx']
    sim['diff_pct'] = (sim['Mean_dEdx'] - sim['nist_dEdx']) / sim['nist_dEdx'] * 100
    
    print("=" * 70)
    print("NIST PSTAR Comparison")
    print("=" * 70)
    print(f"\n{'Energy':>8} {'TOPAS':>10} {'NIST':>10} {'Ratio':>8} {'Diff%':>8}")
    print("-" * 50)
    
    for _, row in sim.iterrows():
        print(f"{row['energy_MeV']:>8.0f} {row['Mean_dEdx']:>10.4f} "
              f"{row['nist_dEdx']:>10.4f} {row['ratio']:>8.3f} {row['diff_pct']:>+8.1f}%")
    
    print("-" * 50)
    print(f"Mean ratio: {sim['ratio'].mean():.3f} ± {sim['ratio'].std():.3f}")
    print(f"Mean diff: {sim['diff_pct'].mean():+.1f}% ± {sim['diff_pct'].std():.1f}%")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: dE/dx vs Energy
    ax1 = axes[0]
    ax1.loglog(nist['Energy_MeV'], nist['dEdx_MeV_per_mm'], 'b-', 
               linewidth=2, label='NIST PSTAR', alpha=0.7)
    ax1.loglog(sim['energy_MeV'], sim['Mean_dEdx'], 'ro', 
               markersize=10, label='TOPAS Simulation')
    ax1.set_xlabel('Energy (MeV)', fontsize=12)
    ax1.set_ylabel('dE/dx (MeV/mm)', fontsize=12)
    ax1.set_title('Stopping Power: TOPAS vs NIST', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(10, 10000)
    
    # Right: Ratio vs Energy
    ax2 = axes[1]
    ax2.semilogx(sim['energy_MeV'], sim['ratio'], 'go-', markersize=10, linewidth=2)
    ax2.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Perfect agreement')
    ax2.axhspan(0.9, 1.1, alpha=0.2, color='green', label='±10% band')
    ax2.set_xlabel('Energy (MeV)', fontsize=12)
    ax2.set_ylabel('TOPAS / NIST', fontsize=12)
    ax2.set_title('Validation: Simulation / Reference Ratio', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.5)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'nist_comparison.png', dpi=150)
    plt.close()
    
    print(f"\nPlot saved: {PLOTS_DIR / 'nist_comparison.png'}")
    
    # Save comparison table
    sim.to_csv(PINN_DIR / 'nist_comparison.csv', index=False)
    print(f"Data saved: {PINN_DIR / 'nist_comparison.csv'}")
    
    return sim

if __name__ == '__main__':
    compare_stopping_power()
