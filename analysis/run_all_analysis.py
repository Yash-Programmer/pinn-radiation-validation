"""
KAVACH BL4S OpenTOPAS Analysis Suite
=====================================
Master Analysis Runner

Orchestrates the full analysis pipeline:
1. Load OpenTOPAS simulation outputs
2. Calculate quenching factors
3. Compare with theoretical predictions
4. Generate figures and reports

Author: Team KAVACH
Date: 2025

Usage:
    python run_all_analysis.py [output_dir]
    
    output_dir: Path to simulation output (default: ../output)
"""

import sys
import os
from pathlib import Path

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from load_topas_output import TOPASOutputLoader, load_all_runs
from birks_analysis import (
    generate_theoretical_predictions,
    compare_simulation_to_theory,
    calculate_quenching_from_simulation
)
from generate_figures import generate_all_figures


def run_full_analysis(output_dir: str = '../output'):
    """
    Run the complete analysis pipeline.
    
    Args:
        output_dir: Path to OpenTOPAS output directory
    """
    print("=" * 70)
    print("KAVACH BL4S OpenTOPAS Simulation Analysis")
    print("Scintillator Quenching Characterization for Hadron Therapy Dosimetry")
    print("=" * 70)
    
    output_path = Path(output_dir)
    results_path = output_path / 'analysis_results'
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate theoretical predictions
    print("\n[1/5] Generating theoretical predictions...")
    predictions = generate_theoretical_predictions()
    predictions.to_csv(results_path / 'theoretical_predictions.csv', index=False)
    print(f"      Saved: {results_path / 'theoretical_predictions.csv'}")
    
    # Step 2: Load simulation results
    print("\n[2/5] Loading simulation outputs...")
    
    sim_data = []
    particles = ['proton', 'pion', 'electron']
    momenta = {
        'proton': [1, 2, 3, 4, 5],
        'pion': [1, 2, 3, 4, 5],
        'electron': [2, 4]
    }
    
    for particle in particles:
        for p in momenta[particle]:
            run_dir = output_path / particle / f'{p}GeV'
            
            if run_dir.exists():
                loader = TOPASOutputLoader(str(run_dir))
                
                # Try to load key outputs
                energy_dep = np.nan
                photons = 0
                
                try:
                    # Look for energy deposit file
                    for csv_file in run_dir.glob('*EnergyDeposit*.csv'):
                        try:
                            df = pd.read_csv(csv_file, comment='#', header=None)
                            energy_dep = df.values.sum()
                            break
                        except:
                            pass
                    
                    # Look for optical photon count
                    for csv_file in run_dir.glob('*OpticalPhoton*.csv'):
                        try:
                            df = pd.read_csv(csv_file, comment='#', header=None)
                            photons = int(df.values.sum())
                            break
                        except:
                            pass
                            
                except Exception as e:
                    print(f"      Warning: Could not load data for {particle} {p}GeV: {e}")
                
                sim_data.append({
                    'Particle': particle,
                    'Momentum_GeV': float(p),
                    'Edep_MeV': energy_dep,
                    'Photons': photons
                })
                print(f"      Loaded: {particle} {p}GeV/c")
            else:
                print(f"      Missing: {particle} {p}GeV/c (directory not found)")
                sim_data.append({
                    'Particle': particle,
                    'Momentum_GeV': float(p),
                    'Edep_MeV': np.nan,
                    'Photons': 0
                })
    
    sim_df = pd.DataFrame(sim_data)
    
    # Step 3: Calculate quenching factors from simulation
    print("\n[3/5] Calculating quenching factors...")
    sim_df['Q_simulated'] = sim_df.apply(
        lambda row: calculate_quenching_from_simulation(
            row['Edep_MeV'], row['Photons']
        ) if not np.isnan(row['Edep_MeV']) else np.nan,
        axis=1
    )
    
    # Step 4: Compare with theory
    print("\n[4/5] Comparing simulation to theory...")
    comparison = compare_simulation_to_theory(
        {row['Particle']: {f"{int(row['Momentum_GeV'])}GeV": {
            'energy_deposit': row['Edep_MeV'],
            'optical_photons': row['Photons']
        }} for _, row in sim_df.iterrows()},
        predictions
    )
    
    comparison.to_csv(results_path / 'comparison_results.csv', index=False)
    print(f"      Saved: {results_path / 'comparison_results.csv'}")
    
    # Step 5: Generate figures
    print("\n[5/5] Generating figures...")
    figures_path = results_path / 'figures'
    generate_all_figures(predictions, comparison, str(figures_path))
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    print("\nTheoretical Predictions Summary:")
    print("-" * 50)
    for particle in ['proton', 'pion', 'electron']:
        mask = predictions['Particle'] == particle
        data = predictions[mask]
        print(f"\n{particle.upper()}:")
        print(f"  Momentum range: {data['Momentum_GeV'].min():.1f} - {data['Momentum_GeV'].max():.1f} GeV/c")
        print(f"  dE/dx range: {data['dEdx_MeV_cm'].min():.2f} - {data['dEdx_MeV_cm'].max():.2f} MeV/cm")
        print(f"  Q range: {data['Q_predicted'].min():.4f} - {data['Q_predicted'].max():.4f}")
    
    # Check if simulation data was loaded
    valid_sims = comparison['Q_simulated'].notna().sum()
    if valid_sims > 0:
        print(f"\nSimulation Results: {valid_sims} runs with valid quenching data")
        mean_diff = comparison['Q_difference'].abs().mean()
        print(f"  Mean |ΔQ| (theory-sim): {mean_diff:.4f}")
    else:
        print("\nNote: No simulation data loaded yet. Run OpenTOPAS simulations first:")
        print("  cd opentopas_simulation")
        print("  run_all.bat")
    
    print(f"\nOutput files saved to: {results_path}")
    print("=" * 70)
    
    return comparison


if __name__ == "__main__":
    # Get output directory from command line or use default
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = '../output'
    
    # Check if running from analysis directory
    if not Path(output_dir).exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        output_dir = str(script_dir.parent / 'output')
    
    run_full_analysis(output_dir)
