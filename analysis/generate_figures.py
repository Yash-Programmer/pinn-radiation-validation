"""
KAVACH BL4S OpenTOPAS Analysis Suite
=====================================
Figure Generation Module

Creates publication-quality matplotlib figures for:
- Quenching factor vs momentum
- Theory vs simulation comparison
- dE/dx validation
- Optical photon yields

Author: Team KAVACH
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from typing import Dict, List, Optional

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'lines.markersize': 8
})


def plot_quenching_vs_momentum(predictions: pd.DataFrame,
                                simulation: Optional[pd.DataFrame] = None,
                                output_file: str = 'quenching_vs_momentum.png'):
    """
    Plot quenching factor Q vs beam momentum.
    
    Args:
        predictions: DataFrame with theoretical predictions
        simulation: Optional DataFrame with simulation results
        output_file: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {'proton': '#1f77b4', 'pion': '#ff7f0e', 'electron': '#2ca02c'}
    markers = {'proton': 'o', 'pion': 's', 'electron': '^'}
    
    # Plot theoretical predictions
    for particle in ['proton', 'pion', 'electron']:
        mask = predictions['Particle'] == particle
        data = predictions[mask]
        
        ax.plot(data['Momentum_GeV'], data['Q_predicted'], 
               color=colors[particle], linestyle='--', alpha=0.7,
               label=f'{particle.capitalize()} (theory)')
        
        ax.fill_between(data['Momentum_GeV'],
                        data['Q_predicted'] - data['Q_uncertainty'],
                        data['Q_predicted'] + data['Q_uncertainty'],
                        color=colors[particle], alpha=0.2)
    
    # Plot simulation results if provided
    if simulation is not None and 'Q_simulated' in simulation.columns:
        for particle in ['proton', 'pion', 'electron']:
            mask = (simulation['Particle'] == particle) & (~simulation['Q_simulated'].isna())
            data = simulation[mask]
            
            if len(data) > 0:
                ax.scatter(data['Momentum_GeV'], data['Q_simulated'],
                          color=colors[particle], marker=markers[particle], s=100,
                          edgecolors='black', linewidth=0.5, zorder=5,
                          label=f'{particle.capitalize()} (simulation)')
    
    ax.set_xlabel('Beam Momentum (GeV/c)')
    ax.set_ylabel('Quenching Factor Q')
    ax.set_title('Scintillator Quenching Factor vs Beam Momentum\n(EJ-200 Plastic Scintillator, kB = 0.0126 cm/MeV)')
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0.95, 1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.legend(loc='lower right', ncol=2)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='No quenching')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_dEdx_vs_momentum(predictions: pd.DataFrame,
                          output_file: str = 'dEdx_vs_momentum.png'):
    """
    Plot stopping power dE/dx vs momentum.
    
    Args:
        predictions: DataFrame with dE/dx values
        output_file: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {'proton': '#1f77b4', 'pion': '#ff7f0e', 'electron': '#2ca02c'}
    
    for particle in ['proton', 'pion', 'electron']:
        mask = predictions['Particle'] == particle
        data = predictions[mask]
        
        ax.plot(data['Momentum_GeV'], data['dEdx_MeV_cm'],
               color=colors[particle], marker='o', linestyle='-',
               label=f'{particle.capitalize()}')
    
    # MIP reference line
    ax.axhline(y=1.7, color='gray', linestyle='--', alpha=0.7, 
               label='MIP (~1.7 MeV/cm)')
    
    ax.set_xlabel('Beam Momentum (GeV/c)')
    ax.set_ylabel('Stopping Power dE/dx (MeV/cm)')
    ax.set_title('Stopping Power in EJ-200 Plastic Scintillator\n(Bethe-Bloch Calculation)')
    
    ax.set_xlim(0, 6)
    ax.set_ylim(1.0, 3.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_photon_yield(simulation: pd.DataFrame,
                      output_file: str = 'photon_yield.png'):
    """
    Plot optical photon yield vs energy deposited.
    
    Args:
        simulation: DataFrame with simulation results
        output_file: Output filename
    """
    if 'Photons' not in simulation.columns or 'Edep_MeV' not in simulation.columns:
        print("Warning: Photon or energy data not available in simulation results")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {'proton': '#1f77b4', 'pion': '#ff7f0e', 'electron': '#2ca02c'}
    markers = {'proton': 'o', 'pion': 's', 'electron': '^'}
    
    valid_data = simulation.dropna(subset=['Photons', 'Edep_MeV'])
    
    for particle in ['proton', 'pion', 'electron']:
        mask = valid_data['Particle'] == particle
        data = valid_data[mask]
        
        if len(data) > 0:
            ax.scatter(data['Edep_MeV'], data['Photons'],
                      color=colors[particle], marker=markers[particle], s=100,
                      label=f'{particle.capitalize()}')
    
    ax.set_xlabel('Energy Deposited (MeV)')
    ax.set_ylabel('Optical Photons Detected')
    ax.set_title('Optical Photon Yield vs Energy Deposition')
    
    ax.legend(loc='upper left')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_comparison_table(comparison: pd.DataFrame,
                          output_file: str = 'comparison_table.png'):
    """
    Create a figure showing tabular comparison of theory vs simulation.
    
    Args:
        comparison: DataFrame with comparison data
        output_file: Output filename
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Select columns for table
    display_cols = ['Particle', 'Momentum_GeV', 'dEdx_MeV_cm', 
                   'Q_predicted', 'Q_simulated', 'Q_difference']
    
    display_data = comparison[display_cols].copy()
    display_data.columns = ['Particle', 'p (GeV/c)', 'dE/dx (MeV/cm)',
                           'Q (theory)', 'Q (sim)', 'ΔQ']
    
    # Format numeric columns
    for col in ['dE/dx (MeV/cm)', 'Q (theory)', 'Q (sim)', 'ΔQ']:
        display_data[col] = display_data[col].apply(
            lambda x: f'{x:.4f}' if pd.notna(x) else '—'
        )
    
    table = ax.table(cellText=display_data.values,
                     colLabels=display_data.columns,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header styling
    for i in range(len(display_data.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternating row colors
    for i in range(1, len(display_data) + 1):
        for j in range(len(display_data.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
    
    ax.set_title('Quenching Factor Comparison: Theory vs OpenTOPAS Simulation\n',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def generate_all_figures(predictions: pd.DataFrame,
                         simulation: Optional[pd.DataFrame] = None,
                         output_dir: str = '.'):
    """
    Generate all analysis figures.
    
    Args:
        predictions: Theoretical predictions DataFrame
        simulation: Simulation results DataFrame (optional)
        output_dir: Output directory for figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating analysis figures...")
    
    plot_quenching_vs_momentum(
        predictions, simulation,
        str(output_path / 'quenching_vs_momentum.png')
    )
    
    plot_dEdx_vs_momentum(
        predictions,
        str(output_path / 'dEdx_vs_momentum.png')
    )
    
    if simulation is not None:
        plot_photon_yield(
            simulation,
            str(output_path / 'photon_yield.png')
        )
        
        plot_comparison_table(
            simulation,
            str(output_path / 'comparison_table.png')
        )
    
    print(f"\nAll figures saved to: {output_path}")


if __name__ == "__main__":
    from birks_analysis import generate_theoretical_predictions
    
    # Generate predictions
    predictions = generate_theoretical_predictions()
    
    # Generate figures with theoretical predictions only
    generate_all_figures(predictions, output_dir='../output/figures')
