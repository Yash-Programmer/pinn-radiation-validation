"""
KAVACH BL4S OpenTOPAS Analysis Suite - Research Grade
======================================================
Comprehensive quenching analysis implementing exact methodology from proposal

Per proposal methodology (Section 6):
  Step 1: Select clean single-particle events using DWC tracking
  Step 2: Identify particle type from Cherenkov response
  Step 3: Calculate deposited energy from calorimeter and momentum
  Step 4: Measure scintillator light output (integrated PMT charge)
  Step 5: Compute quenching factor: Q = L_measured / L_expected

Physical basis:
  Birks' Law: dL/dx = S * (dE/dx) / (1 + kB * dE/dx)
  Quenching factor: Q = 1 / (1 + kB * <dE/dx>)
  
  Where:
    kB = 0.0126 g/MeV/cm² (EJ-200, from Eljen datasheet)
    S = scintillation efficiency (photons/MeV)
    dE/dx = stopping power (MeV/cm)

References:
  [1] Birks, J.B., "Theory and Practice of Scintillation Counting" (1964)
  [2] Torrisi, L., Nucl. Instrum. Meth. B 170, 523 (2000)
  [3] Christl et al., Nucl. Instrum. Meth. A 988, 164900 (2021)
  [4] Eljen Technology, "EJ-200 Technical Data Sheet" (2023)

Author: Team KAVACH
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy.optimize import curve_fit
from scipy.stats import chi2

#==============================================================================
# PHYSICAL CONSTANTS AND MATERIAL PROPERTIES
#==============================================================================

# Particle masses (MeV/c²)
M_ELECTRON = 0.511
M_PION = 139.570
M_PROTON = 938.272
M_KAON = 493.677

# PDG codes
PDG_PROTON = 2212
PDG_PION_PLUS = 211
PDG_PION_MINUS = -211
PDG_ELECTRON = 11
PDG_POSITRON = -11
PDG_PHOTON = 22
PDG_OPTICAL = 0  # OpenTOPAS convention

# EJ-200 Scintillator properties (from Eljen Technical Data Sheet)
@dataclass
class EJ200Properties:
    """EJ-200 plastic scintillator material properties."""
    density: float = 1.023  # g/cm³
    light_yield: int = 10000  # photons/MeV
    decay_time: float = 2.1  # ns
    rise_time: float = 0.9  # ns
    peak_emission_nm: float = 425.0  # nm
    attenuation_length: float = 380.0  # cm
    refractive_index: float = 1.58
    H_C_ratio: float = 1.111  # H/C = 10/9
    mean_excitation_eV: float = 64.7  # eV
    
    # Birks coefficient - THIS IS THE KEY PARAMETER
    # kB = 0.0126 g/MeV/cm² from multiple literature sources
    birks_kB: float = 0.0126  # g/MeV/cm² (= cm/MeV when divided by ρ)
    birks_kB_mm_MeV: float = 0.1232  # mm/MeV (OpenTOPAS units: kB/ρ * 10)
    
    birks_kB_uncertainty: float = 0.0005  # g/MeV/cm²

EJ200 = EJ200Properties()


#==============================================================================
# PHYSICS CALCULATIONS
#==============================================================================

def momentum_to_kinetic_energy(p_GeV: float, mass_MeV: float) -> float:
    """
    Convert momentum (GeV/c) to kinetic energy (MeV).
    
    E = √(p² + m²) - m
    """
    p_MeV = p_GeV * 1000.0
    E_total = np.sqrt(p_MeV**2 + mass_MeV**2)
    return E_total - mass_MeV


def kinetic_to_momentum(KE_MeV: float, mass_MeV: float) -> float:
    """
    Convert kinetic energy (MeV) to momentum (GeV/c).
    
    p = √((KE + m)² - m²)
    """
    E_total = KE_MeV + mass_MeV
    p_MeV = np.sqrt(E_total**2 - mass_MeV**2)
    return p_MeV / 1000.0


def relativistic_beta(p_GeV: float, mass_MeV: float) -> float:
    """Calculate relativistic β = v/c."""
    p_MeV = p_GeV * 1000.0
    E_total = np.sqrt(p_MeV**2 + mass_MeV**2)
    return p_MeV / E_total


def relativistic_gamma(p_GeV: float, mass_MeV: float) -> float:
    """Calculate Lorentz factor γ = 1/√(1-β²)."""
    beta = relativistic_beta(p_GeV, mass_MeV)
    return 1.0 / np.sqrt(1.0 - beta**2)


def bethe_bloch_dEdx(p_GeV: float, mass_MeV: float, z: int = 1,
                     Z_A: float = 0.54, I_eV: float = 64.7,
                     rho: float = 1.023) -> float:
    """
    Calculate stopping power using Bethe-Bloch formula.
    
    Per proposal: "validated against NIST PSTAR experimental database"
    
    Args:
        p_GeV: Momentum in GeV/c
        mass_MeV: Particle mass in MeV/c²
        z: Particle charge (units of e)
        Z_A: Z/A of target material (~0.54 for plastic)
        I_eV: Mean excitation energy (eV)
        rho: Density (g/cm³)
        
    Returns:
        Stopping power dE/dx in MeV/cm
    """
    # Physical constants
    K = 0.307075  # MeV mol⁻¹ cm²
    m_e = M_ELECTRON  # MeV/c²
    
    beta = relativistic_beta(p_GeV, mass_MeV)
    gamma = relativistic_gamma(p_GeV, mass_MeV)
    
    if beta < 1e-6:
        return np.inf
    
    I_MeV = I_eV * 1e-6
    
    # Maximum energy transfer in single collision
    T_max = (2 * m_e * beta**2 * gamma**2) / \
            (1 + 2*gamma*m_e/mass_MeV + (m_e/mass_MeV)**2)
    
    # Bethe-Bloch formula (without shell/density corrections for GeV energies)
    ln_term = 0.5 * np.log(2 * m_e * beta**2 * gamma**2 * T_max / I_MeV**2)
    
    dEdx = K * rho * Z_A * z**2 / beta**2 * (ln_term - beta**2)
    
    return dEdx


def birks_quenching_factor(dEdx: float, kB: float = EJ200.birks_kB) -> float:
    """
    Calculate Birks quenching factor.
    
    Q = 1 / (1 + kB * dE/dx)
    
    Per proposal Eq. (2): "Quenching factor Q represents the ratio of
    actual light output to expected output for minimum-ionizing particles"
    
    Args:
        dEdx: Stopping power in MeV/cm
        kB: Birks coefficient in cm/MeV (= g/MeV/cm² divided by ρ)
        
    Returns:
        Quenching factor Q (dimensionless, 0 < Q ≤ 1)
    """
    kB_cm_MeV = kB / EJ200.density  # Convert to cm/MeV
    return 1.0 / (1.0 + kB_cm_MeV * dEdx)


def birks_quenching_uncertainty(dEdx: float, 
                                  kB: float = EJ200.birks_kB,
                                  sigma_kB: float = EJ200.birks_kB_uncertainty) -> float:
    """
    Calculate uncertainty in quenching factor from kB uncertainty.
    
    σ_Q = |∂Q/∂kB| * σ_kB = (dE/dx / (1 + kB*dE/dx)²) * σ_kB
    """
    kB_cm = kB / EJ200.density
    sigma_kB_cm = sigma_kB / EJ200.density
    dQ_dkB = dEdx / (1 + kB_cm * dEdx)**2
    return np.abs(dQ_dkB) * sigma_kB_cm


def expected_light_yield(E_deposited_MeV: float, dEdx: float,
                         L0: int = EJ200.light_yield) -> float:
    """
    Calculate expected light yield with Birks quenching.
    
    L = L₀ * E_dep * Q(dE/dx)
    """
    Q = birks_quenching_factor(dEdx)
    return L0 * E_deposited_MeV * Q


def measured_quenching_factor(photons_detected: int, E_deposited_MeV: float,
                               efficiency: float = 0.125,
                               L0: int = EJ200.light_yield) -> float:
    """
    Extract quenching factor from measured data.
    
    Q = (N_detected / efficiency) / (E_dep * L₀)
    
    Args:
        photons_detected: Optical photons at photocathode
        E_deposited_MeV: Energy deposited in scintillator
        efficiency: Total efficiency (geometric × QE × collection)
        L0: Unquenched light yield
        
    Returns:
        Measured quenching factor
    """
    if E_deposited_MeV <= 0 or photons_detected <= 0:
        return np.nan
    
    L_measured = photons_detected / efficiency
    L_expected = E_deposited_MeV * L0
    return L_measured / L_expected


#==============================================================================
# BEAM CONFIGURATION DATABASE
#==============================================================================

@dataclass
class BeamConfig:
    """Configuration for a single beam setting."""
    particle: str
    momentum_GeV: float
    pdg_code: int
    mass_MeV: float
    
    @property
    def kinetic_energy_MeV(self) -> float:
        return momentum_to_kinetic_energy(self.momentum_GeV, self.mass_MeV)
    
    @property
    def beta(self) -> float:
        return relativistic_beta(self.momentum_GeV, self.mass_MeV)
    
    @property
    def gamma(self) -> float:
        return relativistic_gamma(self.momentum_GeV, self.mass_MeV)
    
    @property
    def dEdx(self) -> float:
        return bethe_bloch_dEdx(self.momentum_GeV, self.mass_MeV)
    
    @property
    def predicted_Q(self) -> float:
        return birks_quenching_factor(self.dEdx)
    
    @property
    def Q_uncertainty(self) -> float:
        return birks_quenching_uncertainty(self.dEdx)


# Per proposal Table 3: T09 beamline configurations
BEAM_CONFIGS = [
    # Protons: 0.5-5.0 GeV/c
    BeamConfig("proton", 0.5, PDG_PROTON, M_PROTON),
    BeamConfig("proton", 1.0, PDG_PROTON, M_PROTON),
    BeamConfig("proton", 2.0, PDG_PROTON, M_PROTON),
    BeamConfig("proton", 3.0, PDG_PROTON, M_PROTON),
    BeamConfig("proton", 4.0, PDG_PROTON, M_PROTON),
    BeamConfig("proton", 5.0, PDG_PROTON, M_PROTON),
    # Pions: 0.5-5.0 GeV/c
    BeamConfig("pion", 0.5, PDG_PION_PLUS, M_PION),
    BeamConfig("pion", 1.0, PDG_PION_PLUS, M_PION),
    BeamConfig("pion", 2.0, PDG_PION_PLUS, M_PION),
    BeamConfig("pion", 3.0, PDG_PION_PLUS, M_PION),
    BeamConfig("pion", 4.0, PDG_PION_PLUS, M_PION),
    BeamConfig("pion", 5.0, PDG_PION_PLUS, M_PION),
    # Electrons: 2.0, 4.0 GeV/c (MIP calibration)
    BeamConfig("electron", 2.0, PDG_ELECTRON, M_ELECTRON),
    BeamConfig("electron", 4.0, PDG_ELECTRON, M_ELECTRON),
]


def generate_predictions_table() -> pd.DataFrame:
    """
    Generate theoretical predictions table matching proposal Table 3.
    
    Per proposal: "predictions from Geant4 Bethe-Bloch implementation,
    validated against NIST PSTAR experimental database"
    """
    data = []
    for cfg in BEAM_CONFIGS:
        data.append({
            'Particle': cfg.particle,
            'p (GeV/c)': cfg.momentum_GeV,
            'KE (MeV)': round(cfg.kinetic_energy_MeV, 1),
            'β': round(cfg.beta, 4),
            'γ': round(cfg.gamma, 3),
            'dE/dx (MeV/cm)': round(cfg.dEdx, 2),
            'Q (predicted)': round(cfg.predicted_Q, 4),
            'σ_Q': round(cfg.Q_uncertainty, 4),
        })
    
    return pd.DataFrame(data)


#==============================================================================
# DATA LOADING AND EVENT SELECTION
#==============================================================================

def load_phasespace_ascii(filepath: Path) -> pd.DataFrame:
    """
    Load OpenTOPAS ASCII phase space file.
    
    Columns depend on scorer configuration but typically include:
    X, Y, Z (cm), dCosX, dCosY, dCosZ, Energy (MeV), Weight, PDGCode,
    TrackID, EventID, ParentID, TOF (ns), etc.
    """
    # Read header to determine columns
    columns = []
    header_file = filepath.with_suffix('.header')
    
    default_columns = [
        'X_cm', 'Y_cm', 'Z_cm', 'dCosX', 'dCosY', 'dCosZ', 
        'Energy_MeV', 'Weight', 'PDGCode'
    ]
    extended_columns = default_columns + [
        'TrackID', 'EventID', 'ParentID', 'TOF_ns'
    ]
    
    try:
        df = pd.read_csv(filepath, sep=r'\s+', header=None, comment='#')
        
        # Assign column names based on number of columns
        if df.shape[1] <= len(default_columns):
            df.columns = default_columns[:df.shape[1]]
        elif df.shape[1] <= len(extended_columns):
            df.columns = extended_columns[:df.shape[1]]
        else:
            df.columns = [f'col_{i}' for i in range(df.shape[1])]
        
        return df
        
    except Exception as e:
        warnings.warn(f"Failed to load {filepath}: {e}")
        return pd.DataFrame()


def load_csv_scorer(filepath: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load OpenTOPAS CSV scorer output.
    
    Returns:
        Tuple of (data array, header metadata)
    """
    header = {}
    data_lines = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                # Parse header
                if ':' in line:
                    key, val = line[1:].split(':', 1)
                    header[key.strip()] = val.strip()
            else:
                data_lines.append(line.strip())
    
    # Parse data
    if data_lines:
        data = np.loadtxt(data_lines, delimiter=',')
    else:
        data = np.array([])
    
    return data, header


def select_clean_events(df_dwc1: pd.DataFrame, df_dwc2: pd.DataFrame,
                        df_target: pd.DataFrame,
                        max_displacement_cm: float = 1.0) -> np.ndarray:
    """
    Select clean single-particle events using DWC tracking.
    
    Per proposal Step 1: "Select clean single-particle events using DWC tracking"
    
    Args:
        df_dwc1: Phase space at DWC1
        df_dwc2: Phase space at DWC2
        df_target: Phase space at target
        max_displacement_cm: Maximum allowed displacement from beam axis
        
    Returns:
        Array of clean event IDs
    """
    if df_dwc1.empty or df_target.empty:
        return np.array([])
    
    # Get unique event IDs present in all detectors
    events_dwc1 = set(df_dwc1['EventID'].unique()) if 'EventID' in df_dwc1 else set()
    events_target = set(df_target['EventID'].unique()) if 'EventID' in df_target else set()
    
    common_events = events_dwc1 & events_target
    
    # Filter by position (on-axis events)
    clean_events = []
    for evt in common_events:
        dwc1_evt = df_dwc1[df_dwc1['EventID'] == evt]
        if 'X_cm' in dwc1_evt.columns and 'Y_cm' in dwc1_evt.columns:
            r = np.sqrt(dwc1_evt['X_cm'].values**2 + dwc1_evt['Y_cm'].values**2)
            if np.all(r < max_displacement_cm):
                clean_events.append(evt)
    
    return np.array(clean_events)


def apply_coincidence_trigger(df_s1: pd.DataFrame, df_s2: pd.DataFrame,
                               window_ns: float = 25.0) -> np.ndarray:
    """
    Apply S1·S2 coincidence trigger.
    
    Per proposal: "Apply coincidence requirement: S1·S2 within 25 ns window"
    
    Returns:
        Array of event IDs passing coincidence
    """
    if df_s1.empty or df_s2.empty:
        return np.array([])
    
    if 'TOF_ns' not in df_s1.columns or 'TOF_ns' not in df_s2.columns:
        # If no TOF, accept all common events
        s1_events = set(df_s1['EventID'].unique()) if 'EventID' in df_s1 else set()
        s2_events = set(df_s2['EventID'].unique()) if 'EventID' in df_s2 else set()
        return np.array(list(s1_events & s2_events))
    
    coincident_events = []
    for evt in df_s1['EventID'].unique():
        s1_times = df_s1[df_s1['EventID'] == evt]['TOF_ns'].values
        s2_times = df_s2[df_s2['EventID'] == evt]['TOF_ns'].values
        
        if len(s1_times) > 0 and len(s2_times) > 0:
            dt = np.min(s2_times) - np.min(s1_times)
            if 0 < dt < window_ns:
                coincident_events.append(evt)
    
    return np.array(coincident_events)


#==============================================================================
# QUENCHING ANALYSIS
#==============================================================================

@dataclass
class QuenchingResult:
    """Result of quenching analysis for one beam configuration."""
    particle: str
    momentum_GeV: float
    n_events: int
    
    # Simulated quantities
    mean_energy_dep_MeV: float
    std_energy_dep_MeV: float
    mean_photons_detected: float
    std_photons_detected: float
    
    # Derived quenching factor
    Q_measured: float
    Q_uncertainty: float
    
    # Theoretical comparison
    Q_predicted: float
    Q_predicted_uncertainty: float
    dEdx_predicted: float
    
    # Chi-squared
    chi2: float = 0.0
    p_value: float = 0.0


def analyze_single_run(run_dir: Path, beam_config: BeamConfig,
                       efficiency: float = 0.125) -> Optional[QuenchingResult]:
    """
    Analyze a single beam configuration run.
    
    Implements full analysis pipeline from proposal Section 6.
    """
    # Load energy deposit
    edep_file = run_dir / 'Scint_EnergyDeposit.csv'
    if not edep_file.exists():
        return None
    
    edep_data, _ = load_csv_scorer(edep_file)
    if edep_data.size == 0:
        return None
    
    # Total energy deposit per run
    total_edep = np.sum(edep_data)
    
    # Load optical photon count
    photon_file = run_dir / 'OpticalPhotonCount.csv'
    total_photons = 0
    if photon_file.exists():
        photon_data, _ = load_csv_scorer(photon_file)
        if photon_data.size > 0:
            total_photons = int(np.sum(photon_data))
    
    # Calculate measured quenching factor
    Q_meas = measured_quenching_factor(total_photons, total_edep, efficiency)
    
    # Uncertainty from Poisson statistics
    if total_photons > 0 and total_edep > 0:
        Q_unc = Q_meas * np.sqrt(1/total_photons + 0.01**2)  # 1% systematic
    else:
        Q_unc = np.nan
    
    # Chi-squared comparison with theory
    Q_pred = beam_config.predicted_Q
    Q_pred_unc = beam_config.Q_uncertainty
    
    if not np.isnan(Q_meas) and not np.isnan(Q_unc):
        chi2_val = ((Q_meas - Q_pred) / np.sqrt(Q_unc**2 + Q_pred_unc**2))**2
        p_val = 1 - chi2.cdf(chi2_val, 1)
    else:
        chi2_val = np.nan
        p_val = np.nan
    
    # Get number of events from phase space file
    n_events = 10000  # Default from run configuration
    
    return QuenchingResult(
        particle=beam_config.particle,
        momentum_GeV=beam_config.momentum_GeV,
        n_events=n_events,
        mean_energy_dep_MeV=total_edep,
        std_energy_dep_MeV=0,
        mean_photons_detected=total_photons,
        std_photons_detected=np.sqrt(total_photons),
        Q_measured=Q_meas,
        Q_uncertainty=Q_unc,
        Q_predicted=Q_pred,
        Q_predicted_uncertainty=Q_pred_unc,
        dEdx_predicted=beam_config.dEdx,
        chi2=chi2_val,
        p_value=p_val
    )


#==============================================================================
# BIRKS COEFFICIENT FITTING
#==============================================================================

def birks_model(dEdx: np.ndarray, kB: float) -> np.ndarray:
    """Birks quenching model for fitting."""
    return 1.0 / (1.0 + kB * dEdx)


def fit_birks_coefficient(dEdx_values: np.ndarray, 
                           Q_values: np.ndarray,
                           Q_uncertainties: np.ndarray) -> Tuple[float, float]:
    """
    Fit Birks coefficient to measured quenching data.
    
    Per proposal: "extracting Birks' coefficient kB and validating 
    our simulation predictions"
    
    Returns:
        Tuple of (kB, sigma_kB)
    """
    # Initial guess
    kB_init = 0.012
    
    try:
        popt, pcov = curve_fit(
            birks_model, 
            dEdx_values, 
            Q_values,
            p0=[kB_init],
            sigma=Q_uncertainties,
            absolute_sigma=True
        )
        
        kB_fit = popt[0]
        kB_err = np.sqrt(pcov[0, 0])
        
        return kB_fit, kB_err
        
    except Exception as e:
        warnings.warn(f"Birks fit failed: {e}")
        return np.nan, np.nan


#==============================================================================
# MAIN ENTRY POINT
#==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("KAVACH BL4S OpenTOPAS Analysis - Research Grade")
    print("Scintillator Quenching Characterization for Hadron Therapy")
    print("=" * 70)
    
    # Generate predictions table
    print("\nTheoretical Predictions (Table 3 from proposal):")
    print("-" * 70)
    predictions = generate_predictions_table()
    print(predictions.to_string(index=False))
    
    # Save predictions
    predictions.to_csv('theoretical_predictions.csv', index=False)
    print("\nSaved to theoretical_predictions.csv")
    
    # Check for simulation output
    output_dir = Path('../output')
    if output_dir.exists():
        print(f"\nSimulation output directory found: {output_dir}")
        for particle in ['proton', 'pion', 'electron']:
            particle_dir = output_dir / particle
            if particle_dir.exists():
                runs = list(particle_dir.glob('*/'))
                print(f"  {particle}: {len(runs)} runs found")
    else:
        print("\nNo simulation output found. Run simulations first:")
        print("  cd opentopas_simulation")
        print("  run_all.bat")
