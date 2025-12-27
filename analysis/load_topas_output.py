"""
KAVACH BL4S OpenTOPAS Analysis Suite
=====================================
Data loading utilities for OpenTOPAS output files
Supports: CSV, ASCII Phase Space, Binary Phase Space

Author: Team KAVACH
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import struct
import warnings


class TOPASOutputLoader:
    """
    Loader class for OpenTOPAS simulation output files.
    Handles CSV scores and ASCII/Binary phase space files.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize loader with output directory path.
        
        Args:
            output_dir: Path to OpenTOPAS output directory
        """
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            warnings.warn(f"Output directory does not exist: {output_dir}")
    
    def load_csv_scorer(self, filename: str) -> pd.DataFrame:
        """
        Load a CSV scorer output file.
        
        Args:
            filename: Name of CSV file (without path)
            
        Returns:
            DataFrame with scorer data
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            # Try with .csv extension
            filepath = self.output_dir / f"{filename}.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        # TOPAS CSV format: header lines start with #
        # Read and skip header lines
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find data start (first line not starting with #)
        data_start = 0
        header_info = {}
        for i, line in enumerate(lines):
            if line.startswith('#'):
                # Parse header info
                if ':' in line:
                    key, val = line[1:].split(':', 1)
                    header_info[key.strip()] = val.strip()
                data_start = i + 1
            else:
                break
        
        # Read data
        df = pd.read_csv(filepath, skiprows=data_start, header=None)
        
        # Add metadata
        df.attrs['header'] = header_info
        df.attrs['source_file'] = str(filepath)
        
        return df
    
    def load_phase_space_ascii(self, filename: str) -> pd.DataFrame:
        """
        Load an ASCII phase space file.
        
        Args:
            filename: Name of phase space file
            
        Returns:
            DataFrame with particle data
        """
        # Find the header file
        header_file = self.output_dir / f"{filename}.header"
        phsp_file = self.output_dir / f"{filename}.phsp"
        
        if not header_file.exists():
            header_file = self.output_dir / f"{filename.replace('.phsp', '')}.header"
        if not phsp_file.exists():
            phsp_file = self.output_dir / filename
        
        # Parse header for column information
        columns = self._parse_phsp_header(header_file) if header_file.exists() else None
        
        # Default columns if header not found
        if columns is None:
            columns = ['X', 'Y', 'Z', 'dCosX', 'dCosY', 'dCosZ', 'Energy', 
                      'Weight', 'PDGCode', 'TrackID', 'EventID', 'ParentID']
        
        # Read phase space data
        try:
            df = pd.read_csv(phsp_file, sep=r'\s+', header=None, 
                            names=columns[:len(pd.read_csv(phsp_file, sep=r'\s+', 
                                                            nrows=1).columns)])
        except Exception as e:
            raise IOError(f"Failed to read phase space file: {e}")
        
        return df
    
    def _parse_phsp_header(self, header_file: Path) -> Optional[List[str]]:
        """Parse phase space header file for column names."""
        if not header_file.exists():
            return None
        
        columns = []
        with open(header_file, 'r') as f:
            for line in f:
                if 'Columns of data' in line or 'contains' in line.lower():
                    # Extract column names from header description
                    pass
        
        return columns if columns else None
    
    def load_energy_deposit(self, particle: str, momentum: str) -> float:
        """
        Load total energy deposit from scorer output.
        
        Args:
            particle: Particle type ('proton', 'pion', 'electron')
            momentum: Momentum string (e.g., '1GeV', '2GeV')
            
        Returns:
            Total energy deposited (MeV)
        """
        output_path = self.output_dir / particle / momentum
        loader = TOPASOutputLoader(str(output_path))
        
        try:
            df = loader.load_csv_scorer('Scint_EnergyDeposit')
            # Sum column (typically column 0 for single-value scorers)
            return df.values.sum()
        except FileNotFoundError:
            return np.nan
    
    def load_optical_photon_count(self, particle: str, momentum: str) -> int:
        """
        Load optical photon count from scorer output.
        
        Args:
            particle: Particle type
            momentum: Momentum string
            
        Returns:
            Total optical photons detected
        """
        output_path = self.output_dir / particle / momentum
        loader = TOPASOutputLoader(str(output_path))
        
        try:
            df = loader.load_csv_scorer('OpticalPhotonCount')
            return int(df.values.sum())
        except FileNotFoundError:
            return 0


def load_all_runs(base_output_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load summary data from all simulation runs.
    
    Args:
        base_output_dir: Base output directory containing proton/pion/electron subdirs
        
    Returns:
        Nested dictionary: {particle: {momentum: {quantity: value}}}
    """
    base = Path(base_output_dir)
    results = {}
    
    particles = ['proton', 'pion', 'electron']
    momenta = {
        'proton': ['1GeV', '2GeV', '3GeV', '4GeV', '5GeV'],
        'pion': ['1GeV', '2GeV', '3GeV', '4GeV', '5GeV'],
        'electron': ['2GeV', '4GeV']
    }
    
    for particle in particles:
        results[particle] = {}
        for momentum in momenta[particle]:
            run_dir = base / particle / momentum
            if run_dir.exists():
                loader = TOPASOutputLoader(str(run_dir))
                results[particle][momentum] = {
                    'energy_deposit': loader.load_energy_deposit('.', ''),
                    'optical_photons': loader.load_optical_photon_count('.', '')
                }
            else:
                results[particle][momentum] = {
                    'energy_deposit': np.nan,
                    'optical_photons': 0
                }
    
    return results


if __name__ == "__main__":
    # Test loading
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "../output"
    
    loader = TOPASOutputLoader(output_dir)
    print(f"Initialized loader for: {output_dir}")
    print(f"Directory exists: {loader.output_dir.exists()}")
