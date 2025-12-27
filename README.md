# KAVACH BL4S OpenTOPAS Simulation Suite

## Scintillator Quenching Characterization for Hadron Therapy Dosimetry

This simulation suite implements a research-grade Monte Carlo study of scintillator quenching
using OpenTOPAS (based on GEANT4) for the BL4S 2026 proposal by Team KAVACH.

---

## Quick Start

### Prerequisites

1. **OpenTOPAS 4.2.0** built and installed (see `OpenTOPAS_quickStart_*.md` guides)
2. **Python 3.8+** with numpy, pandas, matplotlib

### Running Simulations

```batch
cd opentopas_simulation
run_all.bat
```

Or run individual configurations:
```batch
topas runs\Proton_1GeV.txt
```

### Running Analysis

```bash
cd opentopas_simulation/analysis
python run_all_analysis.py
```

---

## Directory Structure

```
opentopas_simulation/
├── materials/
│   └── EJ200_Materials.txt       # Material definitions (EJ-200, CO2, LeadGlass, etc.)
├── geometry/
│   └── KAVACH_Beamline.txt       # Full T09 detector geometry
├── physics/
│   └── KAVACH_Physics.txt        # Physics list configuration
├── scorers/
│   └── KAVACH_Scorers.txt        # 21 comprehensive scorers
├── sources/
│   ├── ProtonBeam.txt            # Proton source definition
│   ├── PionBeam.txt              # Pion source definition
│   └── ElectronBeam.txt          # Electron source definition
├── runs/
│   ├── Proton_{1-5}GeV.txt       # Proton run configurations
│   ├── Pion_{1-5}GeV.txt         # Pion run configurations
│   └── Electron_{2,4}GeV.txt     # Electron calibration runs
├── output/
│   ├── proton/{1-5}GeV/          # Proton simulation outputs
│   ├── pion/{1-5}GeV/            # Pion simulation outputs
│   └── electron/{2,4}GeV/        # Electron simulation outputs
├── analysis/
│   ├── load_topas_output.py      # Data loading utilities
│   ├── birks_analysis.py         # Quenching factor analysis
│   ├── generate_figures.py       # Publication-quality plots
│   └── run_all_analysis.py       # Master analysis runner
└── run_all.bat                   # Batch execution script
```

---

## Beam Configurations

| Particle | Momenta (GeV/c) | Events | Purpose |
|----------|-----------------|--------|---------|
| Proton   | 1.0, 2.0, 3.0, 4.0, 5.0 | 10,000 | Quenching measurement |
| Pion+    | 1.0, 2.0, 3.0, 4.0, 5.0 | 10,000 | Quenching measurement |
| Electron | 2.0, 4.0 | 10,000 | MIP calibration reference |

---

## Detector Components

1. **S1 Trigger** - 100×100×5 mm³ plastic scintillator
2. **CO2 Cherenkov** - 100 cm cylinder for particle ID
3. **DWC1** - Delay wire chamber for position tracking
4. **Target EJ-200** - 50×50×10 mm³ primary measurement scintillator
5. **PMT Assembly** - Photocathode with 25% QE at 420nm
6. **DWC2** - Second position measurement
7. **Lead Glass Calorimeter** - 4×4 module array
8. **S2 Trigger** - Coincidence completion

---

## Scorers (21 total)

- Energy deposit (total, primary, secondary, by particle)
- Dose to medium
- Proton LET
- Fluence and energy fluence
- Optical photon count (scintillation + Cherenkov)
- Phase space at all detector surfaces
- Trigger counts

---

## Expected Results

Based on Birks' law with kB = 0.0126 cm/MeV:

| Particle | p (GeV/c) | dE/dx (MeV/cm) | Q (predicted) |
|----------|-----------|----------------|---------------|
| Proton   | 1.0       | ~1.9           | 0.976 ± 0.002 |
| Proton   | 3.0       | ~1.6           | 0.980 ± 0.002 |
| Pion     | 1.0       | ~1.7           | 0.979 ± 0.002 |
| Electron | 2.0       | ~1.7           | 0.979 ± 0.002 |

---

## References

- Birks, J.B., "The Theory and Practice of Scintillation Counting" (1964)
- Eljen Technology EJ-200 Data Sheet
- OpenTOPAS Documentation: https://github.com/OpenTOPAS/OpenTOPAS

---

## Team KAVACH

**K**nowledge-driven **A**nalysis for **V**erification of **A**ctive **C**ancer **H**adron dosimetry

BL4S 2026 Proposal
