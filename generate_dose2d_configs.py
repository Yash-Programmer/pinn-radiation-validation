"""
Generate TOPAS 2D dose distribution configuration files.
KAVACH BL4S - 2D Dose Validation Framework

Extends 1D depth-only simulations to 2D (depth × lateral) dose maps,
showing framework scalability to realistic geometries.
"""

import os

# Fewer energies for 2D (more computationally intensive)
ENERGIES = [70, 150, 250, 500, 1000, 2000]

TEMPLATE_2D = '''# KAVACH BL4S - 2D Dose Distribution: {energy} MeV Proton
# Full 2D scoring (lateral × depth) for realistic geometry validation

i:Ts/Seed = {seed}
i:Ts/NumberOfThreads = 4
i:Ts/ShowHistoryCountAtInterval = 1000
b:Ts/PauseBeforeQuit = "False"
b:Ts/ShowCPUTime = "True"
i:Ts/SequenceVerbosity = 0

# Physics with fine cuts
s:Ph/ListName = "Research"
s:Ph/Research/Type = "Geant4_Modular"
sv:Ph/Research/Modules = 3 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay"
d:Ph/Research/CutForGamma = 0.05 mm
d:Ph/Research/CutForElectron = 0.05 mm
d:Ph/Research/CutForPositron = 0.05 mm
d:Ph/Research/CutForProton = 0.05 mm

# World
s:Ge/World/Material = "G4_AIR"
d:Ge/World/HLX = 50.0 cm
d:Ge/World/HLY = 50.0 cm
d:Ge/World/HLZ = 200.0 cm
b:Ge/World/Invisible = "True"

# EJ-200 with Birks' Law
sv:Ma/EJ200/Components = 2 "Hydrogen" "Carbon"
uv:Ma/EJ200/Fractions = 2 0.0847 0.9153
d:Ma/EJ200/Density = 1.023 g/cm3
d:Ma/EJ200/MeanExcitationEnergy = 64.7 eV
s:Ma/EJ200/DefaultColor = "lightblue"
u:Ma/EJ200/BirksConstant = 0.1232

# Larger target for 2D scoring
s:Ge/TargetScint/Type = "TsBox"
s:Ge/TargetScint/Parent = "World"
s:Ge/TargetScint/Material = "EJ200"
d:Ge/TargetScint/HLX = 30.0 mm
d:Ge/TargetScint/HLY = 30.0 mm
d:Ge/TargetScint/HLZ = 10.0 mm
d:Ge/TargetScint/TransZ = 0.0 cm
s:Ge/TargetScint/Color = "lightblue"

# Proton beam
s:So/Proton/Type = "Beam"
s:So/Proton/Component = "BeamPosition"
s:So/Proton/BeamParticle = "proton"
d:So/Proton/BeamEnergy = {energy} MeV
u:So/Proton/BeamEnergySpread = 0.01
s:So/Proton/BeamPositionDistribution = "Gaussian"
s:So/Proton/BeamPositionCutoffShape = "Ellipse"
d:So/Proton/BeamPositionCutoffX = 15.0 mm
d:So/Proton/BeamPositionCutoffY = 15.0 mm
d:So/Proton/BeamPositionSpreadX = 3.0 mm
d:So/Proton/BeamPositionSpreadY = 3.0 mm
s:So/Proton/BeamAngularDistribution = "Gaussian"
d:So/Proton/BeamAngularCutoffX = 90. deg
d:So/Proton/BeamAngularCutoffY = 90. deg
d:So/Proton/BeamAngularSpreadX = 0.001 rad
d:So/Proton/BeamAngularSpreadY = 0.001 rad
i:So/Proton/NumberOfHistoriesInRun = 20000

# 2D Dose Scoring (X lateral × Z depth)
s:Sc/Dose2D/Quantity = "DoseToMedium"
s:Sc/Dose2D/Component = "TargetScint"
s:Sc/Dose2D/OutputFile = "Dose2D_{energy}MeV"
s:Sc/Dose2D/OutputType = "csv"
b:Sc/Dose2D/OutputToConsole = "FALSE"
s:Sc/Dose2D/IfOutputFileAlreadyExists = "Overwrite"
i:Sc/Dose2D/XBins = 30
i:Sc/Dose2D/YBins = 1
i:Sc/Dose2D/ZBins = 50

# Radial energy deposit (for cylindrical symmetry analysis)
s:Sc/EnergyDep2D/Quantity = "EnergyDeposit"
s:Sc/EnergyDep2D/Component = "TargetScint"
s:Sc/EnergyDep2D/OutputFile = "EnergyDep2D_{energy}MeV"
s:Sc/EnergyDep2D/OutputType = "csv"
b:Sc/EnergyDep2D/OutputToConsole = "FALSE"
s:Sc/EnergyDep2D/IfOutputFileAlreadyExists = "Overwrite"
i:Sc/EnergyDep2D/XBins = 30
i:Sc/EnergyDep2D/YBins = 1
i:Sc/EnergyDep2D/ZBins = 50

# 1D Depth profile (integrated over lateral)
s:Sc/DepthProfile/Quantity = "EnergyDeposit"
s:Sc/DepthProfile/Component = "TargetScint"
s:Sc/DepthProfile/OutputFile = "DepthProfile_{energy}MeV"
s:Sc/DepthProfile/OutputType = "csv"
b:Sc/DepthProfile/OutputToConsole = "TRUE"
s:Sc/DepthProfile/IfOutputFileAlreadyExists = "Overwrite"
i:Sc/DepthProfile/ZBins = 100

# 1D Lateral profile (at fixed depth, center)
s:Sc/LateralProfile/Quantity = "EnergyDeposit"
s:Sc/LateralProfile/Component = "TargetScint"
s:Sc/LateralProfile/OutputFile = "LateralProfile_{energy}MeV"
s:Sc/LateralProfile/OutputType = "csv"
b:Sc/LateralProfile/OutputToConsole = "TRUE"
s:Sc/LateralProfile/IfOutputFileAlreadyExists = "Overwrite"
i:Sc/LateralProfile/XBins = 60

s:Sc/OutputDir = "../output/dose2d/"
'''

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(script_dir, "runs")
    output_dir = os.path.join(script_dir, "output", "dose2d")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating 2D Dose TOPAS configurations...")
    print("=" * 50)
    
    for i, energy in enumerate(ENERGIES):
        seed = 6000 + i * 100 + energy
        config = TEMPLATE_2D.format(energy=energy, seed=seed)
        
        filename = os.path.join(runs_dir, f"Dose2D_{energy}MeV.txt")
        with open(filename, 'w') as f:
            f.write(config)
        
        print(f"  ✓ Created Dose2D_{energy}MeV.txt (seed={seed})")
    
    print("=" * 50)
    print(f"Generated {len(ENERGIES)} configuration files for 2D dose")
    print(f"Output directory: output/dose2d/")
    print("\nEach simulation scores:")
    print("  - 2D dose map (30 × 50 bins = 1500 points)")
    print("  - Depth profile (100 bins)")
    print("  - Lateral profile (60 bins)")

if __name__ == "__main__":
    main()
