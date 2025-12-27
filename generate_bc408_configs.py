"""
Generate TOPAS BC-408 scintillator simulation configuration files for all 14 energies.
KAVACH BL4S - Framework Generalization

BC-408 (Saint-Gobain) plastic scintillator properties:
- Base: Polyvinyltoluene (PVT)
- Density: 1.032 g/cm³
- Composition: C₁₀H₁₁ approximated as H:8.5%, C:91.5%
- Light yield: 11,000 photons/MeV
- Decay time: 2.1 ns
"""

import os

ENERGIES = [70, 100, 150, 200, 250, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000]

# BC-408 template
TEMPLATE = '''# KAVACH BL4S - BC-408 Scintillator Validation: {energy} MeV Proton
# Framework generalization to second plastic scintillator material

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

# BC-408 Plastic Scintillator (Saint-Gobain)
# Polyvinyltoluene base: C10H11, density 1.032 g/cm3
sv:Ma/BC408/Components = 2 "Hydrogen" "Carbon"
uv:Ma/BC408/Fractions = 2 0.0852 0.9148
d:Ma/BC408/Density = 1.032 g/cm3
d:Ma/BC408/MeanExcitationEnergy = 64.7 eV
s:Ma/BC408/DefaultColor = "green"
u:Ma/BC408/BirksConstant = 0.126

# Target scintillator
s:Ge/TargetBC408/Type = "TsBox"
s:Ge/TargetBC408/Parent = "World"
s:Ge/TargetBC408/Material = "BC408"
d:Ge/TargetBC408/HLX = 25.0 mm
d:Ge/TargetBC408/HLY = 25.0 mm
d:Ge/TargetBC408/HLZ = 5.0 mm
d:Ge/TargetBC408/TransZ = 0.0 cm
s:Ge/TargetBC408/Color = "green"

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
d:So/Proton/BeamPositionSpreadX = 5.0 mm
d:So/Proton/BeamPositionSpreadY = 5.0 mm
s:So/Proton/BeamAngularDistribution = "Gaussian"
d:So/Proton/BeamAngularCutoffX = 90. deg
d:So/Proton/BeamAngularCutoffY = 90. deg
d:So/Proton/BeamAngularSpreadX = 0.001 rad
d:So/Proton/BeamAngularSpreadY = 0.001 rad
i:So/Proton/NumberOfHistoriesInRun = 10000

# Scorers
s:Sc/EnergyDep/Quantity = "EnergyDeposit"
s:Sc/EnergyDep/Component = "TargetBC408"
s:Sc/EnergyDep/OutputFile = "BC408_EnergyDep_{energy}MeV"
s:Sc/EnergyDep/OutputType = "csv"
b:Sc/EnergyDep/OutputToConsole = "TRUE"
s:Sc/EnergyDep/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/Dose/Quantity = "DoseToMedium"
s:Sc/Dose/Component = "TargetBC408"
s:Sc/Dose/OutputFile = "BC408_Dose_{energy}MeV"
s:Sc/Dose/OutputType = "csv"
b:Sc/Dose/OutputToConsole = "TRUE"
s:Sc/Dose/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/EnergyDepZ/Quantity = "EnergyDeposit"
s:Sc/EnergyDepZ/Component = "TargetBC408"
s:Sc/EnergyDepZ/OutputFile = "BC408_EnergyDepZ_{energy}MeV"
s:Sc/EnergyDepZ/OutputType = "csv"
s:Sc/EnergyDepZ/IfOutputFileAlreadyExists = "Overwrite"
i:Sc/EnergyDepZ/ZBins = 100

s:Sc/OutputDir = "../output/bc408/"
'''

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(script_dir, "runs")
    output_dir = os.path.join(script_dir, "output", "bc408")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating BC-408 TOPAS configurations...")
    print("=" * 50)
    
    for i, energy in enumerate(ENERGIES):
        seed = 8000 + i * 100 + energy  # Unique seed per energy
        config = TEMPLATE.format(energy=energy, seed=seed)
        
        filename = os.path.join(runs_dir, f"BC408_{energy}MeV.txt")
        with open(filename, 'w') as f:
            f.write(config)
        
        print(f"  ✓ Created BC408_{energy}MeV.txt (seed={seed})")
    
    print("=" * 50)
    print(f"Generated {len(ENERGIES)} configuration files in runs/")
    print(f"Output directory: output/bc408/")
    print("\nTo run simulations:")
    print("  bash run_bc408.sh")
    print("  OR run each individually:")
    print("  topas runs/BC408_70MeV.txt")

if __name__ == "__main__":
    main()
