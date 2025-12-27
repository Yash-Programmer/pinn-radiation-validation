#!/bin/bash
# KAVACH BL4S - Run All Proton Simulations
# 14 energy points from 70 MeV to 6000 MeV

export TOPAS_G4_DATA_DIR=~/Applications/GEANT4/G4DATA
export LD_LIBRARY_PATH=~/Applications/TOPAS/OpenTOPAS-install/lib:$LD_LIBRARY_PATH
TOPAS=~/Applications/TOPAS/OpenTOPAS-install/bin/topas

echo "=============================================="
echo "KAVACH BL4S - Multi-Energy Proton Simulation"
echo "=============================================="
echo ""
echo "Energy Points:"
echo "  Clinical: 70, 100, 150, 200, 250 MeV"
echo "  Bridge:   300, 500, 750, 1000 MeV"
echo "  BL4S:     1500, 2000, 3000, 4500, 6000 MeV"
echo ""
echo "Total: 14 runs x 10,000 histories = 140,000 events"
echo ""

cd /mnt/c/Users/Yash/Videos/beamline3/opentopas_simulation/runs

# Clinical Range (70-250 MeV)
echo "[1/14] Running 70 MeV..."
$TOPAS Proton_70MeV.txt > ../output/log_70MeV.txt 2>&1

echo "[2/14] Running 100 MeV..."
$TOPAS Proton_100MeV.txt > ../output/log_100MeV.txt 2>&1

echo "[3/14] Running 150 MeV..."
$TOPAS Proton_150MeV.txt > ../output/log_150MeV.txt 2>&1

echo "[4/14] Running 200 MeV..."
$TOPAS Proton_200MeV.txt > ../output/log_200MeV.txt 2>&1

echo "[5/14] Running 250 MeV..."
$TOPAS Proton_250MeV.txt > ../output/log_250MeV.txt 2>&1

# Bridge Range (300-1000 MeV)
echo "[6/14] Running 300 MeV..."
$TOPAS Proton_300MeV.txt > ../output/log_300MeV.txt 2>&1

echo "[7/14] Running 500 MeV..."
$TOPAS Proton_500MeV.txt > ../output/log_500MeV.txt 2>&1

echo "[8/14] Running 750 MeV..."
$TOPAS Proton_750MeV.txt > ../output/log_750MeV.txt 2>&1

echo "[9/14] Running 1000 MeV..."
$TOPAS Proton_1000MeV.txt > ../output/log_1000MeV.txt 2>&1

# BL4S Range (1500-6000 MeV)
echo "[10/14] Running 1500 MeV..."
$TOPAS Proton_1500MeV.txt > ../output/log_1500MeV.txt 2>&1

echo "[11/14] Running 2000 MeV..."
$TOPAS Proton_2000MeV.txt > ../output/log_2000MeV.txt 2>&1

echo "[12/14] Running 3000 MeV..."
$TOPAS Proton_3000MeV.txt > ../output/log_3000MeV.txt 2>&1

echo "[13/14] Running 4500 MeV..."
$TOPAS Proton_4500MeV.txt > ../output/log_4500MeV.txt 2>&1

echo "[14/14] Running 6000 MeV..."
$TOPAS Proton_6000MeV.txt > ../output/log_6000MeV.txt 2>&1

echo ""
echo "=============================================="
echo "ALL SIMULATIONS COMPLETE"
echo "=============================================="
echo ""
echo "Output files in: opentopas_simulation/runs/"
echo "Log files in: opentopas_simulation/output/"
