#!/bin/bash
# KAVACH BL4S - Run Water Simulations
# Runs TOPAS for all 14 energy configurations

mkdir -p output/water

echo "Starting Water Phantom Simulations..."
echo "Output directory: output/water/"

for energy in 70 100 150 200 250 300 500 750 1000 1500 2000 3000 4500 6000
do
    echo "Running Water_${energy}MeV.txt..."
    topas runs/Water_${energy}MeV.txt
done

echo "All simulations completed."
