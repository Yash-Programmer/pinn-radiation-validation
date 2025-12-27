#!/bin/bash
# KAVACH BL4S - Run all BC-408 simulations
# Execute from opentopas_simulation directory

echo "===== KAVACH BL4S BC-408 Simulations ====="
echo "Starting at: $(date)"

# Create output directory
mkdir -p output/bc408

# Define energies
ENERGIES=(70 100 150 200 250 300 500 750 1000 1500 2000 3000 4500 6000)

# Run each simulation
for E in "${ENERGIES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running BC-408 simulation at ${E} MeV..."
    echo "=========================================="
    topas runs/BC408_${E}MeV.txt
    
    if [ $? -eq 0 ]; then
        echo "✓ ${E} MeV completed successfully"
    else
        echo "✗ ${E} MeV FAILED"
    fi
done

echo ""
echo "===== All BC-408 simulations complete ====="
echo "Finished at: $(date)"
