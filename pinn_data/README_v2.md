# PINN Training Data v2 (CORRECTED)

## Version History
- **v2** (Dec 27, 2025): Corrected dose, real NIST dE/dx
- **v1**: Deprecated (had calculation errors)

## Corrections Applied in v2
1. Dose calculation fixed (was 0.0 Gy)
2. NIST PSTAR reference integrated (real data from nist.gov)
3. EJ-200 properties verified from Eljen datasheet
4. Column names clarified (sim_dEdx vs nist_dEdx)

## Files

| File | Description |
|------|-------------|
| `pinn_training_data_v2.csv` | Master dataset (1400 rows) |
| `train_split_v2.csv` | Training (8 energies, 800 rows) |
| `val_split_v2.csv` | Validation (4 energies, 400 rows) |
| `test_split_v2.csv` | Test (2 energies, 200 rows) |
| `scaler_features_v2.pkl` | StandardScaler for inputs |
| `scaler_target_v2.pkl` | MinMaxScaler for target |
| `material_properties_VERIFIED.json` | Eljen datasheet values |

## Columns (v2)

| Column | Units | Description |
|--------|-------|-------------|
| energy_MeV | MeV | Proton kinetic energy |
| z_bin | - | Bin index (0-99) |
| depth_mm | mm | Depth in scintillator |
| energy_deposit_MeV | MeV | Total energy in bin |
| sim_dEdx_MeV_per_mm | MeV/mm | TOPAS simulated dE/dx |
| dose_Gy | Gy | Absorbed dose per proton |
| beta | - | Relativistic v/c |
| gamma | - | Lorentz factor |
| momentum_MeV_c | MeV/c | Momentum |
| nist_dEdx_MeV_per_mm | MeV/mm | NIST PSTAR reference |
| nist_range_mm | mm | CSDA range from NIST |
| residual_range_mm | mm | Remaining range |

## Sources Verified

- **EJ-200**: [Eljen Technology](https://eljentechnology.com/products/plastic-scintillators/ej-200)
- **NIST PSTAR**: [physics.nist.gov](https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html)

## Notes

- Dose is per-proton (multiply by N_histories for total)
- sim_dEdx includes nuclear interactions; nist_dEdx is electronic only
- ~10-30% higher sim vs NIST at high energies is expected
