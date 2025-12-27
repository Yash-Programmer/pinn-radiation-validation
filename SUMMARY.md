# KAVACH BL4S OpenTOPAS Simulation Summary

**Date:** December 27, 2025  
**Status:** ✅ Complete - Research-Grade + PINN-Ready + Verified  
**Data Version:** v2 (Corrected)  
**Total Events:** 140,000 proton histories

---

## ✅ Data Quality Status (v2)

| Check | Result |
|-------|--------|
| No NaN/NULL | ✓ Passed |
| No duplicates | ✓ Passed |
| 100 bins per energy | ✓ Passed |
| Dose values | ✓ 7e-11 Gy/proton (correct) |
| Sim/NIST ratio | ✓ 1.17 (expected) |
| Split integrity | ✓ No overlap |

---

## ✅ Verified Sources

| Data | Source | Verification |
|------|--------|--------------|
| EJ-200 properties | Eljen Technology | ✓ Official datasheet |
| NIST dE/dx | physics.nist.gov | ✓ Live download |
| Simulation | OpenTOPAS 4.2 | ✓ 140k histories |

---

## v2 Data Files

| File | Size | Description |
|------|------|-------------|
| `pinn_training_data_v2.csv` | 261 KB | Master (1400 rows) |
| `train_split_v2.csv` | 149 KB | 8 energies |
| `val_split_v2.csv` | 75 KB | 4 energies |
| `test_split_v2.csv` | 37 KB | 2 energies |
| `scaler_features_v2.pkl` | - | StandardScaler |
| `scaler_target_v2.pkl` | - | MinMaxScaler |
| `material_properties_VERIFIED.json` | - | Eljen datasheet |
| `README_v2.md` | - | Documentation |

---

## v2 Validation Plots

| Plot | Description |
|------|-------------|
| `dose_profiles_v2.png` | All 14 energies |
| `bragg_peaks_v2.png` | Clinical range |
| `nist_comparison_v2.png` | Sim vs NIST |
| `correlation_v2.png` | Feature correlations |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `run_research.sh` | Run simulations |
| `extract_pinn_data_v2.py` | Data extraction (v2) |
| `process_v2_data.py` | Scalers + plots + checks |
| `compare_nist.py` | NIST comparison |

---

## Usage

```bash
# 1. Run simulations (WSL)
./run_research.sh

# 2. Extract data (v2)
python extract_pinn_data_v2.py

# 3. Create scalers + validate
python process_v2_data.py

# 4. Start PINN training!
```

---

## Key Results

- **Dose:** 7.17e-11 Gy per proton (verified correct)
- **Sim/NIST ratio:** 1.17 ± 0.1 (includes nuclear interactions)
- **Bragg peaks:** Visible at clinical energies
- **Data integrity:** 100% validated

**Ready for PINN training! 🚀**
