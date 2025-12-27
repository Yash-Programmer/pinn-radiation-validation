@echo off
REM ==============================================================================
REM KAVACH BL4S OpenTOPAS Simulation - Research-Grade Master Run Script
REM 
REM Per proposal timeline (Table 2):
REM   Days 5-7: Data: Protons - Five momentum points, 10,000 events each
REM   Days 8-10: Data: Pions - Five momentum points, verification
REM   Day 4: Calibration - Electron runs for reference
REM
REM Runs all 14 beam configurations (6 proton + 6 pion + 2 electron)
REM ==============================================================================

echo ============================================================
echo KAVACH BL4S OpenTOPAS Research Simulation Suite
echo Team KAVACH - Scintillator Quenching Characterization
echo ============================================================
echo.
echo Configuration:
echo   - 14 beam configurations (6p + 6pi + 2e)
echo   - 10,000 events per configuration
echo   - Momentum range: 0.5-5.0 GeV/c
echo   - Expected runtime: ~2-4 hours total
echo.

REM Set OpenTOPAS executable path (modify as needed)
SET TOPAS_EXE=topas

REM Verify OpenTOPAS is available
where %TOPAS_EXE% >nul 2>nul
if errorlevel 1 (
    echo ERROR: OpenTOPAS executable not found in PATH
    echo Please set TOPAS_EXE to the correct path
    exit /b 1
)

REM Create output directories
echo Creating output directories...
if not exist "output\proton\0p5GeV" mkdir "output\proton\0p5GeV"
if not exist "output\proton\1GeV" mkdir "output\proton\1GeV"
if not exist "output\proton\2GeV" mkdir "output\proton\2GeV"
if not exist "output\proton\3GeV" mkdir "output\proton\3GeV"
if not exist "output\proton\4GeV" mkdir "output\proton\4GeV"
if not exist "output\proton\5GeV" mkdir "output\proton\5GeV"
if not exist "output\pion\0p5GeV" mkdir "output\pion\0p5GeV"
if not exist "output\pion\1GeV" mkdir "output\pion\1GeV"
if not exist "output\pion\2GeV" mkdir "output\pion\2GeV"
if not exist "output\pion\3GeV" mkdir "output\pion\3GeV"
if not exist "output\pion\4GeV" mkdir "output\pion\4GeV"
if not exist "output\pion\5GeV" mkdir "output\pion\5GeV"
if not exist "output\electron\2GeV" mkdir "output\electron\2GeV"
if not exist "output\electron\4GeV" mkdir "output\electron\4GeV"

echo.
echo ========================================
echo PHASE 1: ELECTRON CALIBRATION (MIP Reference)
echo ========================================
echo Per proposal: "Electron beam at 2.0 and 4.0 GeV/c provides minimal
echo quenching (minimum ionizing), providing reference for absolute light yield"
echo.

echo [1/14] Running Electron 2.0 GeV/c (MIP calibration)...
%TOPAS_EXE% runs\Electron_2GeV.txt
if errorlevel 1 goto :error

echo.
echo [2/14] Running Electron 4.0 GeV/c (MIP calibration)...
%TOPAS_EXE% runs\Electron_4GeV.txt
if errorlevel 1 goto :error

echo.
echo ========================================
echo PHASE 2: PROTON DATA
echo ========================================
echo Per proposal: "Five momentum points, 10,000 events each"
echo Momentum range: 0.5-5.0 GeV/c
echo.

echo [3/14] Running Proton 0.5 GeV/c (highest quenching)...
%TOPAS_EXE% runs\Proton_0p5GeV.txt
if errorlevel 1 goto :error

echo.
echo [4/14] Running Proton 1.0 GeV/c...
%TOPAS_EXE% runs\Proton_1GeV.txt
if errorlevel 1 goto :error

echo.
echo [5/14] Running Proton 2.0 GeV/c...
%TOPAS_EXE% runs\Proton_2GeV.txt
if errorlevel 1 goto :error

echo.
echo [6/14] Running Proton 3.0 GeV/c...
%TOPAS_EXE% runs\Proton_3GeV.txt
if errorlevel 1 goto :error

echo.
echo [7/14] Running Proton 4.0 GeV/c...
%TOPAS_EXE% runs\Proton_4GeV.txt
if errorlevel 1 goto :error

echo.
echo [8/14] Running Proton 5.0 GeV/c (approaching MIP)...
%TOPAS_EXE% runs\Proton_5GeV.txt
if errorlevel 1 goto :error

echo.
echo ========================================
echo PHASE 3: PION DATA
echo ========================================
echo Per proposal: "Five momentum points, particle ID verification"
echo.

echo [9/14] Running Pion 0.5 GeV/c...
%TOPAS_EXE% runs\Pion_0p5GeV.txt
if errorlevel 1 goto :error

echo.
echo [10/14] Running Pion 1.0 GeV/c...
%TOPAS_EXE% runs\Pion_1GeV.txt
if errorlevel 1 goto :error

echo.
echo [11/14] Running Pion 2.0 GeV/c...
%TOPAS_EXE% runs\Pion_2GeV.txt
if errorlevel 1 goto :error

echo.
echo [12/14] Running Pion 3.0 GeV/c...
%TOPAS_EXE% runs\Pion_3GeV.txt
if errorlevel 1 goto :error

echo.
echo [13/14] Running Pion 4.0 GeV/c...
%TOPAS_EXE% runs\Pion_4GeV.txt
if errorlevel 1 goto :error

echo.
echo [14/14] Running Pion 5.0 GeV/c...
%TOPAS_EXE% runs\Pion_5GeV.txt
if errorlevel 1 goto :error

echo.
echo ============================================================
echo ALL SIMULATIONS COMPLETED SUCCESSFULLY!
echo ============================================================
echo.
echo Output summary:
echo   Proton runs : output\proton\{0p5,1,2,3,4,5}GeV\
echo   Pion runs   : output\pion\{0p5,1,2,3,4,5}GeV\
echo   Electron    : output\electron\{2,4}GeV\
echo.
echo Next steps:
echo   1. Run analysis: cd analysis ^&^& python run_all_analysis.py
echo   2. Generate figures for proposal validation
echo   3. Compare Q_measured vs Q_predicted from Table 3
echo.
goto :end

:error
echo.
echo ============================================================
echo ERROR: Simulation failed at step above!
echo ============================================================
echo.
echo Troubleshooting:
echo   1. Check OpenTOPAS is properly installed
echo   2. Verify geometry file syntax
echo   3. Check available disk space
echo   4. Review error messages above
echo.
exit /b 1

:end
exit /b 0
