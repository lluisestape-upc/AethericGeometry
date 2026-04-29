@echo off
setlocal

echo ========================================
echo  Aetheric Geometry - Build
echo ========================================
echo.

where pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    echo.
)

echo Building AethericGeometry.exe ...
pyinstaller --onefile --windowed ^
    --collect-all mediapipe ^
    --hidden-import sounddevice ^
    --name AethericGeometry ^
    main.py

echo.
if exist "dist\AethericGeometry.exe" (
    echo SUCCESS ^> dist\AethericGeometry.exe
) else (
    echo Build may have failed. If mediapipe files are missing at runtime,
    echo try the folder build instead ^(remove --onefile^):
    echo   pyinstaller --windowed --collect-all mediapipe --name AethericGeometry main.py
)

pause
