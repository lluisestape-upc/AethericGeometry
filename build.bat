@echo off
REM ===========================================================================
REM  Build the desktop app: dist\AethericGeometry\AethericGeometry.exe
REM
REM  Produces a folder that runs on a machine with no Python installed. Zip it
REM  for download, or point Inno Setup at it for a real installer.
REM ===========================================================================
setlocal
REM Work from the script's own folder, so it can be launched from anywhere.
cd /d "%~dp0"

set VENV=venv\Scripts
if not exist "%VENV%\python.exe" (
    echo Could not find %VENV%\python.exe - create the venv first.
    exit /b 1
)

echo.
echo == Installing build dependency =============================================
"%VENV%\python.exe" -m pip install --quiet --upgrade pyinstaller || exit /b 1

echo.
echo == Checking the models are present =========================================
if not exist "models\vosk-model-small-en-us-0.15" (
    echo WARNING: English model missing - voice will be Spanish-only.
)
if not exist "models\vosk-model-small-es-0.42" (
    echo WARNING: Spanish model missing - voice will be English-only.
)

echo.
echo == Running the test suite ==================================================
REM Shipping a build that does not pass its own tests is how a demo dies.
"%VENV%\python.exe" -m pytest tests -q || (
    echo Tests failed - refusing to build.
    exit /b 1
)

echo.
echo == Freezing ================================================================
"%VENV%\python.exe" -m PyInstaller AethericGeometry.spec --noconfirm || exit /b 1

echo.
echo == Copying the editable config beside the exe ==============================
copy /Y config.yaml "dist\AethericGeometry\config.yaml" >nul

echo.
echo Done: dist\AethericGeometry\AethericGeometry.exe
echo Edit dist\AethericGeometry\config.yaml to change camera, MIDI port, voice.
endlocal
