#!/usr/bin/env bash
# Build script for Aetheric Geometry (Mac / Linux)
set -e

if ! command -v pyinstaller &>/dev/null; then
    echo "PyInstaller not found — installing..."
    pip install pyinstaller
fi

pyinstaller \
    --onefile \
    --windowed \
    --name AethericGeometry \
    --collect-data mediapipe \
    --hidden-import sounddevice \
    main.py

echo ""
echo "Build complete: dist/AethericGeometry"
