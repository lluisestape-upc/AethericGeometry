# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for a one-folder desktop app.

Build:
    pip install pyinstaller
    pyinstaller AethericGeometry.spec --noconfirm

Produces dist/AethericGeometry/, which runs on a machine with no Python. Zip
that folder for distribution, or feed it to Inno Setup for a real installer.

Why one-folder and not --onefile
--------------------------------
A single .exe is tidier to hand over but unpacks itself to a temp directory on
every launch. This bundle carries two Vosk models and MediaPipe's graphs -
roughly 150 MB - so --onefile turns every start into a multi-second stall and
re-extracts the same files each time. One-folder starts immediately, and the
user still only sees one thing to double-click.

config.yaml is deliberately shipped *beside* the executable rather than frozen
inside it, so it can be edited after install: a user with no microphone, a
different camera index or a loopMIDI port to name should not need a rebuild.
"""
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

# This spec lives in packaging/, so the project root is one level up.
PROJECT = Path(SPECPATH).parent

# Set AETHERIC_DEBUG=1 to build a console variant. A windowed build swallows
# startup tracebacks behind a modal dialog that says only "Unhandled exception
# in script", which is not enough to fix anything.
DEBUG = os.environ.get("AETHERIC_DEBUG") == "1"

# MediaPipe is the awkward one. Its solutions are loaded through dynamic
# imports and it reads .binarypb graph definitions and .tflite weights from
# package data at runtime, so collecting data files alone leaves the modules
# themselves behind and `mp.solutions.hands` fails on the frozen build.
# collect_all takes the submodules and native libraries too.
mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")
vosk_datas, vosk_binaries, vosk_hidden = collect_all("vosk")

datas = mp_datas + vosk_datas
binaries = mp_binaries + vosk_binaries
hiddenimports = mp_hidden + vosk_hidden

datas += collect_data_files("sounddevice", include_py_files=False)

for model in (PROJECT / "models").glob("vosk-model-*"):
    if model.is_dir():
        datas.append((str(model), f"models/{model.name}"))

datas.append((str(PROJECT / "config.yaml"), "."))

a = Analysis(
    [str(PROJECT / "main.py")],
    pathex=[str(PROJECT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        "sounddevice",
        "scipy.signal",
        "mido.backends.rtmidi",     # mido loads its backend by name at runtime
    ],
    hookspath=[],
    runtime_hooks=[],
    # matplotlib is NOT excludable, however tempting: mediapipe's
    # drawing_utils imports it at module scope, so dropping it to save ~40 MB
    # produced a frozen build that died on `import mediapipe` with
    # ModuleNotFoundError - and, being windowed, said only "Unhandled exception
    # in script".
    excludes=["tkinter", "pytest", "IPython", "notebook"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AethericGeometry",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=DEBUG,          # windowed normally; AETHERIC_DEBUG=1 for a console
    icon=str(PROJECT / "assets" / "icon.ico")
    if (PROJECT / "assets" / "icon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AethericGeometry",
)
