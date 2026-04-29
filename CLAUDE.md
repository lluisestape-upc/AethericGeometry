# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time hand gesture recognition system that maps gestural movements to harmonic/geometric shapes. A webcam feed is processed to detect hand landmarks; specific gestures trigger a state machine that draws geometric polygons and selects a waveform type (visualization only — no audio output).

## Running the Application

```bash
# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Run
python main.py

# Press 'q' to quit
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Building the executable

```bat
build.bat
```

Produces `dist\AethericGeometry.exe` via PyInstaller. If mediapipe data files fail at runtime with `--onefile`, remove that flag (folder build is more reliable with mediapipe).

## Architecture

All logic lives in `main.py`. No module separation — three clearly separated sections: geometry helpers, drawing helpers, `main()`.

### Processing Pipeline

```
Webcam frame → MediaPipe hand landmarks → Gesture recognition → State transition → OpenCV draw → Display
```

### State Machine

Three states control gesture interpretation and rendering:

| State | Trigger to enter | Trigger to exit |
|---|---|---|
| `IDLE` | Initial / hand lost for 15 frames | Double pinch |
| `HILO` | Double pinch ("beso de pinzas") | Kiss gesture or pyramid |
| `POLIGONO` | Kiss/pyramid from HILO | Kiss gesture → back to HILO |

### Gesture Definitions

All gestures compare Euclidean distances between named landmark constants (`THUMB=4`, `INDEX=8`, `MIDDLE=12`, `RING=16`, `PINKY=20`):

- **Individual pinch:** `dist(thumb_tip, fingertip) < PINCH_THRESH` (60 px)
- **Double pinch / kiss:** both hands pinching AND midpoints within `DOUBLE_PINCH_THRESH` (90 px)
- **Pyramid:** 4+ of the 5 cross-hand fingertip pairs within `PYRAMID_THRESH` (70 px)

### Waveform Selection (left hand, IDLE only)

Left hand extended finger count → `WAVES[n-1]`: 1=SIN, 2=SAW, 3=SQUARE, 4=TRIANGLE. Locked once in HILO/POLIGONO.

### Polygon Vertices (POLIGONO state)

Pinching MIDDLE/RING/PINKY against thumb appends that landmark ID to `active_l` / `active_r`. Polygon points are built by iterating left hand (thumb→pinky) then right hand (pinky→thumb) so the polygon closes cleanly.

### Thresholds

```python
PINCH_THRESH        = 60   # single-finger pinch
DOUBLE_PINCH_THRESH = 90   # kiss midpoint distance
PYRAMID_THRESH      = 70   # cross-hand finger pair
COOLDOWN_FRAMES     = 15   # frames before auto-reset to IDLE
```

### Hand Assignment

Left vs. right determined by `thumb.x < landmark[17].x` (pinky MCP), not MediaPipe's handedness label.
