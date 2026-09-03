# Aetheric Geometry

**A real-time hand-gesture instrument. Point a webcam at your hands, draw shapes in the air, hear chords.**

A webcam feed is processed frame-by-frame by [MediaPipe](https://mediapipe.dev/), which tracks 21 landmarks per hand. Those landmarks drive a small state machine that turns the *geometry* your hands describe — a line, a polygon, its centroid, area and aspect ratio — into *sound*: just-intonation chords rendered by a band-limited additive synth with filter, reverb and tremolo. Everything runs live, on CPU, from a laptop camera.

<!-- Demo video (hosted on YouTube; the file also lives in assets/AethericGeometry-Demo.mp4) -->
[![Watch the demo](https://img.youtube.com/vi/QPWXJZWXsSc/hqdefault.jpg)](https://youtu.be/QPWXJZWXsSc)

> ▶ **[Watch the 2-minute demo](https://youtu.be/QPWXJZWXsSc)**

---

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Gesture reference](#gesture-reference)
- [How it works](#how-it-works)
  - [1. Hands → geometry](#1-hands--geometry)
  - [2. Geometry → gestures → state machine](#2-geometry--gestures--state-machine)
  - [3. Geometry → pitch and effects](#3-geometry--pitch-and-effects)
  - [4. The tuning: exact just intonation](#4-the-tuning-exact-just-intonation)
  - [5. The synth: band-limited DSP](#5-the-synth-band-limited-dsp)
  - [6. Voice control](#6-voice-control)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Building a standalone executable](#building-a-standalone-executable)
- [Tests](#tests)
- [License](#license)

---

## What it does

The instrument is a three-state machine. You move between states with gestures; each state hears the geometry differently.

| State | What you see | What you hear |
|---|---|---|
| **IDLE** | Hand skeleton + pinch guides | Silence. One hand's extended-finger count selects the waveform. |
| **HILO** | A line between your index fingers | One pentatonic note — the distance between your hands sets the pitch. |
| **POLYGON** | A filled polygon traced by your fingertips | A just-intonation chord — its position, area and shape control pitch, filter, reverb and tremolo. |

You enter HILO with a **double pinch** (both hands pinch at once), promote it to a chord with a **kiss** or a **pyramid**, and add voices by pinching more fingers to your thumbs. Nothing is menu-driven: the sound is a continuous function of where your hands are.

---

## Quick start

```bash
# 1. Clone, then create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py                 # press q to quit
```

Requires Python 3.10+ and a webcam.

**Optional — MIDI out:** `pip install mido python-rtmidi`, then set `midi_port` in `config.yaml`.
**Optional — voice control:** download a [Vosk model](https://alphacephei.com/vosk/models) into `models/` (see [Voice control](#6-voice-control)).

No Python? A prebuilt Windows binary is produced by `packaging\build.bat` at `dist\AethericGeometry.exe` — double-click and go.

---

## Gesture reference

### IDLE
| Gesture | Effect |
|---|---|
| One hand, 1–4 fingers up (not pinching) | Select waveform: 1 = sine, 2 = saw, 3 = square, 4 = triangle |
| **Double pinch** (both hands pinch at once) | → HILO |

### HILO
| Gesture | Effect |
|---|---|
| Move hands apart / together | Pitch (distance → frequency) |
| **Kiss** (bring pinched midpoints close) | → POLYGON, 2-voice chord |
| **Pyramid** (press all fingertips together) | → POLYGON, 8-voice chord |
| **Double pinch** (tap twice) | → IDLE |

### POLYGON
| Gesture | Effect |
|---|---|
| Move hands left / right | Pitch (X centroid) |
| Move hands up / down | Filter brightness (Y centroid) |
| Pinch middle / ring / pinky to thumb | Add that finger as a polygon vertex — more voices, more effects |
| **Kiss** | → HILO |

Effect unlocks scale with vertex count: **4 vertices** unlock the filter (Y-axis), **6** unlock reverb (polygon area), **8** unlock tremolo (aspect ratio).

### Runtime keys
| Key | Action | Key | Action |
|---|---|---|---|
| `q` | Quit | `h` / `g` | Hold / release the current sound |
| `f` | Toggle fullscreen | Tab | Next effect (for the hold knob) |
| `m` | Toggle mirror | `1`–`6` | Select effect directly |
| `r` | Start / stop WAV recording | `z` `x` `c` `v` | Sine / saw / square / triangle |

---

## How it works

```
Webcam frame
  → flip → MediaPipe → 21 landmarks/hand → assign_hands()   (left/right)
  → gesture detection (pinch, kiss, pyramid, cross, prayer, open palm)
  → state machine (IDLE / HILO / POLYGON)
  → compute_poly_sound()   (geometry → frequencies + effect amounts)
  → SynthEngine.set_params()   (audio thread picks up new params)
  → OpenCV render (skeleton, shapes, HUD) → cv2.imshow()
```

The design goal throughout is that **the numeric core has no hardware dependencies**: `dsp.py`, `tuning.py` and `knob.py` never import a camera, a sound card or MediaPipe, so the whole geometry-and-sound pipeline can be unit-tested and measured offline (`analysis/measure_dsp.py` reproduces every DSP figure quoted below).

### 1. Hands → geometry

MediaPipe returns 21 normalised landmarks per hand. Everything downstream is built from three primitives in `gestures.py`:

- **Distance** between two landmarks — `dist(a, b) = hypot(aₓ−bₓ, a_y−b_y)`. Pinches, kisses and the HILO pitch are all thresholds on distances.
- **Midpoint** — used for the pinched-fingertip anchors that define the HILO line and the "kiss".
- **Polygon area**, via the shoelace formula over the fingertip vertices:

$$A = \tfrac{1}{2}\left|\sum_{i} \left(x_i\, y_{i+1} - x_{i+1}\, y_i\right)\right|$$

Hands are assigned left/right from **MediaPipe handedness** (calibrated for a mirrored selfie image, so "Left" = your left hand), with a thumb-x-vs-pinky-x fallback when handedness is unavailable.

### 2. Geometry → gestures → state machine

Gestures are pixel-threshold predicates over those distances (all tunable in `config.yaml`):

| Gesture | Definition | Default |
|---|---|---|
| Pinch | `dist(thumb_tip, finger_tip) < PINCH_THRESH` | 60 px |
| Double pinch | Both hands pinching simultaneously | — |
| Kiss | Double pinch **and** midpoints within `DOUBLE_PINCH_THRESH` | 90 px |
| Pyramid | ≥ 4 cross-hand fingertip pairs within `PYRAMID_THRESH` | 70 px |
| Cross | Right index tip is left of left index tip by > 20 px | 20 px |
| Prayer | Wrists within `PRAYER_DIST` **and** ≥ 3 fingertip pairs close | 160 px |
| Open palm | Thumb-to-pinky spread > `OPEN_PALM_SPREAD` | 100 px |

They drive the state machine:

| State | Enter | Exit |
|---|---|---|
| `IDLE` | Startup, or hand lost for > `COOLDOWN_FRAMES` | Double pinch (rising edge) |
| `HILO` | Double pinch from IDLE | Kiss or pyramid → POLYGON; double-tap → IDLE |
| `POLYGON` | Kiss or pyramid from HILO | Kiss → HILO |

A separate "secret" sequence — **cross → prayer → open palm**, each held ~0.7 s — triggers a Rasengan easter egg.

### 3. Geometry → pitch and effects

In POLYGON, `compute_poly_sound()` reads the polygon and maps it to synthesis parameters. The mapping is deliberately axis-aligned so it is learnable by feel:

| Geometry | Controls | Formula |
|---|---|---|
| X of centroid | Pitch | `pitch_from_norm(cx / W)` → pentatonic snap |
| Y of centroid | Filter cutoff | `filter = clip(1 − cy/H, 0, 1)` (top = bright) |
| Polygon area | Reverb wet (≥ 6 verts) | `clip(area / (0.20·W·H), 0, 1)` |
| Aspect ratio | Tremolo depth (≥ 8 verts) | `clip((x_span/y_span − 0.8) / 1.4, 0, 1)` |
| Active fingertips | Voice count | `clip(#fingers, 2, 8)` |

Pitch does **not** map linearly to hand position. Pitch perception is logarithmic, so a linear map cramps the low end and thins the high end. The control uses an exponential map that gives **constant cents per unit of hand travel**:

$$f = f_\text{lo}\left(\frac{f_\text{hi}}{f_\text{lo}}\right)^{t}, \qquad t \in [0,1]$$

The result is then snapped to the nearest note of the **A-minor pentatonic** scale (`quantize_pentatonic`), so there are no wrong notes to land on.

### 4. The tuning: exact just intonation

Within a chord, the voices are tuned in **exact just intonation** — small-integer frequency ratios above the root — not 12-tone equal temperament. The whole point is that the partials of different voices coincide *exactly*, so the beating between them vanishes and the chord fuses.

That property is destroyed by rounding, which is why the ratios are stored as Python `Fraction`s, kept exact until a single float multiply per partial:

| Voices | Ratios above the root |
|---|---|
| 1 | 1/1 |
| 2 | 1/1 · 3/2 |
| 3 | 1/1 · 5/4 · 3/2 |
| 4 | 1/1 · 5/4 · 3/2 · 2/1 |
| 5 | 1/1 · 9/8 · 5/4 · 3/2 · 2/1 |
| 6 | + 7/4 (harmonic seventh) |
| 7 | + 4/3 (perfect fourth) |
| 8 | + 5/3 (major sixth) |

Simpler ratios (the fifth, then the major third) enter first, because their partials coincide soonest and fuse most strongly. Interval size is measured in cents, $1200\log_2 r$; the just major third, for instance, sits 13.7 cents flat of the tempered one.

This is **just intonation *within* a chord over an equal-tempered root**, and that is a deliberate compromise worth stating plainly. A fully just system has no single answer for what happens when the root moves, because stacking exact fifths never closes the octave — twelve of them overshoot by the **Pythagorean comma**:

$$\frac{(3/2)^{12}}{2^{7}} \approx 1.0136 \quad (\approx 23.5\text{ cents})$$

Choosing a fixed ratio set above a tempered, pentatonic-quantised root sidesteps this comma drift, at the cost of transposition purity.

### 5. The synth: band-limited DSP

`synth.py` + `dsp.py` implement the audio chain:

```
partials → sum/N → envelope → drive → 2-pole TPT lowpass → reverb → tremolo → DC blocker → tanh limiter
```

Three ideas shape the DSP, each fixing an audible defect (all reproducible via `analysis/measure_dsp.py`):

- **Band-limited oscillators (PolyBLEP).** A naive saw/square has harmonics at every multiple of $f_0$ with amplitude $1/k$; every one above $f_s/2$ folds back as an inharmonic alias. Because pitch here is a *continuous* function of hand position, those aliases slide around as you move — far more noticeable than static aliasing. PolyBLEP replaces each discontinuity with a two-sample band-limited step, buying **~16 dB** of alias rejection for a couple of multiply-adds per sample.

- **Prewarped filters (TPT).** The naive one-pole $c = e^{-2\pi f_c/f_s}$ is only accurate while $f_c \ll f_s$ — it is ~2 dB off by 20 kHz. The topology-preserving-transform form uses $g = \tan(\pi f_c/f_s)$, so the cutoff lands *exactly* where requested, all the way to Nyquist.

- **Schroeder reverb.** Four damped feedback combs into two allpass diffusers. Each comb solves its **own** feedback gain from a single T60 target — a shared gain would make the decay time proportional to delay length (a 13.7 % spread across the bank). The feedback delays are block-vectorised: when the delay length is ≥ the audio chunk, the recursion unrolls into a delay-line read plus a vector multiply, which is what lets a Python audio callback run four combs and two allpasses with no per-sample loop.

Other details that were bugs before they were features: the envelope stores time constants in **seconds** (`attack_tau`, `release_tau`) so envelope times don't drift with the sample rate; the triangle integrator is seeded from the current phase to kill an 85 % overshoot on every note; and MIDI note numbers add the `+12` that MIDI requires (MIDI 0 is $C_{-1}$, an octave below $C_0$) — without it every emitted note was an octave flat.

References: Välimäki & Huovilainen (2007); Zavalishin (2012), *The Art of VA Filter Design*; Schroeder (1962), *Natural sounding artificial reverberation*.

### 6. Voice control

Optional, off by default. With a [Vosk](https://alphacephei.com/vosk/models) model in `models/`, you can drive the instrument by voice — Spanish (`vosk-model-small-es-0.42`) or English — spoken as ordinary sentences:

```
"jarvis, ponme una onda cuadrada"
"jarvis" … "me gusta, congela" … "ahora el eco" … "sube un poco"
```

It is **keyword spotting, not language understanding**: a result is scanned word by word and any known keyword fires. Four gates separate a command from ordinary speech — final results only, a peak-RMS floor, an utterance-length triage, and a per-word confidence threshold — each one added because measurement demanded it (room tone alone otherwise fired ~11 commands in 40 s). Short commands (`congela`, `eco`) fire immediately; longer sentences need the wake word `eter` (`aether` isn't in the Spanish lexicon, but `eter` is). Every voice command has a keyboard equivalent, so the mic is never required. `python analysis/voice_monitor.py` shows exactly what the model hears, per-word, with the same gates applied.

---

## Configuration

Every threshold, timing, colour and audio parameter lives in **`config.yaml`**; `config.py` loads it with `pyyaml` and falls back to hardcoded defaults if YAML is unavailable. Tune the instrument without touching code:

```yaml
thresholds:
  pinch: 60             # px — thumb-to-finger distance for a pinch
  cooldown_frames: 45   # frames before auto-reset when hands leave the frame

audio:
  amplitude_hilo: 0.28
  reverb_tail_alpha: 0.0005   # how slowly the reverb tail decays
  midi_port: null             # set to a port name to enable MIDI output
```

---

## Project layout

```
main.py             — entry point: event loop, state machine, key handling
config.yaml         — user-editable thresholds, timings, colours, audio params
pyproject.toml      — package metadata + pytest config
aetheric/           — the engine package
├── gestures.py     — geometry helpers, gesture detectors, hand assignment
├── knob.py         — rotary "hold" knob: grip detection, accumulating rotation
├── voice.py        — offline (Vosk) voice commands on a background thread
├── vocabulary.py   — bilingual command → keyword tables
├── ui.py           — HUD toolkit: TrueType text, panels, dial, tile cache
├── renderer.py     — gesture overlays (OpenCV) + HUD composition
├── audio_map.py    — compute_poly_sound(): polygon geometry → synth parameters
├── tuning.py       — exact just-intonation ratios, pitch/note/MIDI helpers
├── dsp.py          — DSP primitives: PolyBLEP oscillators, TPT filter, comb/allpass
├── synth.py        — SynthEngine: oscillator bank, filter, reverb, hold, recording, MIDI
├── midiout.py      — optional MIDI output controller
└── config.py       — loads config.yaml, exposes flat constants with safe fallbacks
tests/              — pytest suite (gestures, audio_map, synth, dsp, tuning, knob, voice)
analysis/           — offline measurement scripts that reproduce every quoted figure
packaging/          — PyInstaller spec + build scripts (build.bat / build.sh)
docs/               — LaTeX technical report (PDF + source)
models/             — Vosk speech models (not committed; download separately)
assets/             — demo video and images (not committed; video is on YouTube)
```

The modules live in an importable `aetheric` package; `main.py` stays at the
repo root as the entry point (`python main.py`). `config.yaml` and `models/`
also stay at the root — user-facing files the program resolves relative to the
project, not to the package.

The HUD is worth a note: text goes through Pillow with real TrueType faces (OpenCV's Hershey fonts looked unfinished), and rasterised panels are cached by content. Rendering from scratch cost 39 ms/frame; caching, premultiplied RGB→BGR blits and OpenCV compositing bring the steady state to **5.7 ms**, and **11.6 ms** while a knob is turning.

---

## Building a standalone executable

**Windows:**
```bat
packaging\build.bat
```
Produces `dist\AethericGeometry\AethericGeometry.exe` (a one-folder build). It runs the test suite first and copies `config.yaml` beside the executable so it stays editable after install.

**macOS / Linux:**
```bash
bash packaging/build.sh
```

Both scripts can be launched from anywhere — they resolve the repo root themselves.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

The suite runs with no camera, microphone or sound card — gesture and synth code is exercised through lightweight `SimpleNamespace` mocks (195 tests).

---

## License

MIT — see [LICENSE](LICENSE).
