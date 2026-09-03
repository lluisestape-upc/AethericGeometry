# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Project Overview

**Aetheric Geometry** — a real-time hand gesture instrument. A webcam feed drives MediaPipe hand-landmark detection; detected gestures feed a state machine that maps hand geometry to just-intonation audio synthesis via OpenCV rendering.

## Running the Application

```bash
.\venv\Scripts\activate    # Windows
python main.py
# Press q to quit
```

## Dependencies

```bash
pip install -r requirements.txt
# Optional MIDI: pip install mido python-rtmidi
# Tests:         pip install pytest
```

## Building the Executable

```bat
build.bat           # Windows  → dist\AethericGeometry.exe
bash build.sh       # Mac/Linux
```

## Architecture

```
main.py        — thin event loop, state machine, key handling
gestures.py    — geometry helpers, gesture detectors, hand assignment
knob.py        — rotary hand knob: grip detection, accumulating rotation
voice.py       — offline (Vosk) voice commands on a background thread
ui.py          — HUD toolkit: TrueType text, panels, dial, tile cache
renderer.py    — gesture overlays (OpenCV) + HUD composition (via ui.py)
audio_map.py   — compute_poly_sound(): polygon geometry → synth parameters
tuning.py      — exact just-intonation ratios, pitch/note/MIDI helpers
dsp.py         — DSP primitives: PolyBLEP oscillators, TPT filter, comb/allpass
synth.py       — SynthEngine: oscillator bank, filter, reverb, hold, recording, MIDI
config.py      — loads config.yaml, exposes flat constants with safe fallbacks
config.yaml    — user-editable thresholds, timings, colors, audio params
tests/         — pytest suite (gestures, audio_map, synth, dsp, tuning, knob)
analysis/      — measure_dsp.py: reproduces every number quoted below
```

`dsp.py`, `tuning.py` and `knob.py` are deliberately free of camera, audio and
MediaPipe dependencies, so the whole numeric core is testable and measurable
with no hardware attached.

### Processing Pipeline

```
Webcam frame → flip → MediaPipe RGB → assign_hands()
  → waveform selector (IDLE, one hand, 5-frame debounce)
  → gesture detection (pinch, kiss, pyramid, cross, prayer, open palm)
  → state machine (IDLE / HILO / POLIGONO)
  → Rasengan sequence (cross → pray → open palm)
  → compute_poly_sound() → SynthEngine.set_params()
  → OpenCV render → cv2.imshow()
```

## State Machine

| State | Enter | Exit |
|---|---|---|
| `IDLE` | startup / hand lost > `COOLDOWN_FRAMES` | double pinch (rise edge) |
| `HILO` | double pinch from IDLE | kiss → POLIGONO; pyramid → POLIGONO; double-tap → IDLE |
| `POLIGONO` | kiss or pyramid from HILO | kiss → HILO |

## Gesture Definitions

All thresholds are pixels; configurable in `config.yaml`:

| Gesture | Definition |
|---|---|
| **Pinch** | `dist(thumb_tip, finger_tip) < PINCH_THRESH` (60 px) |
| **Double pinch** | Both hands pinching simultaneously |
| **Kiss** | Double pinch AND midpoints within `DOUBLE_PINCH_THRESH` (90 px) |
| **Pyramid** | ≥4 cross-hand fingertip pairs within `PYRAMID_THRESH` (70 px) |
| **Cross pose** | Right index tip is LEFT of left index tip by > 20 px |
| **Prayer pose** | Wrists within `PRAYER_DIST` (160 px) AND ≥3 fingertip pairs close |
| **Open palm** | Thumb-to-pinky distance > `OPEN_PALM_SPREAD` (100 px) |

## Hand Assignment

`assign_hands()` in `gestures.py` uses **MediaPipe handedness** as primary signal (calibrated for mirrored/selfie images: "Left" = user's left hand). Falls back to thumb-x vs. pinky-MCP heuristic when handedness is unavailable.

## Waveform Selection

One hand only, IDLE state, not pinching. Extended finger count → `WAVES[n-1]`: 1=SIN, 2=SAW, 3=SQUARE, 4=TRIANGLE. A 5-frame debounce prevents jitter.

## Audio Engine (`synth.py` + `dsp.py`)

Chain: `partials → sum/N → envelope → drive → 2-pole TPT lowpass → reverb →
tremolo → DC blocker → tanh limiter`.

- **Additive synthesis:** N oscillators on exact just-intonation ratios
  (`tuning.CHORD_RATIOS`, stored as `Fraction`, not rounded floats)
- **Band-limited oscillators:** PolyBLEP saw/square, integrated BLEP triangle.
  Buys ~16 dB of alias rejection across the range
- **Asymmetric envelope:** time constants in **seconds** (`attack_tau`,
  `release_tau`), so envelopes do not change with the device sample rate
- **TPT lowpass:** prewarped `g = tan(pi*fc/fs)`, two stages. Exact -3 dB at
  the requested cutoff; the old `exp(-2*pi*fc/fs)` was 2 dB out by 20 kHz
- **Schroeder reverb:** 4 damped combs → 2 allpass diffusers. Each comb solves
  its own feedback gain from a single T60 target, because a shared gain makes
  T60 proportional to delay length (a 13.7 % spread across the bank)
- **Tremolo:** aspect-ratio-based LFO, 0.5–10 Hz
- **Hold:** freezes pitch/waveform/level so the hands are free for the knob;
  `set_params` is ignored while held, `set_effect` still applies
- **Recording:** `r` key → WAV via stdlib `wave`, no extra dependency
- **MIDI output:** optional via `mido`; set `midi_port` in `config.yaml`

### Fixed defects, all reproducible via `analysis/measure_dsp.py`

| Defect | Effect | Status |
|---|---|---|
| Phase written back one sample short per block | −3.385 cents flat, plus a discontinuity at fs/N = 86.1 Hz | fixed |
| Naive saw/square | alias-to-signal −12 dB at 2 kHz | fixed (PolyBLEP) |
| One feedback gain shared by four combs | T60 1.12–1.28 s across the bank | fixed (solve per comb) |
| No allpass diffusion | four bare echo trains | fixed |
| Envelope stored as raw coefficients | every envelope time changed with fs | fixed (seconds) |
| `12*log2(f/C0)` used as a MIDI note | every MIDI note an octave flat | fixed (`+12`) |
| Triangle integrator started from zero | 85 % overshoot for ~16 ms on every note | fixed (seed from phase) |

## Hold + rotary knob

Say **"hold"** (or press `h`) to freeze the sound. The hands are then free: pinch
and rotate to turn the selected effect's knob, and name an effect to select it
(`filter`, `reverb`, `decay`, `damping`, `tremolo`, `drive`). Say **"release"**
(or `g`) to let it go.

## Voice (`voice.py` + `vocabulary.py`)

Spanish by default (`vosk-model-small-es-0.42`), spoken as ordinary sentences:

```
"jarvis, ponme una onda cuadrada"
"jarvis"  ...  "me gusta, congela"  ...  "ahora el eco"  ...  "sube un poco"
```

It is **keyword spotting, not language understanding**. A result is scanned word
by word and anything in `vocabulary` fires. The sentence is not parsed.

Gates between a sound and a command, each answering a different question, and
every one added because measurement demanded it:

| Gate | Question | Measured without it |
|---|---|---|
| Final results only | is the utterance finished? | partials fired 11 commands in 25 s of silence |
| `min_level` (peak RMS) | did anyone actually speak? | room tone fired 1 per 45 s at high confidence |
| Length triage | was it meant for us? | **11 commands in 40 s** of ordinary room sound |
| `min_confidence` | which word was it? | — |

### The triage rule

Requiring a wake word for everything would make the instrument unusable — you
have to be able to bark "eco" mid-performance. Requiring nothing makes room
conversation play the instrument for you. So **length decides**:

| Utterance | Wake word | Result |
|---|---|---|
| `congela` | no | fires — short commands need no ceremony |
| `sube el eco` | no | fires — still within `short_utterance_words` |
| `pues mira sube un poco mas` | no | **dropped whole** — conversation, despite containing two command words |
| `eter pillame el eco y baja treinta por ciento` | yes | parsed fully: selects reverb, lowers it 30 % |

Words *before* the wake word are not considered, and the wake word opens a
window (`wake_timeout`) so a follow-up sentence needs no repeat.

Spoken numbers are read as percentages (`treinta por ciento` → 0.30); a bare
`sube` uses `NUDGE_STEP`. Vosk emits words, never digits, so these are looked up
rather than parsed.

**`aether` is not in the Spanish lexicon** — the wake word is `eter`, which is.
Set `wake_word: null` to listen continuously and accept the strays.

### Vocabulary traps, all found by probing the lexicon

A Vosk grammar can only contain words the model already knows; anything else is
dropped with a native-stderr warning and the command silently never fires.

* **The Spanish lexicon is unaccented**: `mas`, `triangulo`, `atras` exist;
  `más`, `triángulo`, `atrás` do not. A test enforces ASCII-only Spanish keys.
* **The obvious technical words are missing**: neither `sinusoidal` nor `reverb`
  is in the Spanish model, and `tremolo` is in neither model. The plain words —
  `seno`, `eco`, `temblor` — are what work.
* **English `sine` is /saɪn/**, not "SEE-neh"; a Spanish speaker reading the word
  aloud produces something the English model cannot map. Hence `round`,
  `smooth`, `pure`.
* Filler words (`ponme`, `una`, `onda`, …) are in the grammar but map to
  nothing. Without them the other words of a sentence smear onto commands.

`tests/test_voice.py` enforces that every command has at least two trigger words
and that both languages expose the same command set.

## Waveform by voice

Say `cuadrada`, `triangular`, `sierra`, `seno` (or `square`, `triangle`, `saw`,
`round`) at any moment — before a note exists, while a shape is sounding, or
while one is held. `SynthEngine.set_waveform()` is deliberately *not* gated on
the hold the way `set_params()` is, and it leaves the phase accumulators alone,
so switching shape under a sustained chord is an audition rather than a
retrigger.

Every voice command has a keyboard equivalent — `h`, `g`, Tab, `1`–`6`,
`z`/`x`/`c`/`v` — so the instrument never depends on the microphone or on Vosk
being installed.

Run `python analysis/voice_monitor.py` to see exactly what the model hears, with
per-word confidence, level, and the same four gates applied.

## HUD (`ui.py`)

Text goes through Pillow with real TrueType faces, not `cv2.putText`: the
Hershey fonts are single-stroke vector outlines with no hinting or kerning, and
they were the main reason the interface looked unfinished. Panels are drawn into
RGBA tiles at 2x and box-filtered down, which also antialiases Pillow's arcs and
rounded corners. Sizes derive from frame height so the HUD holds its proportions
from a small window up to fullscreen.

Rendering the HUD from scratch every frame cost 39 ms. Three things fixed it:

| Change | Effect |
|---|---|
| Cache rasterised tiles, keyed by their content (`ui.TileCache`) | most frames became a blit |
| Premultiply and convert RGB→BGR once at rasterise time | blit 5.9 ms → 1.4 ms |
| Composite with OpenCV instead of numpy expressions | SIMD, no temporaries |
| Split the hold panel into chrome / dial / rows tiles | turning a knob 21 ms → ~6 ms |

Steady state is now **5.7 ms**, and **11.6 ms** while a knob is turning.

One trap worth knowing: the effect-meter loop in `draw_status_bar` originally
used `key` as its loop variable, which shadowed the cache key computed above it.
Tiles were stored under the wrong key and the cache never hit — the panel looked
correct and simply ran at full cost forever.

Grip detection requires a pinch **and** at least two other fingers extended: a
fist also brings thumb and index together, so pinch alone would keep the knob
held once the hand relaxed. Both tests have hysteresis. Extension is measured
radially from the wrist in palm widths, not by tip-above-knuckle — this gesture
rotates the hand, which is exactly when a vertical test stops working.

## Key Bindings (runtime)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `f` | Toggle fullscreen |
| `m` | Toggle mirror |
| `r` | Start / stop WAV recording |
| `h` | Hold the current sound |
| `g` | Release the hold |
| Tab | Next effect |
| `1`–`6` | Select effect directly |
| `z` `x` `c` `v` | Sine / saw / square / triangle |

## Configuration

All magic numbers live in `config.yaml`. `config.py` loads it with `pyyaml` and falls back to hardcoded defaults if YAML is unavailable.

## Running Tests

```bash
pytest tests/ -v
```

Tests use simple `types.SimpleNamespace` mocks — no MediaPipe, camera, or audio hardware required.
