import threading
import numpy as np

try:
    import sounddevice as sd
    _AUDIO_OK = True
except ImportError:
    _AUDIO_OK = False

SAMPLE_RATE = 44100
BLOCK_SIZE  = 512

# Just-intonation chord ratios, indexed by voice count
CHORD_RATIOS = {
    1: [1.0],
    2: [1.0, 1.5],
    3: [1.0, 1.25, 1.5],
    4: [1.0, 1.25, 1.5, 2.0],
    5: [1.0, 1.125, 1.25, 1.5, 2.0],
    6: [1.0, 1.125, 1.25, 1.5, 1.75, 2.0],
    7: [1.0, 1.125, 1.25, 1.333, 1.5, 1.75, 2.0],
    8: [1.0, 1.125, 1.25, 1.333, 1.5, 1.667, 1.75, 2.0],
}

_A2         = 110.0
_C0         = 16.352
_PENTA_ST   = [0, 2, 4, 7, 9]
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Schroeder comb-filter delays (samples at 44100 Hz)
_COMB_DELAYS   = (1557, 1617, 1491, 1422)
_COMB_FEEDBACK = 0.82


# ── Pitch helpers ─────────────────────────────────────────────────────────────
def pitch_from_norm(t: float, lo: float = _A2, hi: float = 880.0) -> float:
    return lo * (hi / lo) ** float(np.clip(t, 0.0, 1.0))


def quantize_pentatonic(freq: float) -> float:
    st     = 12.0 * np.log2(max(freq, 1.0) / _A2)
    octave = int(st // 12)
    rem    = st % 12
    candidates = _PENTA_ST + [s + 12 for s in _PENTA_ST]
    nearest    = min(candidates, key=lambda s: abs(s - rem))
    return _A2 * 2.0 ** ((octave * 12 + nearest) / 12.0)


def freq_to_note(freq: float) -> str:
    st = round(12.0 * np.log2(max(freq, 1.0) / _C0))
    return f"{_NOTE_NAMES[st % 12]}{st // 12}"


# ── Waveform generation ───────────────────────────────────────────────────────
def _oscillate(waveform: str, t: np.ndarray) -> np.ndarray:
    if   waveform == "SIN":    return np.sin(2.0 * np.pi * t)
    elif waveform == "SAW":    return 2.0 * (t % 1.0) - 1.0
    elif waveform == "SQUARE": return np.sign(np.sin(2.0 * np.pi * t)).astype(np.float64)
    else:                      return 2.0 * np.abs(2.0 * (t % 1.0) - 1.0) - 1.0


# ── First-order IIR lowpass (Python loop — fast enough for 512-sample blocks) ─
def _lowpass(samples: np.ndarray, cutoff: float, state: float) -> tuple:
    omega = 2.0 * np.pi * np.clip(cutoff, 20.0, SAMPLE_RATE * 0.499) / SAMPLE_RATE
    c     = np.exp(-omega)
    b0    = 1.0 - c
    out   = np.empty_like(samples)
    y     = state
    for i in range(len(samples)):
        y      = b0 * samples[i] + c * y
        out[i] = y
    return out, y


# ── Synth engine ──────────────────────────────────────────────────────────────
class SynthEngine:
    _AMP_ALPHA = 0.002   # amplitude smoothing (~11 ms at 44100 Hz)

    def __init__(self):
        self._lock = threading.Lock()

        # Shared parameters (main thread → callback, protected by lock)
        self._freqs         = [440.0]
        self._waveform      = "SIN"
        self._target_amp    = 0.0
        self._filter_bright = 1.0   # 0 = dark/closed, 1 = open/bright
        self._reverb_wet    = 0.0   # 0 = dry, 1 = fully wet
        self._tremolo_depth = 0.0   # 0 = off, 1 = full tremolo

        # Synthesis state (audio thread only — no lock needed)
        self._phases      = [0.0]
        self._curr_amp    = 0.0
        self._filter_st   = 0.0
        self._comb_bufs   = [np.zeros(d) for d in _COMB_DELAYS]
        self._comb_pos    = [0] * len(_COMB_DELAYS)
        self._trem_phase  = 0.0

        self._stream = None

    # ── Public API ────────────────────────────────────────────────────────────
    def start(self) -> bool:
        if not _AUDIO_OK:
            return False
        try:
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=BLOCK_SIZE,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            return True
        except Exception:
            self._stream = None
            return False

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_params(self, active: bool, frequencies: list, waveform: str,
                   amplitude: float = 0.28, effects: dict = None):
        ef = effects or {}
        with self._lock:
            self._freqs         = list(frequencies) if frequencies else [440.0]
            self._waveform      = waveform
            self._target_amp    = float(amplitude) if active else 0.0
            self._filter_bright = float(ef.get("filter",  1.0))
            self._reverb_wet    = float(ef.get("reverb",  0.0))
            self._tremolo_depth = float(ef.get("tremolo", 0.0))

    # ── Audio callback (audio thread) ─────────────────────────────────────────
    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            freqs         = list(self._freqs)
            waveform      = self._waveform
            target_amp    = self._target_amp
            filter_bright = self._filter_bright
            reverb_wet    = self._reverb_wet
            tremolo_depth = self._tremolo_depth

        # 1. Additive oscillators
        n = len(freqs)
        if len(self._phases) != n:
            self._phases = [0.0] * n
        idx = np.arange(frames, dtype=np.float64)
        out = np.zeros(frames, dtype=np.float64)
        for i, freq in enumerate(freqs):
            t = self._phases[i] + idx * (freq / SAMPLE_RATE)
            out += _oscillate(waveform, t)
            self._phases[i] = float(t[-1] % 1.0)
        if n > 0:
            out /= n

        # 2. Amplitude envelope (smooth transitions, no clicks)
        a       = self._AMP_ALPHA
        decay   = np.power(1.0 - a, idx)
        amp_env = target_amp + (self._curr_amp - target_amp) * decay
        self._curr_amp = float(amp_env[-1])
        out *= amp_env

        # 3. Lowpass filter  (Y-axis control: filter_bright 0→dark, 1→open)
        if filter_bright < 0.999:
            cutoff = 200.0 * (8000.0 / 200.0) ** filter_bright  # 200–8000 Hz log scale
            out, self._filter_st = _lowpass(out, cutoff, self._filter_st)

        # 4. Schroeder reverb  (polygon area → wet)
        if reverb_wet > 0.001:
            out = self._comb_reverb(out, reverb_wet)

        # 5. Tremolo  (bounding-box aspect ratio → depth, fully vectorised)
        if tremolo_depth > 0.001:
            rate = 0.5 + tremolo_depth * 9.5          # 0.5–10 Hz
            lfo  = 0.5 + 0.5 * np.sin(2.0 * np.pi * rate * idx / SAMPLE_RATE + self._trem_phase)
            out *= 1.0 - tremolo_depth * 0.7 * (1.0 - lfo)
            self._trem_phase = (self._trem_phase + 2.0 * np.pi * rate * frames / SAMPLE_RATE) % (2.0 * np.pi)

        outdata[:, 0] = np.clip(out, -1.0, 1.0).astype(np.float32)

    # ── Schroeder comb reverb (4 parallel comb filters) ──────────────────────
    def _comb_reverb(self, dry: np.ndarray, wet: float) -> np.ndarray:
        n   = len(dry)
        rev = np.zeros(n, dtype=np.float64)
        for j in range(len(_COMB_DELAYS)):
            D   = _COMB_DELAYS[j]
            buf = self._comb_bufs[j]
            pos = self._comb_pos[j]
            for i in range(n):
                echo    = buf[(pos + 1) % D]
                buf[pos] = dry[i] + echo * _COMB_FEEDBACK
                rev[i]  += echo
                pos      = (pos + 1) % D
            self._comb_pos[j] = pos
        rev /= len(_COMB_DELAYS)
        return dry * (1.0 - wet * 0.4) + rev * wet
