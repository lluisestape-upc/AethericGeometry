"""Audio synthesis engine for Aetheric Geometry.

Signal chain
------------
``partials → sum/N → envelope → drive → 2-pole TPT lowpass → Schroeder reverb
→ tremolo → DC blocker → tanh limiter``

Notable corrections against the first version of this engine, all of which are
measurable and are reproduced by ``analysis/measure_dsp.py``:

* **Phase accumulation was one sample short per block.** The oscillator wrote
  back the phase of the *last* sample of the block rather than the phase the
  *next* block should start from, so every block replayed one sample of phase.
  With a 512-sample block the oscillator advanced 511 increments where it owed
  512 — a constant detuning of :math:`1200\\log_2(511/512) = -3.4` cents plus a
  phase discontinuity repeating at :math:`f_s/N = 86.1` Hz. In an instrument
  built around exact frequency ratios, being globally 3.4 cents flat with an
  86 Hz buzz on top rather defeats the purpose.

* **Oscillators were not band-limited.** See :mod:`dsp`.

* **The four comb filters shared one feedback gain**, which means four
  different decay times (:math:`T_{60} \\propto D` at fixed :math:`g`). Each
  comb now solves for its own gain from a single :math:`T_{60}` target, so the
  tail decays as one object. Allpass diffusers were missing entirely.

* **Envelope coefficients were raw per-sample constants**, so every envelope
  time changed silently with the device sample rate. They are time constants in
  seconds now.

* **MIDI note numbers were an octave flat.** See :func:`tuning.freq_to_midi`.
"""
from __future__ import annotations

import logging
import threading
import wave

import numpy as np

from .dsp import (
    BlepTriangle, DCBlocker, DampedComb, OnePoleTPT, SchroederAllpass,
    blep_saw, blep_square, soft_clip, tau_to_alpha,
)
# Re-exported so existing importers keep working.
from .tuning import (  # noqa: F401
    CHORD_RATIOS, chord_frequencies, freq_to_midi, freq_to_note,
    pitch_from_norm, quantize_pentatonic,
)

log = logging.getLogger(__name__)

try:
    import sounddevice as sd
    _AUDIO_OK = True
except ImportError:
    _AUDIO_OK = False
    log.warning("sounddevice not installed — running visual-only. pip install sounddevice")

try:
    import mido
    _MIDI_OK = True
except ImportError:
    _MIDI_OK = False

from .config import (
    SAMPLE_RATE, BLOCK_SIZE,
    ATTACK_TAU, RELEASE_TAU, REVERB_TAIL_TAU,
    REVERB_RT60_MIN, REVERB_RT60_MAX,
    FILTER_MIN_HZ, FILTER_MAX_HZ, DRIVE_MAX,
    MIDI_PORT,
)

# Schroeder/Freeverb delay lengths, quoted at 44.1 kHz and rescaled below.
# They are mutually near-prime so the four echo trains take a long time to
# line up; coincident echoes are what a listener hears as a metallic ring.
_COMB_DELAYS_44K = (1557, 1617, 1491, 1422)
_ALLPASS_DELAYS_44K = (556, 225)
_ALLPASS_G = 0.5

#: Waveforms in order of harmonic content, so morphing across them is a
#: monotonic brightness ramp rather than an arbitrary tour.
#:
#: Sine has one partial, triangle rolls off as 1/k^2, sawtooth as 1/k with every
#: harmonic present, square as 1/k with odd harmonics only but a far harsher
#: edge. Ordering them this way makes the shape knob behave like a single
#: "edge" control: turning it up always adds bite, never takes some away and
#: gives other back.
WAVE_RAMP = ("SIN", "TRIANGLE", "SAW", "SQUARE")

#: Knob-controllable effects, each normalised to 0..1 with its default.
#: ``shape`` leads because it is the sound itself rather than something done
#: to it afterwards.
EFFECT_DEFAULTS: dict[str, float] = {
    "shape": 0.0,     # 0 = sine, 1 = square, continuous in between
    "filter": 1.0,    # 0 = dark, 1 = fully open
    "reverb": 0.0,    # wet mix
    "decay": 0.35,    # maps to T60
    "damping": 0.35,  # tone of the tail
    "tremolo": 0.0,
    "drive": 0.0,
}
EFFECT_ORDER = ("shape", "filter", "reverb", "decay", "damping",
                "tremolo", "drive")


def _scaled(delays: tuple, sample_rate: float) -> tuple:
    """Rescale 44.1 kHz delay lengths to the running sample rate."""
    factor = float(sample_rate) / 44100.0
    return tuple(max(4, int(round(d * factor))) for d in delays)


class SynthEngine:
    """Additive engine driven from the gesture thread, rendered on the audio thread."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)

        self._lock = threading.Lock()
        self._rec_lock = threading.Lock()

        # ── Shared parameters (gesture thread → audio thread) ─────────────────
        self._freqs: list[float] = [440.0]
        self._waveform = "SIN"
        self._target_amp = 0.0
        self._effects: dict[str, float] = dict(EFFECT_DEFAULTS)
        self._held = False

        # ── Synthesis state (audio thread only) ───────────────────────────────
        self._phases: list[float] = [0.0]
        self._triangles: list[BlepTriangle] = []
        self._curr_amp = 0.0
        self._trem_phase = 0.0
        self._wet_smooth = 0.0

        self._filter = OnePoleTPT(self.sample_rate, stages=2)
        self._dc = DCBlocker(self.sample_rate)
        self._combs = [DampedComb(d, self.sample_rate)
                       for d in _scaled(_COMB_DELAYS_44K, self.sample_rate)]
        self._allpasses = [SchroederAllpass(d, _ALLPASS_G)
                           for d in _scaled(_ALLPASS_DELAYS_44K, self.sample_rate)]
        self._rt60_applied = -1.0
        self._damp_applied = -1.0

        # Envelope coefficients from time constants, so they survive a sample
        # rate change.
        self._attack_a = tau_to_alpha(ATTACK_TAU, self.sample_rate)
        self._release_a = tau_to_alpha(RELEASE_TAU, self.sample_rate)
        self._tail_a = tau_to_alpha(REVERB_TAIL_TAU, self.sample_rate)

        # ── Recording ─────────────────────────────────────────────────────────
        self._recording = False
        self._rec_buf: list[np.ndarray] = []

        self._stream = None
        self._midi_out = None
        self._last_midi_note: int | None = None
        self._init_midi()

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def _init_midi(self) -> None:
        if not _MIDI_OK or not MIDI_PORT:
            return
        try:
            self._midi_out = mido.open_output(MIDI_PORT)
            log.info("MIDI output opened: %s", MIDI_PORT)
        except Exception as exc:
            log.warning("Could not open MIDI port '%s': %s", MIDI_PORT, exc)

    def start(self) -> bool:
        if not _AUDIO_OK:
            return False
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.block_size,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            log.info("Audio stream started — %d Hz, block %d",
                     self.sample_rate, self.block_size)
            return True
        except Exception as exc:
            log.error("Failed to start audio stream: %s", exc)
            self._stream = None
            return False

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._midi_out:
            self._midi_out.close()
            self._midi_out = None

    # ── Parameter API ─────────────────────────────────────────────────────────
    def set_params(self, active: bool, frequencies: list, waveform: str,
                   amplitude: float = 0.28, effects: dict | None = None) -> None:
        """Push a new gesture frame.

        Ignored entirely while the sound is held: that is what "held" means —
        the hands stop driving pitch and level so they are free to do something
        else. Effects keep moving, but only through :meth:`set_effect`.
        """
        with self._lock:
            if self._held:
                return
            self._freqs = list(frequencies) if frequencies else [440.0]
            self._target_amp = float(amplitude) if active else 0.0
            # Only snap the morph when the named shape actually changes. This
            # runs every frame, so snapping unconditionally would drag the
            # shape knob back to an integer point the moment it was turned.
            if waveform != self._waveform:
                self._waveform = waveform
                if waveform in WAVE_RAMP:
                    self._effects["shape"] = (WAVE_RAMP.index(waveform)
                                              / (len(WAVE_RAMP) - 1))
            if effects:
                for key, value in effects.items():
                    if key in self._effects:
                        self._effects[key] = float(np.clip(value, 0.0, 1.0))

    def set_waveform(self, waveform: str) -> None:
        """Snap to one of the four named shapes, sounding or not, held or not.

        Deliberately not gated on :attr:`_held`, unlike :meth:`set_params`.
        Auditioning a shape against a chord you are already holding is the
        whole point of being able to say it out loud, and the phase
        accumulators are untouched, so the change is seamless rather than a
        retrigger.

        Naming a shape also moves the ``shape`` morph to that exact point, so
        the two controls never disagree: say "square", then turn the knob back,
        and you slide out of square rather than jumping somewhere unrelated.
        """
        name = str(waveform)
        with self._lock:
            self._waveform = name
            if name in WAVE_RAMP:
                self._effects["shape"] = (WAVE_RAMP.index(name)
                                          / (len(WAVE_RAMP) - 1))

    def set_effect(self, name: str, value: float) -> None:
        """Set one normalised effect value. Works held or not."""
        if name not in EFFECT_DEFAULTS:
            return
        with self._lock:
            self._effects[name] = float(np.clip(value, 0.0, 1.0))

    def get_effect(self, name: str) -> float:
        with self._lock:
            return float(self._effects.get(name, 0.0))

    def snapshot_effects(self) -> dict[str, float]:
        with self._lock:
            return dict(self._effects)

    # ── Hold ──────────────────────────────────────────────────────────────────
    def hold(self) -> bool:
        """Freeze the current note so it sustains with no hands on it.

        Returns ``False`` if there is nothing sounding to hold, which lets the
        caller give feedback rather than silently arming an empty hold.
        """
        with self._lock:
            if self._target_amp <= 0.0 or not self._freqs:
                return False
            self._held = True
            log.info("HOLD engaged — %d partials, root %.1f Hz",
                     len(self._freqs), self._freqs[0])
            return True

    def release_hold(self) -> None:
        with self._lock:
            if not self._held:
                return
            self._held = False
            self._target_amp = 0.0
            log.info("HOLD released")

    @property
    def is_held(self) -> bool:
        with self._lock:
            return self._held

    # ── MIDI ──────────────────────────────────────────────────────────────────
    def send_midi_note(self, freq: float | None = None) -> None:
        if not self._midi_out:
            return
        try:
            if freq is None:
                if self._last_midi_note is not None:
                    self._midi_out.send(
                        mido.Message("note_off", note=self._last_midi_note, velocity=0))
                    self._last_midi_note = None
                return
            note = freq_to_midi(freq)
            if note != self._last_midi_note:
                if self._last_midi_note is not None:
                    self._midi_out.send(
                        mido.Message("note_off", note=self._last_midi_note, velocity=0))
                self._midi_out.send(mido.Message("note_on", note=note, velocity=80))
                self._last_midi_note = note
        except Exception as exc:
            log.debug("MIDI send error: %s", exc)

    # ── Recording ─────────────────────────────────────────────────────────────
    def start_recording(self) -> None:
        with self._rec_lock:
            self._rec_buf.clear()
            self._recording = True
        log.info("Recording started")

    def stop_recording(self, path: str = "recording.wav") -> str:
        with self._rec_lock:
            self._recording = False
            chunks = self._rec_buf
            self._rec_buf = []
        if not chunks:
            log.warning("Nothing to save — recording buffer is empty.")
            return ""
        data = np.concatenate(chunks)
        pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        log.info("Recording saved: %s (%d samples)", path, len(pcm))
        return path

    # ── Oscillator bank ───────────────────────────────────────────────────────
    def _wave(self, kind: str, index: int, phase: np.ndarray,
              inc: float) -> np.ndarray:
        """One partial's samples for a named waveform, phase-aligned.

        All four are returned with their fundamental in phase with
        :math:`\\sin(2\\pi p)`. That alignment is invisible on a single shape -
        inverting a sawtooth or shifting a triangle by a quarter cycle is
        inaudible on its own - but it is the difference between a morph that
        works and one that does not.

        Blended in their natural forms, the shapes partially cancel: a rising
        ramp is negative through the first half of the cycle while a square is
        positive through it, so mixing them subtracts. Measured, the SAW to
        SQUARE segment collapsed to 14 dB below either endpoint, which reads as
        the volume dropping out in the middle of the knob's travel.

        Hence the negated saw (a falling ramp has the opposite-signed
        fundamental) and the quarter-cycle offset on the triangle (whose
        integrator otherwise peaks at mid-cycle, in quadrature with the rest).
        """
        if kind == "SIN":
            return np.sin(2.0 * np.pi * phase)
        if kind == "SAW":
            return -blep_saw(phase, inc)
        if kind == "SQUARE":
            return blep_square(phase, inc)
        return self._triangles[index].process((phase + 0.25) % 1.0, inc)

    def _render_partials(self, freqs: list, shape: float,
                         frames: int) -> np.ndarray:
        """Render the partial bank, morphing continuously across the ramp.

        ``shape`` is a position along :data:`WAVE_RAMP`, so the four named
        waveforms are just the integer points of one continuous control. The
        blend is a plain crossfade between the two neighbouring shapes: both
        are band-limited and share a phase accumulator, so their sum is
        band-limited too and no new aliasing is introduced by mixing them.

        The triangle keeps its integrator state per partial even when it is
        only one half of a blend, so morphing through it does not restart it.
        """
        n = len(freqs)
        if len(self._phases) != n:
            self._phases = (self._phases + [0.0] * n)[:n]
        while len(self._triangles) < n:
            self._triangles.append(BlepTriangle(self.sample_rate))

        pos = float(np.clip(shape, 0.0, 1.0)) * (len(WAVE_RAMP) - 1)
        low = int(pos)
        high = min(low + 1, len(WAVE_RAMP) - 1)
        blend = pos - low
        kind_a, kind_b = WAVE_RAMP[low], WAVE_RAMP[high]

        ramp = np.arange(frames, dtype=np.float64)
        out = np.zeros(frames, dtype=np.float64)

        for i, freq in enumerate(freqs):
            inc = float(np.clip(freq, 0.0, self.sample_rate * 0.49)) / self.sample_rate
            phase = (self._phases[i] + ramp * inc) % 1.0

            a = self._wave(kind_a, i, phase, inc)
            if blend > 1e-4 and kind_b != kind_a:
                b = self._wave(kind_b, i, phase, inc)
                out += a + blend * (b - a)
            else:
                out += a
                # Keep the triangle's integrator moving even when it is not
                # being heard, so morphing into it starts from the right place
                # instead of from a cold integrator.
                if kind_a != "TRIANGLE" and "TRIANGLE" in (kind_a, kind_b):
                    self._triangles[i].process(phase, inc)

            # Advance by the full block: the next block starts one increment
            # past the last sample rendered, not on it.
            self._phases[i] = (self._phases[i] + frames * inc) % 1.0

        if n > 0:
            out /= n
        return out

    # ── Reverb ────────────────────────────────────────────────────────────────
    def _reverb(self, dry: np.ndarray, wet: float, decay: float, damp: float) -> np.ndarray:
        rt60 = REVERB_RT60_MIN + decay * (REVERB_RT60_MAX - REVERB_RT60_MIN)
        if abs(rt60 - self._rt60_applied) > 1e-3:
            for comb in self._combs:
                comb.set_rt60(rt60)
            self._rt60_applied = rt60
        if abs(damp - self._damp_applied) > 1e-3:
            for comb in self._combs:
                comb.set_damping(damp * 0.9)
            self._damp_applied = damp

        acc = np.zeros_like(dry)
        for comb in self._combs:
            acc += comb.process(dry)
        acc /= len(self._combs)

        for ap in self._allpasses:
            acc = ap.process(acc)

        return dry * (1.0 - 0.5 * wet) + acc * wet

    # ── Audio callback ────────────────────────────────────────────────────────
    def _callback(self, outdata, frames, time_info, status):  # noqa: ARG002
        with self._lock:
            freqs = list(self._freqs)
            target_amp = self._target_amp
            fx = dict(self._effects)

        out = self._render_partials(freqs, fx.get("shape", 0.0), frames)

        # Asymmetric one-pole envelope: fast attack, slower release.
        ramp = np.arange(frames, dtype=np.float64)
        a = self._attack_a if target_amp > self._curr_amp else self._release_a
        env = target_amp + (self._curr_amp - target_amp) * np.power(1.0 - a, ramp)
        self._curr_amp = float(target_amp + (self._curr_amp - target_amp)
                               * (1.0 - a) ** frames)
        out *= env

        drive = fx.get("drive", 0.0)
        if drive > 1e-3:
            gain = 1.0 + drive * (DRIVE_MAX - 1.0)
            # Divide by tanh(gain) so the peak level stays put as drive rises;
            # otherwise the knob doubles as a volume control.
            out = np.tanh(out * gain) / np.tanh(gain)

        bright = fx.get("filter", 1.0)
        if bright < 0.999:
            self._filter.set_cutoff(
                FILTER_MIN_HZ * (FILTER_MAX_HZ / FILTER_MIN_HZ) ** bright)
            out = self._filter.process(out)

        # Smooth the wet mix so a released hold rings out instead of stopping.
        target_wet = fx.get("reverb", 0.0)
        self._wet_smooth += self._tail_a * frames * (target_wet - self._wet_smooth)
        self._wet_smooth = float(np.clip(self._wet_smooth, 0.0, 1.0))
        if self._wet_smooth > 1e-3:
            out = self._reverb(out, self._wet_smooth,
                               fx.get("decay", 0.35), fx.get("damping", 0.35))

        depth = fx.get("tremolo", 0.0)
        if depth > 1e-3:
            rate = 0.5 + depth * 9.5
            lfo = 0.5 + 0.5 * np.sin(
                2.0 * np.pi * rate * ramp / self.sample_rate + self._trem_phase)
            out *= 1.0 - depth * 0.7 * (1.0 - lfo)
            self._trem_phase = float(
                (self._trem_phase + 2.0 * np.pi * rate * frames / self.sample_rate)
                % (2.0 * np.pi))

        out = soft_clip(self._dc.process(out))
        block = out.astype(np.float32)
        outdata[:, 0] = block

        if self._recording:
            with self._rec_lock:
                if self._recording:
                    self._rec_buf.append(block.copy())
