"""Signal-processing primitives for Aetheric Geometry.

Everything here is deliberately free of audio-hardware dependencies so the
blocks can be measured offline (see ``analysis/measure_dsp.py``) and unit
tested without a sound card.

The three ideas that shape this module:

1. **Band-limited oscillators.** A trivially generated sawtooth or square has
   harmonics at every multiple of :math:`f_0` with amplitude :math:`1/k`. All
   of the ones above :math:`f_s/2` fold back into the audible band as
   inharmonic partials. In an instrument whose pitch is a continuous function
   of hand position the aliases slide around as the hand moves, which is far
   more noticeable than static aliasing on a fixed note. PolyBLEP replaces the
   discontinuity with a two-sample polynomial approximation of a band-limited
   step, cancelling most of that energy for a couple of multiply-adds per
   sample.

2. **Prewarped filters.** The naive one-pole :math:`c = e^{-2\\pi f_c/f_s}` is
   only accurate while :math:`f_c \\ll f_s`. The TPT (topology-preserving
   transform) form below uses :math:`g = \\tan(\\pi f_c / f_s)` so the cutoff
   lands exactly where it is asked to, all the way up to Nyquist.

3. **Block-vectorised feedback delays.** A feedback comb ``y[n] = x[n] +
   g*y[n-D]`` looks like a sequential recursion, but when ``D`` is at least the
   processing chunk length every sample it reads is already in the past. The
   recursion then unrolls into a plain delay-line read plus a vector multiply.
   :func:`_chunks` splits each block at the delay-line wrap so that property
   always holds, which is what lets a Python audio callback run four combs and
   two allpasses without a per-sample loop.

References
----------
Välimäki & Huovilainen (2007), *Antialiasing oscillators in subtractive
synthesis*; Zavalishin (2012), *The Art of VA Filter Design*; Schroeder (1962),
*Natural sounding artificial reverberation*.
"""
from __future__ import annotations

import numpy as np

try:  # scipy gives us a C-speed IIR; the fallback keeps the module importable
    from scipy.signal import lfilter, lfilter_zi  # type: ignore
    _SCIPY = True
except ImportError:  # pragma: no cover - exercised only on installs without scipy
    _SCIPY = False


# ══════════════════════════════════════════════════════════════════════════════
#  First-order IIR helper
# ══════════════════════════════════════════════════════════════════════════════
def _iir1(b: np.ndarray, a: np.ndarray, x: np.ndarray, zi: np.ndarray):
    """Run a first-order IIR, returning ``(y, zi)``.

    Uses :func:`scipy.signal.lfilter` when available and falls back to an
    explicit loop otherwise, so behaviour is identical either way.
    """
    if _SCIPY:
        y, zf = lfilter(b, a, x, zi=zi)
        return y, zf
    y = np.empty_like(x)
    s = float(zi[0])
    b0, b1 = float(b[0]), float(b[1]) if len(b) > 1 else 0.0
    a1 = float(a[1]) if len(a) > 1 else 0.0
    for i in range(len(x)):
        y[i] = b0 * x[i] + s
        s = b1 * x[i] - a1 * y[i]
    return y, np.array([s])


# ══════════════════════════════════════════════════════════════════════════════
#  Band-limited oscillators (PolyBLEP)
# ══════════════════════════════════════════════════════════════════════════════
def poly_blep(phase: np.ndarray, dt: float) -> np.ndarray:
    """Two-sample polynomial BLEP residual, scaled for a jump of amplitude 2.

    ``phase`` is the normalised phase in ``[0, 1)`` and ``dt = f0/fs`` is the
    per-sample phase increment. The residual is non-zero only in the two-sample
    neighbourhood of the wrap point, where it approximates the difference
    between an ideal band-limited step and the hard step the naive waveform
    contains.
    """
    out = np.zeros_like(phase)
    if dt <= 0.0:
        return out

    # Sample lands just after the discontinuity.
    after = phase < dt
    if after.any():
        t = phase[after] / dt
        out[after] = t + t - t * t - 1.0

    # Sample lands just before it (i.e. the step happens within the next step).
    before = phase > 1.0 - dt
    if before.any():
        t = (phase[before] - 1.0) / dt
        out[before] = t * t + t + t + 1.0

    return out


def blep_saw(phase: np.ndarray, dt: float) -> np.ndarray:
    """Band-limited sawtooth, falling edge corrected."""
    return (2.0 * phase - 1.0) - poly_blep(phase, dt)


def blep_square(phase: np.ndarray, dt: float, duty: float = 0.5) -> np.ndarray:
    """Band-limited square/pulse: one BLEP per edge."""
    naive = np.where(phase < duty, 1.0, -1.0)
    rising = poly_blep(phase, dt)
    falling = poly_blep((phase - duty) % 1.0, dt)
    return naive + rising - falling


def naive_triangle(phase: np.ndarray | float):
    """Ideal (aliasing) triangle, used only to seed the integrator."""
    return np.where(phase < 0.5, 4.0 * phase - 1.0, 3.0 - 4.0 * phase)


class BlepTriangle:
    """Band-limited triangle, formed by integrating a BLEP square.

    A unit-amplitude triangle of unit period has slope :math:`\\pm 4` per unit
    phase, so integrating the square with a per-sample gain of :math:`4\\,dt`
    reproduces it at the right level. The integrator is made leaky
    (:math:`a = e^{-2\\pi f_\\text{leak}/f_s}`) to stop DC and numerical drift
    from accumulating; the price is a gentle droop below ``leak_hz``, which
    sits an octave below the instrument's lowest note.

    **Seeding the integrator matters.** Starting from a zero state integrates
    the square from nothing, so the output climbs :math:`0 \\to 2 \\to 0` and
    only settles onto :math:`\\pm 1` as the leak drains the offset — an 85 %
    overshoot lasting :math:`1/(2\\pi f_\\text{leak}) \\approx 16` ms, which is
    long enough to be heard as a thump on every new note and to eat headroom.
    The first call therefore seeds the state from the closed-form triangle
    value at the starting phase, so the very first sample is already correct.
    """

    def __init__(self, sample_rate: float, leak_hz: float = 10.0):
        self.sample_rate = float(sample_rate)
        self._a = float(np.exp(-2.0 * np.pi * leak_hz / self.sample_rate))
        self._zi = np.zeros(1)
        self._seeded = False

    def reset(self) -> None:
        self._zi = np.zeros(1)
        self._seeded = False

    def process(self, phase: np.ndarray, dt: float) -> np.ndarray:
        sq = blep_square(phase, dt)

        if not self._seeded and len(phase):
            # scipy's convention is y[0] = b0*x[0] + zi[0]; solve for the zi
            # that makes y[0] land on the true triangle value at this phase.
            self._zi = np.array([float(naive_triangle(phase[0]))
                                 - 4.0 * dt * float(sq[0])])
            self._seeded = True

        b = np.array([4.0 * dt, 0.0])
        a = np.array([1.0, -self._a])
        y, self._zi = _iir1(b, a, sq, self._zi)
        return y


# ══════════════════════════════════════════════════════════════════════════════
#  TPT one-pole lowpass
# ══════════════════════════════════════════════════════════════════════════════
class OnePoleTPT:
    """Topology-preserving-transform one-pole lowpass, 6 dB/oct per stage.

    With :math:`g = \\tan(\\pi f_c/f_s)` and :math:`G = g/(1+g)` the trapezoidal
    integrator collapses to the difference equation

    .. math:: y[n] = G\\,x[n] + G\\,x[n-1] + (1-2G)\\,y[n-1]

    which is exactly the bilinear transform of :math:`1/(1+s/\\omega_c)` with
    frequency prewarping, so :math:`|H|` is :math:`-3` dB at :math:`f_c` for any
    cutoff below Nyquist — unlike ``exp(-2*pi*fc/fs)``, which drifts sharp as
    the cutoff rises. Unity gain at DC and an exact zero at Nyquist follow by
    substituting :math:`z = 1` and :math:`z = -1`.
    """

    def __init__(self, sample_rate: float, stages: int = 2):
        self.sample_rate = float(sample_rate)
        self.stages = int(stages)
        self._zi = [np.zeros(1) for _ in range(self.stages)]
        self._cutoff = 0.0
        self._b = np.array([1.0, 0.0])
        self._a = np.array([1.0, 0.0])

    def reset(self) -> None:
        self._zi = [np.zeros(1) for _ in range(self.stages)]

    def set_cutoff(self, hz: float) -> None:
        hz = float(np.clip(hz, 10.0, self.sample_rate * 0.49))
        if hz == self._cutoff:
            return
        self._cutoff = hz
        g = np.tan(np.pi * hz / self.sample_rate)
        G = g / (1.0 + g)
        self._b = np.array([G, G])
        self._a = np.array([1.0, -(1.0 - 2.0 * G)])

    def process(self, x: np.ndarray) -> np.ndarray:
        y = x
        for i in range(self.stages):
            y, self._zi[i] = _iir1(self._b, self._a, y, self._zi[i])
        return y


class DCBlocker:
    """``y[n] = x[n] - x[n-1] + R*y[n-1]`` — a zero at DC, pole just inside it."""

    def __init__(self, sample_rate: float, cutoff_hz: float = 8.0):
        R = float(np.exp(-2.0 * np.pi * cutoff_hz / float(sample_rate)))
        self._b = np.array([1.0, -1.0])
        self._a = np.array([1.0, -R])
        self._zi = np.zeros(1)

    def reset(self) -> None:
        self._zi = np.zeros(1)

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self._zi = _iir1(self._b, self._a, x, self._zi)
        return y


# ══════════════════════════════════════════════════════════════════════════════
#  Delay-line building blocks
# ══════════════════════════════════════════════════════════════════════════════
def _chunks(pos: int, delay: int, n: int):
    """Yield ``(pos, length)`` slices that never cross the delay-line wrap.

    Because each slice is at most ``delay - pos`` long, every sample a feedback
    delay reads inside a slice was written at least ``delay`` samples ago. That
    is what makes the recursion vectorisable.
    """
    done = 0
    while done < n:
        run = min(delay - pos, n - done)
        yield pos, run, done
        done += run
        pos = (pos + run) % delay


def rt60_to_feedback(rt60_s: float, delay_samples: int, sample_rate: float) -> float:
    """Feedback gain giving a chosen :math:`T_{60}` for one comb.

    A comb of delay :math:`D` emits echoes every :math:`D/f_s` seconds, the
    :math:`n`-th attenuated by :math:`g^n`. Reaching :math:`-60` dB means
    :math:`g^n = 10^{-3}`, i.e. :math:`n = -3/\\log_{10} g`, so

    .. math:: T_{60} = \\frac{-3\\,D}{f_s \\log_{10} g}
              \\quad\\Longleftrightarrow\\quad
              g = 10^{-3D/(f_s T_{60})}

    Solving per comb rather than sharing one gain is the whole point: with a
    common :math:`g`, longer delay lines decay *slower* in wall-clock time, so
    the four combs finish at four different moments and the tail sounds like it
    is falling apart. Deriving each gain from a single target ties them
    together.
    """
    rt60_s = max(float(rt60_s), 1e-3)
    exponent = -3.0 * float(delay_samples) / (float(sample_rate) * rt60_s)
    return float(np.clip(10.0 ** exponent, 0.0, 0.999))


class DampedComb:
    """Feedback comb with a one-pole lowpass in the loop (Freeverb topology).

    ``y[n] = buf[n-D]``, ``buf[n] = x[n] + g * lp(y[n])``. The in-loop lowpass
    makes :math:`T_{60}` frequency-dependent — high partials die first, as they
    do in a real room, instead of the whole spectrum decaying as one block.
    """

    def __init__(self, delay: int, sample_rate: float):
        self.delay = int(delay)
        self.sample_rate = float(sample_rate)
        self._buf = np.zeros(self.delay)
        self._pos = 0
        self._g = 0.8
        self._damp = 0.2
        self._zi = np.zeros(1)

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._pos = 0
        self._zi = np.zeros(1)

    def set_rt60(self, rt60_s: float) -> None:
        self._g = rt60_to_feedback(rt60_s, self.delay, self.sample_rate)

    def set_damping(self, damp: float) -> None:
        self._damp = float(np.clip(damp, 0.0, 0.95))

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        b = np.array([1.0 - self._damp, 0.0])
        a = np.array([1.0, -self._damp])
        for pos, run, off in _chunks(self._pos, self.delay, len(x)):
            echo = self._buf[pos:pos + run].copy()
            damped, self._zi = _iir1(b, a, echo, self._zi)
            self._buf[pos:pos + run] = x[off:off + run] + self._g * damped
            out[off:off + run] = echo
        self._pos = (self._pos + len(x)) % self.delay
        return out


class SchroederAllpass:
    """Unity-magnitude allpass, single-delay-line form.

    .. math:: H(z) = \\frac{-g + z^{-D}}{1 - g z^{-D}}

    :math:`|H(e^{j\\omega})| = 1` for all :math:`\\omega`, so it colours nothing
    while scrambling phase. Chaining a couple of these after the comb bank is
    what turns four discrete echo trains into something that reads as diffuse:
    the combs set the decay, the allpasses set the echo density.
    """

    def __init__(self, delay: int, g: float = 0.5):
        self.delay = int(delay)
        self._buf = np.zeros(self.delay)
        self._pos = 0
        self._g = float(g)

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._pos = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        g = self._g
        for pos, run, off in _chunks(self._pos, self.delay, len(x)):
            v_del = self._buf[pos:pos + run].copy()
            v = x[off:off + run] + g * v_del
            self._buf[pos:pos + run] = v
            out[off:off + run] = v_del - g * v
        self._pos = (self._pos + len(x)) % self.delay
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Non-linearities and envelopes
# ══════════════════════════════════════════════════════════════════════════════
def soft_clip(x: np.ndarray) -> np.ndarray:
    """``tanh`` limiter.

    Hard clipping generates an infinite series of odd harmonics from the corner
    and every one above Nyquist aliases. ``tanh`` is :math:`C^\\infty`, so its
    harmonic series rolls off quickly and the audible aliasing from an
    occasional peak is far lower for the same amount of level control.
    """
    return np.tanh(x)


def tau_to_alpha(tau_s: float, sample_rate: float) -> float:
    """One-pole coefficient for a given time constant.

    :math:`\\alpha = 1 - e^{-1/(\\tau f_s)}`, so a parameter expressed in
    seconds means the same thing at 44.1 and 48 kHz. The previous code stored
    raw per-sample coefficients, which silently changed every envelope time
    whenever the device sample rate did.
    """
    tau_s = max(float(tau_s), 1e-6)
    return float(1.0 - np.exp(-1.0 / (tau_s * float(sample_rate))))
