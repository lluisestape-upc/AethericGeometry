"""Just-intonation chord ratios and pitch helpers.

Why exact fractions
-------------------
The point of just intonation is that partials of different voices coincide
*exactly*, so the beating between them vanishes. That property is destroyed by
rounding: the earlier table stored ``1.333`` and ``1.667`` for the perfect
fourth and the major sixth. Against the true ``4/3`` the error is

.. math:: 1200 \\log_2\\!\\left(\\frac{1.333}{4/3}\\right) \\approx -0.43\\ \\text{cents}

which is inaudible as pitch but is *not* inaudible as beating: a fourth above
220 Hz is 293.33 Hz exact and 293.26 Hz rounded, and the third partial of the
root then beats against the second partial of the fourth at roughly 0.15 Hz —
a slow wow across a sustained chord, exactly the artefact just intonation
exists to remove. Storing :class:`~fractions.Fraction` costs nothing and makes
the intent explicit.

What this tuning is and is not
------------------------------
These are *fixed ratios above a movable root*. The root itself is quantised to
a pentatonic subset of 12-TET (see :func:`quantize_pentatonic`), which means
the instrument is justly tuned **within** a chord but equal-tempered
**between** chords. That is a deliberate compromise and worth stating plainly:
a fully just system has no single answer for what happens when the root moves,
because stacking exact fifths never closes the octave — twelve of them
overshoot by the Pythagorean comma, :math:`(3/2)^{12}/2^7 \\approx 1.0136`, or
about 23.5 cents. Choosing a fixed ratio set over a tempered root sidesteps
comma drift at the cost of transposition purity.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

# ── Interval inventory ────────────────────────────────────────────────────────
UNISON = Fraction(1, 1)
MAJOR_SECOND = Fraction(9, 8)      # 203.9 c
MINOR_THIRD = Fraction(6, 5)       # 315.6 c
MAJOR_THIRD = Fraction(5, 4)       # 386.3 c
PERFECT_FOURTH = Fraction(4, 3)    # 498.0 c
PERFECT_FIFTH = Fraction(3, 2)     # 702.0 c
MAJOR_SIXTH = Fraction(5, 3)       # 884.4 c
HARMONIC_SEVENTH = Fraction(7, 4)  # 968.8 c
OCTAVE = Fraction(2, 1)

#: Chord voicings indexed by voice count. Each is a stack of exact ratios above
#: the root, built to add the next-simplest interval as voices are added: fifth,
#: then major third, then octave, then ninth, then harmonic seventh, then
#: fourth, then sixth. Simpler ratios enter first because their partials
#: coincide soonest and therefore fuse most strongly.
CHORD_RATIOS: dict[int, tuple[Fraction, ...]] = {
    1: (UNISON,),
    2: (UNISON, PERFECT_FIFTH),
    3: (UNISON, MAJOR_THIRD, PERFECT_FIFTH),
    4: (UNISON, MAJOR_THIRD, PERFECT_FIFTH, OCTAVE),
    5: (UNISON, MAJOR_SECOND, MAJOR_THIRD, PERFECT_FIFTH, OCTAVE),
    6: (UNISON, MAJOR_SECOND, MAJOR_THIRD, PERFECT_FIFTH, HARMONIC_SEVENTH,
        OCTAVE),
    7: (UNISON, MAJOR_SECOND, MAJOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH,
        HARMONIC_SEVENTH, OCTAVE),
    8: (UNISON, MAJOR_SECOND, MAJOR_THIRD, PERFECT_FOURTH, PERFECT_FIFTH,
        MAJOR_SIXTH, HARMONIC_SEVENTH, OCTAVE),
}

_INTERVAL_NAMES = {
    UNISON: "1/1 unison",
    MAJOR_SECOND: "9/8 major second",
    MINOR_THIRD: "6/5 minor third",
    MAJOR_THIRD: "5/4 major third",
    PERFECT_FOURTH: "4/3 perfect fourth",
    PERFECT_FIFTH: "3/2 perfect fifth",
    MAJOR_SIXTH: "5/3 major sixth",
    HARMONIC_SEVENTH: "7/4 harmonic seventh",
    OCTAVE: "2/1 octave",
}

_A2 = 110.0
_C0 = 16.351597831287414  # C0 = 440 * 2**(-57/12)
_PENTATONIC_STEPS = (0, 2, 4, 7, 9)
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def interval_name(ratio: Fraction) -> str:
    """Human-readable label for a ratio, e.g. ``'3/2 perfect fifth'``."""
    return _INTERVAL_NAMES.get(ratio, f"{ratio.numerator}/{ratio.denominator}")


def cents(ratio: Fraction | float) -> float:
    """Interval size in cents: :math:`1200 \\log_2 r`."""
    return 1200.0 * float(np.log2(float(ratio)))


def cents_from_12tet(ratio: Fraction | float) -> float:
    """Signed deviation of a ratio from its nearest 12-TET neighbour, in cents.

    This is the number that says how much a just interval "leans" — the just
    major third sits 13.7 cents flat of the tempered one, the fifth 2.0 cents
    sharp. Useful for the analysis write-up and for sanity-checking the table.
    """
    c = cents(ratio)
    return c - 100.0 * round(c / 100.0)


def chord_frequencies(root_hz: float, n_voices: int) -> list[float]:
    """Absolute frequencies for a chord of ``n_voices`` above ``root_hz``.

    Ratios are kept exact until the final multiply, so the only rounding is the
    single float conversion per partial.
    """
    n = int(np.clip(n_voices, 1, 8))
    ratios = CHORD_RATIOS[n]
    return [float(root_hz) * (r.numerator / r.denominator) for r in ratios]


def pitch_from_norm(t: float, lo: float = _A2, hi: float = 880.0) -> float:
    """Exponential map from a normalised control value to frequency.

    Pitch perception is logarithmic in frequency, so a linear map would make
    the low end of the gesture range musically cramped and the high end sparse.
    :math:`f = f_\\text{lo}(f_\\text{hi}/f_\\text{lo})^t` gives constant cents
    per unit of hand travel.
    """
    return float(lo) * (float(hi) / float(lo)) ** float(np.clip(t, 0.0, 1.0))


def quantize_pentatonic(freq: float) -> float:
    """Snap a frequency to the nearest note of the A minor pentatonic scale."""
    st = 12.0 * np.log2(max(float(freq), 1.0) / _A2)
    octave = int(st // 12)
    rem = st % 12
    candidates = list(_PENTATONIC_STEPS) + [s + 12 for s in _PENTATONIC_STEPS]
    nearest = min(candidates, key=lambda s: abs(s - rem))
    return _A2 * 2.0 ** ((octave * 12 + nearest) / 12.0)


def freq_to_note(freq: float) -> str:
    """Nearest 12-TET note name, e.g. ``'A3'``."""
    st = int(round(12.0 * np.log2(max(float(freq), 1.0) / _C0)))
    return f"{_NOTE_NAMES[st % 12]}{st // 12}"


def freq_to_midi(freq: float) -> int:
    """Nearest MIDI note number, clamped to the legal 0–127 range.

    Note the ``+ 12``. Semitones counted from :math:`C_0` are *not* MIDI note
    numbers: MIDI 0 is :math:`C_{-1}`, an octave below :math:`C_0`. The old code
    sent ``12*log2(f/C0)`` straight out as a note number, so every note the
    instrument emitted over MIDI was exactly one octave flat — audible the
    moment the output drove anything other than the internal synth.
    """
    st = int(round(12.0 * np.log2(max(float(freq), 1.0) / _C0)))
    return int(np.clip(st + 12, 0, 127))
