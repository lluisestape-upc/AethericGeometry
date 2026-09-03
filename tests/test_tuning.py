"""Unit tests for tuning.py — exactness is the whole point, so test for it."""
import math
from fractions import Fraction

import pytest

from aetheric.tuning import (
    CHORD_RATIOS, MAJOR_THIRD, OCTAVE, PERFECT_FIFTH, PERFECT_FOURTH,
    cents, cents_from_12tet, chord_frequencies, freq_to_midi, freq_to_note,
    interval_name, pitch_from_norm, quantize_pentatonic,
)


class TestRatiosAreExact:
    def test_every_ratio_is_a_fraction(self):
        """Floats would reintroduce exactly the error this module exists to
        remove."""
        for ratios in CHORD_RATIOS.values():
            assert all(isinstance(r, Fraction) for r in ratios)

    def test_fifth_is_exactly_three_halves(self):
        assert PERFECT_FIFTH == Fraction(3, 2)

    def test_rounded_fourth_was_measurably_wrong(self):
        """1.333 is 0.43 cents flat of 4/3 — small as pitch, audible as
        beating on a sustained chord."""
        err = 1200.0 * math.log2(1.333 / float(PERFECT_FOURTH))
        assert err == pytest.approx(-0.4329, abs=1e-3)

    def test_chord_frequencies_are_exact_multiples(self):
        root = 220.0
        freqs = chord_frequencies(root, 3)
        assert freqs[1] == pytest.approx(root * 5 / 4, rel=1e-15)
        assert freqs[2] == pytest.approx(root * 3 / 2, rel=1e-15)

    def test_partials_coincide_exactly(self):
        """Just intonation earns its name here: the 3rd partial of the root and
        the 2nd of the fifth must be the same number, not nearly."""
        root = 220.0
        fifth = root * float(PERFECT_FIFTH)
        assert 2.0 * fifth == pytest.approx(3.0 * root, rel=1e-15)


class TestCents:
    def test_octave_is_1200_cents(self):
        assert cents(OCTAVE) == pytest.approx(1200.0)

    def test_fifth_is_702(self):
        assert cents(PERFECT_FIFTH) == pytest.approx(701.955, abs=1e-3)

    def test_just_third_leans_flat_of_tempered(self):
        assert cents_from_12tet(MAJOR_THIRD) == pytest.approx(-13.686, abs=1e-3)

    def test_just_fifth_leans_sharp(self):
        assert cents_from_12tet(PERFECT_FIFTH) == pytest.approx(1.955, abs=1e-3)

    def test_pythagorean_comma(self):
        """Twelve just fifths overshoot seven octaves by ~23.46 cents — the
        reason a movable root cannot also be justly transposed."""
        comma = (Fraction(3, 2) ** 12) / (Fraction(2, 1) ** 7)
        assert cents(comma) == pytest.approx(23.460, abs=1e-3)


class TestChordTable:
    def test_voice_counts_one_to_eight(self):
        assert set(CHORD_RATIOS) == set(range(1, 9))

    def test_length_matches_key(self):
        for n, ratios in CHORD_RATIOS.items():
            assert len(ratios) == n

    def test_starts_on_the_root_and_ascends(self):
        for ratios in CHORD_RATIOS.values():
            assert ratios[0] == Fraction(1, 1)
            assert list(ratios) == sorted(ratios)

    def test_no_duplicates(self):
        for ratios in CHORD_RATIOS.values():
            assert len(set(ratios)) == len(ratios)

    def test_within_one_octave(self):
        for ratios in CHORD_RATIOS.values():
            assert all(Fraction(1, 1) <= r <= Fraction(2, 1) for r in ratios)

    def test_clamps_out_of_range_voice_counts(self):
        assert len(chord_frequencies(220.0, 0)) == 1
        assert len(chord_frequencies(220.0, 99)) == 8

    def test_interval_names(self):
        assert "fifth" in interval_name(PERFECT_FIFTH)
        assert interval_name(Fraction(11, 8)) == "11/8"


class TestPitchHelpers:
    def test_pitch_map_is_exponential(self):
        lo, hi = 110.0, 880.0
        assert pitch_from_norm(0.5, lo, hi) == pytest.approx(math.sqrt(lo * hi))

    def test_equal_steps_give_equal_cents(self):
        f = [pitch_from_norm(t, 110.0, 880.0) for t in (0.0, 0.25, 0.5, 0.75)]
        steps = [1200.0 * math.log2(b / a) for a, b in zip(f, f[1:])]
        assert steps[0] == pytest.approx(steps[1]) == pytest.approx(steps[2])

    def test_quantize_lands_on_the_scale(self):
        for f in (95.0, 137.0, 260.0, 505.0, 900.0):
            st = round(12.0 * math.log2(quantize_pentatonic(f) / 110.0))
            assert st % 12 in (0, 2, 4, 7, 9)

    def test_note_names(self):
        assert freq_to_note(440.0) == "A4"
        assert freq_to_note(261.6256) == "C4"

    def test_midi_numbers_are_not_an_octave_flat(self):
        """The old code emitted 12*log2(f/C0) directly; MIDI 0 is C-1, not C0,
        so every note went out an octave low."""
        assert freq_to_midi(440.0) == 69
        assert freq_to_midi(261.6256) == 60
        assert freq_to_midi(8.0) == 0        # clamped, not negative
        assert freq_to_midi(30000.0) == 127  # clamped, not out of range
