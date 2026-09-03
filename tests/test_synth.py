"""Unit tests for synth.py — pitch helpers and waveform generation."""
import numpy as np
import pytest

from aetheric.synth import (
    pitch_from_norm, quantize_pentatonic, freq_to_note, CHORD_RATIOS,
)

_A2 = 110.0
_PENTA_FREQS = [_A2 * 2.0 ** (s / 12.0) for s in [0, 2, 4, 7, 9]]


# ── pitch_from_norm ───────────────────────────────────────────────────────────
class TestPitchFromNorm:
    def test_zero_returns_lo(self):
        assert pitch_from_norm(0.0, lo=100.0, hi=1000.0) == pytest.approx(100.0)

    def test_one_returns_hi(self):
        assert pitch_from_norm(1.0, lo=100.0, hi=1000.0) == pytest.approx(1000.0)

    def test_half_is_geometric_mean(self):
        lo, hi   = 100.0, 1000.0
        expected = (lo * hi) ** 0.5
        assert pitch_from_norm(0.5, lo=lo, hi=hi) == pytest.approx(expected, rel=1e-3)

    def test_clamps_below_zero(self):
        assert pitch_from_norm(-1.0, lo=100.0, hi=1000.0) == pytest.approx(100.0)

    def test_clamps_above_one(self):
        assert pitch_from_norm(2.0, lo=100.0, hi=1000.0) == pytest.approx(1000.0)

    def test_monotone_increasing(self):
        freqs = [pitch_from_norm(t) for t in np.linspace(0, 1, 20)]
        assert all(a <= b for a, b in zip(freqs, freqs[1:]))


# ── quantize_pentatonic ───────────────────────────────────────────────────────
class TestQuantizePentatonic:
    def test_output_on_scale(self):
        for freq_in in np.linspace(80, 1200, 40):
            result = quantize_pentatonic(freq_in)
            # Result must be A2 * 2^(n/12) for some integer n, and n % 12 in penta
            ratio  = result / _A2
            st     = round(12.0 * np.log2(ratio))
            assert st % 12 in [0, 2, 4, 7, 9], (
                f"freq {freq_in:.1f} → {result:.2f} Hz (semitone {st % 12} not in pentatonic)"
            )

    def test_already_on_scale(self):
        for base in _PENTA_FREQS:
            assert quantize_pentatonic(base) == pytest.approx(base, rel=1e-4)

    def test_positive_output(self):
        assert quantize_pentatonic(1.0) > 0


# ── freq_to_note ──────────────────────────────────────────────────────────────
class TestFreqToNote:
    def test_a4(self):
        # A4 = 440 Hz
        assert freq_to_note(440.0) == "A4"

    def test_c4(self):
        # C4 ≈ 261.63 Hz
        assert freq_to_note(261.63) == "C4"

    def test_format(self):
        note = freq_to_note(440.0)
        # Must be note name followed by octave number
        assert note[:-1] in ["C", "C#", "D", "D#", "E", "F",
                              "F#", "G", "G#", "A", "A#", "B"]
        assert note[-1].isdigit()


# ── Oscillator bank ───────────────────────────────────────────────────────────
class TestOscillatorBank:
    """The engine's oscillator bank, exercised without a sound card."""

    @staticmethod
    def _engine():
        from aetheric.synth import SynthEngine
        return SynthEngine()

    @staticmethod
    def _shape_of(name):
        from aetheric.synth import WAVE_RAMP
        return WAVE_RAMP.index(name) / (len(WAVE_RAMP) - 1)

    @pytest.mark.parametrize("wave", ["SIN", "SAW", "SQUARE", "TRIANGLE"])
    def test_output_is_bounded_and_finite(self, wave):
        eng = self._engine()
        out = eng._render_partials([220.0], self._shape_of(wave), 1024)
        assert np.isfinite(out).all()
        # PolyBLEP overshoots slightly at the corners; 1.2 is ample headroom.
        assert np.max(np.abs(out)) < 1.2

    def test_phase_advances_by_a_full_block(self):
        """The classic off-by-one: the next block starts one increment past
        the last sample rendered, not on it."""
        eng = self._engine()
        freq, frames = 440.0, 512
        eng._render_partials([freq], self._shape_of("SAW"), frames)
        expected = (frames * freq / eng.sample_rate) % 1.0
        assert eng._phases[0] == pytest.approx(expected, abs=1e-12)

    def test_block_boundary_is_seamless(self):
        """Rendering 2x512 must equal rendering 1024 in one go."""
        s = self._shape_of("SAW")
        a = self._engine()
        joined = np.concatenate([a._render_partials([311.0], s, 512),
                                 a._render_partials([311.0], s, 512)])
        b = self._engine()
        whole = b._render_partials([311.0], s, 1024)
        np.testing.assert_allclose(joined, whole, atol=1e-12)

    def test_partials_are_averaged(self):
        s = self._shape_of("SIN")
        eng = self._engine()
        one = eng._render_partials([220.0], s, 256)
        eng2 = self._engine()
        many = eng2._render_partials([220.0] * 4, s, 256)
        # Four identical partials averaged must equal one of them.
        np.testing.assert_allclose(many, one, atol=1e-12)


# ── Waveform morph ────────────────────────────────────────────────────────────
class TestShapeMorph:
    @staticmethod
    def _engine():
        from aetheric.synth import SynthEngine
        return SynthEngine()

    def test_integer_points_are_the_named_waveforms(self):
        """The four shapes are just the whole-number stops of one control."""
        from aetheric.dsp import BlepTriangle
        from aetheric.synth import WAVE_RAMP
        inc = 220.0 / 44100.0
        phase = (np.arange(512) * inc) % 1.0
        for i, name in enumerate(WAVE_RAMP):
            a = self._engine()
            b = self._engine()
            b._triangles = [BlepTriangle(b.sample_rate)]   # _wave assumes one
            morphed = a._render_partials([220.0], i / (len(WAVE_RAMP) - 1), 512)
            named = b._wave(name, 0, phase, inc)
            np.testing.assert_allclose(morphed, named, atol=1e-12)

    def test_halfway_is_between_its_neighbours(self):
        """A blend must sit between the two shapes it mixes, sample by sample."""
        from aetheric.synth import WAVE_RAMP
        step = 1.0 / (len(WAVE_RAMP) - 1)
        lo = self._engine()._render_partials([220.0], 0.0, 512)
        hi = self._engine()._render_partials([220.0], step, 512)
        mid = self._engine()._render_partials([220.0], step * 0.5, 512)
        between = ((mid >= np.minimum(lo, hi) - 1e-9)
                   & (mid <= np.maximum(lo, hi) + 1e-9))
        assert between.all()

    def test_morph_is_continuous(self):
        """No audible step anywhere along the sweep, including where the pair
        of blended shapes changes over."""
        prev = None
        largest = 0.0
        for pos in np.linspace(0.0, 1.0, 61):
            block = self._engine()._render_partials([220.0], float(pos), 256)
            if prev is not None:
                largest = max(largest, float(np.max(np.abs(block - prev))))
            prev = block
        assert largest < 0.25, largest

    def test_no_level_notch_across_the_sweep(self):
        """The bug phase alignment fixes.

        Blended in their natural forms the shapes partially cancel - a rising
        ramp is negative through the first half of a cycle while a square is
        positive through it - and the SAW-to-SQUARE segment measured 14 dB
        below either endpoint, which sounds like the volume dropping out
        mid-travel. No point in the sweep may sit far below its neighbours.
        """
        levels = []
        for pos in np.linspace(0.0, 1.0, 41):
            block = self._engine()._render_partials([220.0], float(pos), 4096)
            levels.append(float(np.sqrt(np.mean(block ** 2))))

        for i in range(1, len(levels) - 1):
            floor = min(levels[i - 1], levels[i + 1])
            assert levels[i] > floor * 0.9, (
                f"level notch at {i / (len(levels) - 1):.2f}: "
                f"{levels[i]:.3f} against {floor:.3f}")

        # And nothing anywhere may fall far below the quietest pure shape.
        assert min(levels) > 0.45, min(levels)

    def test_shapes_share_a_fundamental_phase(self):
        """What makes the blend add instead of subtract: every shape's
        fundamental must point the same way."""
        from aetheric.synth import WAVE_RAMP
        from aetheric.dsp import BlepTriangle
        inc = 220.0 / 44100.0
        n = 4096
        phase = (np.arange(n) * inc) % 1.0
        reference = np.sin(2.0 * np.pi * phase)
        for name in WAVE_RAMP:
            eng = self._engine()
            eng._triangles = [BlepTriangle(eng.sample_rate)]
            x = eng._wave(name, 0, phase, inc)
            # Projection onto the reference sine: positive means in phase.
            assert float(np.dot(x, reference)) > 0.0, name

    def test_naming_a_shape_snaps_the_morph(self):
        """Say "square", then turn the knob back, and you slide out of square
        rather than jumping somewhere unrelated."""
        eng = self._engine()
        eng.set_waveform("SQUARE")
        assert eng.get_effect("shape") == pytest.approx(1.0)
        eng.set_waveform("SIN")
        assert eng.get_effect("shape") == pytest.approx(0.0)

    def test_gesture_frames_do_not_drag_the_knob_back(self):
        """set_params runs every frame; snapping unconditionally would undo a
        turn of the shape knob the instant it happened."""
        eng = self._engine()
        eng.set_params(True, [220.0], "SIN", amplitude=0.3)
        eng.set_effect("shape", 0.42)
        for _ in range(10):
            eng.set_params(True, [220.0], "SIN", amplitude=0.3)
        assert eng.get_effect("shape") == pytest.approx(0.42)

    def test_shape_is_reachable_while_held(self):
        eng = self._engine()
        eng.set_params(True, [220.0], "SIN", amplitude=0.3)
        eng.hold()
        eng.set_effect("shape", 0.8)
        assert eng.get_effect("shape") == pytest.approx(0.8)


# ── Hold ──────────────────────────────────────────────────────────────────────
class TestHold:
    @staticmethod
    def _engine():
        from aetheric.synth import SynthEngine
        return SynthEngine()

    def test_hold_refuses_when_silent(self):
        eng = self._engine()
        assert eng.hold() is False
        assert eng.is_held is False

    def test_hold_freezes_gesture_input(self):
        eng = self._engine()
        eng.set_params(True, [220.0], "SAW", amplitude=0.3)
        assert eng.hold() is True
        eng.set_params(True, [880.0], "SIN", amplitude=0.1)
        assert eng._freqs == [220.0]
        assert eng._waveform == "SAW"

    def test_effects_still_move_while_held(self):
        eng = self._engine()
        eng.set_params(True, [220.0], "SAW", amplitude=0.3)
        eng.hold()
        eng.set_effect("reverb", 0.75)
        assert eng.get_effect("reverb") == pytest.approx(0.75)

    def test_release_silences(self):
        eng = self._engine()
        eng.set_params(True, [220.0], "SAW", amplitude=0.3)
        eng.hold()
        eng.release_hold()
        assert eng.is_held is False
        assert eng._target_amp == 0.0

    def test_waveform_can_change_while_held(self):
        """Auditioning a shape against a held chord is the point of saying it
        out loud, so set_waveform is not gated on the hold the way set_params is."""
        eng = self._engine()
        eng.set_params(True, [220.0], "SAW", amplitude=0.3)
        eng.hold()
        eng.set_waveform("SQUARE")
        assert eng._waveform == "SQUARE"
        assert eng._freqs == [220.0]      # pitch untouched

    def test_waveform_change_keeps_phase(self):
        """Switching shape mid-note must not restart the oscillator."""
        eng = self._engine()
        eng.set_params(True, [220.0], "SAW", amplitude=0.3)
        eng._render_partials([220.0], eng.get_effect("shape"), 512)
        phase = eng._phases[0]
        eng.set_waveform("SQUARE")
        assert eng._phases[0] == phase


# ── CHORD_RATIOS ──────────────────────────────────────────────────────────────
class TestChordRatios:
    def test_all_voices_present(self):
        for n in range(1, 9):
            assert n in CHORD_RATIOS

    def test_first_ratio_is_root(self):
        for n, ratios in CHORD_RATIOS.items():
            assert ratios[0] == pytest.approx(1.0)

    def test_ratio_count_matches_voice_count(self):
        for n, ratios in CHORD_RATIOS.items():
            assert len(ratios) == n

    def test_ratios_positive_and_ascending(self):
        for n, ratios in CHORD_RATIOS.items():
            assert all(r > 0 for r in ratios)
            assert list(ratios) == sorted(ratios)
