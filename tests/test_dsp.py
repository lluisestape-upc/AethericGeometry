"""Unit tests for dsp.py — the claims each block makes, checked numerically."""
import numpy as np
import pytest

from dsp import (
    BlepTriangle, DCBlocker, DampedComb, OnePoleTPT, SchroederAllpass,
    blep_saw, blep_square, rt60_to_feedback, soft_clip, tau_to_alpha,
)

FS = 44100.0
N = 1 << 16


def _phase(f0: float, n: int, fs: float = FS) -> tuple:
    """Phase ramp plus increment, for an ``f0`` that lands on an exact FFT bin."""
    k = round(f0 * n / fs)
    exact = k * fs / n
    inc = exact / fs
    return (np.arange(n) * inc) % 1.0, inc, k


def alias_to_signal_db(x: np.ndarray, k: int) -> float:
    """Energy off the harmonic grid, relative to energy on it, in dB.

    With ``f0`` placed on bin ``k`` every true harmonic lands exactly on bin
    ``k*m``, so no window is needed and there is no spectral leakage to confuse
    the measurement. Anything that is not on that grid is aliasing.
    """
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec[0] = 0.0  # ignore DC
    harmonic = np.zeros_like(spec, dtype=bool)
    m = 1
    while m * k < len(spec):
        harmonic[m * k] = True
        m += 1
    on = spec[harmonic].sum()
    off = spec[~harmonic].sum()
    return 10.0 * np.log10(max(off, 1e-30) / max(on, 1e-30))


# ── Band-limited oscillators ──────────────────────────────────────────────────
class TestBandLimiting:
    def test_polyblep_saw_beats_naive_saw(self):
        ph, inc, k = _phase(2000.0, N)
        naive = 2.0 * ph - 1.0
        blep = blep_saw(ph, inc)
        naive_db = alias_to_signal_db(naive, k)
        blep_db = alias_to_signal_db(blep, k)
        # The correction must buy a large, unambiguous margin.
        assert blep_db < naive_db - 15.0, (naive_db, blep_db)

    def test_polyblep_square_beats_naive_square(self):
        ph, inc, k = _phase(2000.0, N)
        naive = np.where(ph < 0.5, 1.0, -1.0)
        blep = blep_square(ph, inc)
        assert alias_to_signal_db(blep, k) < alias_to_signal_db(naive, k) - 12.0

    def test_advantage_grows_with_pitch(self):
        """Aliasing is a high-note problem, so the gain must grow with f0."""
        gains = []
        for f0 in (200.0, 2000.0):
            ph, inc, k = _phase(f0, N)
            naive = 2.0 * ph - 1.0
            gains.append(alias_to_signal_db(naive, k)
                         - alias_to_signal_db(blep_saw(ph, inc), k))
        assert gains[1] > gains[0]

    def test_saw_is_bounded(self):
        ph, inc, _ = _phase(1000.0, 4096)
        assert np.max(np.abs(blep_saw(ph, inc))) < 1.2

    def test_triangle_is_bounded_and_centred(self):
        tri = BlepTriangle(FS)
        ph, inc, _ = _phase(440.0, N)
        out = np.concatenate([tri.process(ph[i:i + 512], inc)
                              for i in range(0, N, 512)])
        settled = out[N // 2:]
        assert np.max(np.abs(settled)) < 1.3
        assert abs(float(np.mean(settled))) < 1e-2


# ── Filters ───────────────────────────────────────────────────────────────────
class TestOnePoleTPT:
    @pytest.mark.parametrize("fc", [100.0, 1000.0, 8000.0, 15000.0])
    def test_single_stage_is_minus_3db_at_cutoff(self, fc):
        """The whole reason for prewarping: the cutoff lands where asked even
        when fc is a large fraction of Nyquist."""
        filt = OnePoleTPT(FS, stages=1)
        filt.set_cutoff(fc)
        imp = np.zeros(N)
        imp[0] = 1.0
        y = np.concatenate([filt.process(imp[i:i + 1024])
                            for i in range(0, N, 1024)])
        H = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(N, 1.0 / FS)
        gain_db = 20.0 * np.log10(H[int(np.argmin(np.abs(freqs - fc)))])
        assert gain_db == pytest.approx(-3.01, abs=0.15)

    def test_naive_exp_one_pole_drifts_where_tpt_does_not(self):
        """Contrast case: exp(-2*pi*fc/fs) is only right for fc << fs."""
        fc = 15000.0
        c = np.exp(-2.0 * np.pi * fc / FS)
        imp = np.zeros(N)
        imp[0] = 1.0
        y = np.empty_like(imp)
        acc = 0.0
        for i, s in enumerate(imp):
            acc = (1.0 - c) * s + c * acc
            y[i] = acc
        H = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(N, 1.0 / FS)
        naive_db = 20.0 * np.log10(H[int(np.argmin(np.abs(freqs - fc)))])
        assert abs(naive_db + 3.01) > 0.5

    def test_dc_gain_is_unity(self):
        filt = OnePoleTPT(FS, stages=2)
        filt.set_cutoff(1000.0)
        y = filt.process(np.ones(20000))
        assert y[-1] == pytest.approx(1.0, abs=1e-6)


class TestDCBlocker:
    def test_removes_constant_offset(self):
        dc = DCBlocker(FS)
        y = dc.process(np.ones(40000) * 0.5)
        assert abs(y[-1]) < 1e-3

    def test_passes_audio_band(self):
        dc = DCBlocker(FS)
        t = np.arange(20000) / FS
        x = np.sin(2.0 * np.pi * 1000.0 * t)
        y = dc.process(x)
        assert np.max(np.abs(y[10000:])) == pytest.approx(1.0, abs=1e-2)


# ── Reverb blocks ─────────────────────────────────────────────────────────────
class TestReverbBlocks:
    def test_rt60_solver_round_trips(self):
        for rt60 in (0.4, 1.5, 4.0):
            for delay in (1422, 1557, 1617):
                g = rt60_to_feedback(rt60, delay, FS)
                recovered = -3.0 * delay / (FS * np.log10(g))
                assert recovered == pytest.approx(rt60, rel=1e-9)

    def test_equal_gain_gives_unequal_decay(self):
        """The defect being fixed: a shared g makes T60 proportional to D."""
        t60 = [-3.0 * d / (FS * np.log10(0.82)) for d in (1422, 1557, 1617)]
        assert max(t60) / min(t60) > 1.1

    def test_solved_gains_give_equal_decay(self):
        target = 1.5
        for d in (1422, 1557, 1617):
            g = rt60_to_feedback(target, d, FS)
            assert -3.0 * d / (FS * np.log10(g)) == pytest.approx(target, rel=1e-9)

    def test_comb_delays_the_impulse(self):
        comb = DampedComb(600, FS)
        comb.set_rt60(2.0)
        comb.set_damping(0.0)
        imp = np.zeros(2048)
        imp[0] = 1.0
        y = np.concatenate([comb.process(imp[i:i + 256])
                            for i in range(0, 2048, 256)])
        assert int(np.argmax(np.abs(y))) == 600

    def test_comb_chunking_matches_single_block(self):
        """Splitting a block at the delay-line wrap must change nothing."""
        imp = np.random.default_rng(0).standard_normal(4096) * 0.1
        a = DampedComb(600, FS); a.set_rt60(2.0); a.set_damping(0.3)
        b = DampedComb(600, FS); b.set_rt60(2.0); b.set_damping(0.3)
        whole = a.process(imp.copy())
        pieces = np.concatenate([b.process(imp[i:i + 128].copy())
                                 for i in range(0, 4096, 128)])
        np.testing.assert_allclose(whole, pieces, atol=1e-12)

    def test_allpass_has_flat_magnitude(self):
        ap = SchroederAllpass(225, 0.5)
        imp = np.zeros(N)
        imp[0] = 1.0
        y = np.concatenate([ap.process(imp[i:i + 512]) for i in range(0, N, 512)])
        H = np.abs(np.fft.rfft(y))
        # |H| = 1 at every frequency is what makes it an allpass.
        assert np.max(np.abs(H - 1.0)) < 1e-6

    def test_allpass_preserves_energy(self):
        ap = SchroederAllpass(556, 0.5)
        x = np.random.default_rng(1).standard_normal(N) * 0.1
        # Flush the tail: the filter still holds energy when the input stops,
        # so a window that ends with the input under-counts the output.
        padded = np.concatenate([x, np.zeros(1 << 14)])
        y = np.concatenate([ap.process(padded[i:i + 512])
                            for i in range(0, len(padded), 512)])
        assert float(np.sum(y ** 2)) == pytest.approx(float(np.sum(x ** 2)), rel=1e-3)


# ── Helpers ───────────────────────────────────────────────────────────────────
class TestHelpers:
    def test_tau_to_alpha_hits_the_time_constant(self):
        """After tau seconds a step must have covered 1 - 1/e of the distance."""
        tau = 0.01
        a = tau_to_alpha(tau, FS)
        n = int(tau * FS)
        remaining = (1.0 - a) ** n
        assert remaining == pytest.approx(np.exp(-1.0), rel=1e-3)

    def test_tau_is_sample_rate_independent(self):
        """The same time in seconds at any rate.

        Checked on the exact (non-integer) sample count, because that is the
        actual claim: ``(1-alpha)**(tau*fs) == exp(-1)`` identically. Rounding
        to a whole number of samples then costs at most half a sample, which is
        the small residual the integer check below allows for.
        """
        tau = 0.005
        for fs in (44100, 48000, 96000):
            exact = (1.0 - tau_to_alpha(tau, fs)) ** (tau * fs)
            assert exact == pytest.approx(np.exp(-1.0), rel=1e-12)

            n = int(round(tau * fs))
            assert (1.0 - tau_to_alpha(tau, fs)) ** n == pytest.approx(
                np.exp(-1.0), rel=1e-2)

    def test_soft_clip_is_monotone_and_bounded(self):
        x = np.linspace(-8.0, 8.0, 4096)
        y = soft_clip(x)
        assert np.all(np.diff(y) > 0)
        assert np.max(np.abs(y)) < 1.0
