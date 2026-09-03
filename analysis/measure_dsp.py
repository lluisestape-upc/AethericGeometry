"""Reproduce every quantitative claim made about the Aetheric Geometry engine.

Run from the project root::

    python analysis/measure_dsp.py

Writes figures to ``analysis/figures/`` and prints the numbers that appear in
the write-up. Nothing here touches the sound card: each block is driven with
synthetic signals so the results are deterministic and reproducible on any
machine.

Measurements
------------
1. Aliasing of naive vs PolyBLEP oscillators, swept across the instrument's
   pitch range.
2. The block-boundary phase bug: pitch error and the resulting spur.
3. Reverb decay: predicted vs measured :math:`T_{60}`, shared feedback gain vs
   per-comb solved gain.
4. Filter cutoff accuracy: prewarped TPT vs the naive exponential one-pole.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetheric.dsp import (  # noqa: E402
    DampedComb, SchroederAllpass, blep_saw, blep_square, rt60_to_feedback,
)

FS = 44100.0
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Muted palette that survives greyscale printing.
C_BAD, C_GOOD, C_REF = "#c1442e", "#2e6fc1", "#6b6b6b"


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def bin_locked(f0: float, n: int, fs: float = FS):
    """Return ``(phase, inc, k, exact_f0)`` with ``f0`` on an exact FFT bin.

    Placing the fundamental on a bin means every harmonic lands on a bin too,
    so the spectrum needs no window and there is no leakage to be mistaken for
    aliasing. This is the single most important detail in the measurement.
    """
    k = max(1, round(f0 * n / fs))
    exact = k * fs / n
    inc = exact / fs
    return (np.arange(n) * inc) % 1.0, inc, k, exact


def alias_to_signal_db(x: np.ndarray, k: int) -> float:
    """Off-grid energy over on-grid energy, in dB."""
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec[0] = 0.0
    mask = np.zeros_like(spec, dtype=bool)
    m = 1
    while m * k < len(spec):
        mask[m * k] = True
        m += 1
    return 10.0 * np.log10(max(spec[~mask].sum(), 1e-300)
                           / max(spec[mask].sum(), 1e-300))


def schroeder_t60(h: np.ndarray, fs: float = FS) -> float:
    """T60 from the backward-integrated energy decay curve, fitted over T30.

    Fitting between :math:`-5` and :math:`-35` dB avoids both the direct sound
    at the start and the numerical noise floor at the end, then extrapolates to
    :math:`-60` dB — the standard estimator, and honest about the fact that a
    finite impulse response never actually reaches :math:`-60` dB.
    """
    energy = np.cumsum(h[::-1] ** 2)[::-1]
    edc = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-300))
    try:
        i0 = int(np.argmax(edc <= -5.0))
        i1 = int(np.argmax(edc <= -35.0))
    except ValueError:
        return float("nan")
    if i1 <= i0:
        return float("nan")
    t = np.arange(i0, i1) / fs
    slope, _ = np.polyfit(t, edc[i0:i1], 1)
    return float(-60.0 / slope)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Aliasing
# ══════════════════════════════════════════════════════════════════════════════
def measure_aliasing() -> None:
    n = 1 << 16
    print("\n=== 1. Alias-to-signal ratio (dB, lower is better) ===")
    print(f"{'f0 (Hz)':>9} | {'naive saw':>10} {'BLEP saw':>10} {'gain':>7} "
          f"| {'naive sq':>10} {'BLEP sq':>10} {'gain':>7}")
    print("-" * 78)

    f0s = np.geomspace(55.0, 6000.0, 40)
    rows = []
    for f0 in f0s:
        ph, inc, k, exact = bin_locked(f0, n)
        naive_saw = alias_to_signal_db(2.0 * ph - 1.0, k)
        good_saw = alias_to_signal_db(blep_saw(ph, inc), k)
        naive_sq = alias_to_signal_db(np.where(ph < 0.5, 1.0, -1.0), k)
        good_sq = alias_to_signal_db(blep_square(ph, inc), k)
        rows.append((exact, naive_saw, good_saw, naive_sq, good_sq))

    for r in rows[::5]:
        print(f"{r[0]:9.1f} | {r[1]:10.1f} {r[2]:10.1f} {r[1] - r[2]:7.1f} "
              f"| {r[3]:10.1f} {r[4]:10.1f} {r[3] - r[4]:7.1f}")

    arr = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for i, (name, ni, gi) in enumerate((("Sawtooth", 1, 2), ("Square", 3, 4))):
        ax[i].semilogx(arr[:, 0], arr[:, ni], color=C_BAD, lw=1.8, label="naive")
        ax[i].semilogx(arr[:, 0], arr[:, gi], color=C_GOOD, lw=1.8, label="PolyBLEP")
        ax[i].set_title(name)
        ax[i].set_xlabel("fundamental (Hz)")
        ax[i].grid(alpha=0.3, which="both")
        ax[i].legend(frameon=False)
    ax[0].set_ylabel("alias-to-signal ratio (dB)")
    fig.suptitle("Aliasing across the instrument's pitch range "
                 f"($f_s$ = {FS/1000:.1f} kHz)")
    fig.tight_layout()
    fig.savefig(OUT / "aliasing_sweep.png", dpi=150)
    plt.close(fig)

    # Spectrum at a high note, where the difference is most visible.
    ph, inc, k, exact = bin_locked(2000.0, n)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    for sig, col, lbl in ((2.0 * ph - 1.0, C_BAD, "naive"),
                          (blep_saw(ph, inc), C_GOOD, "PolyBLEP")):
        mag = 20.0 * np.log10(np.abs(np.fft.rfft(sig)) / (n / 2) + 1e-12)
        ax.plot(freqs, mag, color=col, lw=0.9, label=lbl, alpha=0.9)
    for m in range(1, int(FS / 2 / exact) + 1):
        ax.axvline(m * exact, color=C_REF, lw=0.5, ls=":", alpha=0.5)
    ax.set_xlim(0, FS / 2)
    ax.set_ylim(-120, 5)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title(f"Sawtooth spectrum at {exact:.1f} Hz — dotted lines are true "
                 "harmonics, everything else is aliasing")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "aliasing_spectrum.png", dpi=150)
    plt.close(fig)
    print(f"\n  figures -> {OUT / 'aliasing_sweep.png'}, "
          f"{OUT / 'aliasing_spectrum.png'}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. The block-boundary phase bug
# ══════════════════════════════════════════════════════════════════════════════
def _render_phase_bug(f0: float, blocks: int, block: int, buggy: bool):
    """Reproduce both phase-accumulation strategies."""
    inc = f0 / FS
    phase = 0.0
    out = []
    for _ in range(blocks):
        ph = (phase + np.arange(block) * inc) % 1.0
        out.append(np.sin(2.0 * np.pi * ph))
        # The bug: writing back the last sample's phase instead of the phase
        # the next block should begin on.
        phase = (ph[-1] if buggy else phase + block * inc) % 1.0
    return np.concatenate(out)


def measure_phase_bug() -> None:
    block, blocks, f0 = 512, 256, 440.0
    n = block * blocks
    print("\n=== 2. Block-boundary phase accumulation ===")

    cents = 1200.0 * math.log2((block - 1) / block)
    print(f"  predicted detuning : {cents:+.3f} cents "
          f"({f0:.1f} Hz -> {f0 * (block - 1) / block:.3f} Hz)")
    print(f"  glitch repetition  : {FS / block:.2f} Hz (one per block)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    win = np.hanning(n)
    for buggy, col, lbl in ((True, C_BAD, "phase written back one sample short"),
                            (False, C_GOOD, "advance by the full block")):
        y = _render_phase_bug(f0, blocks, block, buggy)
        mag = 20.0 * np.log10(np.abs(np.fft.rfft(y * win)) / (n / 4) + 1e-12)
        peak = freqs[int(np.argmax(mag))]
        print(f"  measured f0 ({'buggy' if buggy else 'fixed'}): {peak:.3f} Hz")
        ax[0].plot(freqs, mag, color=col, lw=0.9, label=lbl)
        ax[1].plot(freqs, mag, color=col, lw=1.1)
    ax[0].set_xlim(0, 4000)
    ax[0].set_ylim(-140, 5)
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("magnitude (dB)")
    ax[0].set_title("Full spectrum of a 440 Hz sine")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].set_xlim(300, 600)
    ax[1].set_ylim(-140, 5)
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_title("Detail: sidebands at $\\pm f_s/N$ = $\\pm$86.1 Hz")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase_bug.png", dpi=150)
    plt.close(fig)
    print(f"  figure -> {OUT / 'phase_bug.png'}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. Reverb decay
# ══════════════════════════════════════════════════════════════════════════════
COMB_DELAYS = (1557, 1617, 1491, 1422)
ALLPASS_DELAYS = (556, 225)


def _reverb_impulse(gains, damp: float, n: int = 1 << 17) -> np.ndarray:
    combs = []
    for d, g in zip(COMB_DELAYS, gains):
        c = DampedComb(d, FS)
        c._g = g
        c.set_damping(damp)
        combs.append(c)
    aps = [SchroederAllpass(d, 0.5) for d in ALLPASS_DELAYS]

    imp = np.zeros(n)
    imp[0] = 1.0
    out = np.zeros(n)
    for i in range(0, n, 512):
        blk = imp[i:i + 512]
        acc = np.zeros_like(blk)
        for c in combs:
            acc += c.process(blk)
        acc /= len(combs)
        for ap in aps:
            acc = ap.process(acc)
        out[i:i + 512] = acc
    return out


def measure_reverb() -> None:
    print("\n=== 3. Reverb decay ===")
    print("  (a) shared feedback gain g = 0.82 — each comb decays differently")
    per_comb = []
    for d in COMB_DELAYS:
        t60 = -3.0 * d / (FS * math.log10(0.82))
        per_comb.append(t60)
        print(f"      D = {d:5d}  ->  T60 = {t60:.3f} s")
    spread = max(per_comb) / min(per_comb) - 1.0
    print(f"      spread across the bank: {spread * 100:.1f} %")

    target = 1.50
    solved = [rt60_to_feedback(target, d, FS) for d in COMB_DELAYS]
    print(f"\n  (b) solved per comb for T60 = {target:.2f} s")
    for d, g in zip(COMB_DELAYS, solved):
        print(f"      D = {d:5d}  ->  g = {g:.5f}  "
              f"(check: {-3.0 * d / (FS * math.log10(g)):.3f} s)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for i, (gains, title) in enumerate((
            ([0.82] * 4, "Shared $g$ = 0.82"),
            (solved, f"Solved per comb, $T_{{60}}$ = {target:.2f} s"))):
        for damp, col, lbl in ((0.0, C_BAD, "no damping"),
                               (0.3, C_GOOD, "damped (0.3)")):
            h = _reverb_impulse(gains, damp)
            energy = np.cumsum(h[::-1] ** 2)[::-1]
            edc = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-300))
            t = np.arange(len(h)) / FS
            ax[i].plot(t, edc, color=col, lw=1.4, label=lbl)
            meas = schroeder_t60(h)
            print(f"      measured T60 [{title}, damp={damp}]: {meas:.3f} s")
        ax[i].axhline(-60, color=C_REF, lw=0.8, ls="--")
        ax[i].set_xlim(0, 4)
        ax[i].set_ylim(-90, 2)
        ax[i].set_xlabel("time (s)")
        ax[i].set_title(title)
        ax[i].grid(alpha=0.3)
        ax[i].legend(frameon=False)
    ax[0].set_ylabel("energy decay curve (dB)")
    fig.suptitle("Schroeder backward-integrated decay of the reverb bank")
    fig.tight_layout()
    fig.savefig(OUT / "reverb_decay.png", dpi=150)
    plt.close(fig)
    print(f"  figure -> {OUT / 'reverb_decay.png'}")


# ══════════════════════════════════════════════════════════════════════════════
#  4. Filter cutoff accuracy
# ══════════════════════════════════════════════════════════════════════════════
def measure_filter() -> None:
    print("\n=== 4. One-pole cutoff accuracy (single stage, target -3.01 dB) ===")
    print(f"{'fc (Hz)':>9} | {'TPT (dB)':>10} | {'naive exp (dB)':>15}")
    print("-" * 42)

    n = 1 << 15
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    targets = [100.0, 500.0, 2000.0, 5000.0, 10000.0, 15000.0, 20000.0]
    tpt_err, naive_err = [], []

    for fc in targets:
        g = math.tan(math.pi * fc / FS)
        G = g / (1.0 + g)
        b, a = np.array([G, G]), np.array([1.0, -(1.0 - 2.0 * G)])
        w = 2.0 * np.pi * freqs / FS
        z = np.exp(-1j * w)
        H_tpt = (b[0] + b[1] * z) / (a[0] + a[1] * z)

        c = math.exp(-2.0 * math.pi * fc / FS)
        H_naive = (1.0 - c) / (1.0 - c * z)

        k = int(np.argmin(np.abs(freqs - fc)))
        d_tpt = 20.0 * math.log10(abs(H_tpt[k]))
        d_naive = 20.0 * math.log10(abs(H_naive[k]))
        tpt_err.append(d_tpt + 3.0103)
        naive_err.append(d_naive + 3.0103)
        print(f"{fc:9.0f} | {d_tpt:10.3f} | {d_naive:15.3f}")

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.semilogx(targets, naive_err, "o-", color=C_BAD,
                label=r"naive  $c=e^{-2\pi f_c/f_s}$")
    ax.semilogx(targets, tpt_err, "s-", color=C_GOOD,
                label=r"TPT  $g=\tan(\pi f_c/f_s)$")
    ax.axhline(0.0, color=C_REF, lw=0.8, ls="--")
    ax.set_xlabel("requested cutoff (Hz)")
    ax.set_ylabel("error at $f_c$ (dB from $-3.01$)")
    ax.set_title("Where the cutoff actually lands")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "filter_cutoff.png", dpi=150)
    plt.close(fig)
    print(f"  figure -> {OUT / 'filter_cutoff.png'}")


if __name__ == "__main__":
    measure_aliasing()
    measure_phase_bug()
    measure_reverb()
    measure_filter()
    print(f"\nAll figures written to {OUT}\n")
