"""Unit tests for audio_map.py — polygon-to-sound mapping."""
import pytest

from aetheric.audio_map import compute_poly_sound

W, H = 640, 480


class TestComputePolySoundEmpty:
    def test_empty_returns_default(self):
        freqs, effects, pos = compute_poly_sound([], [8], [8], W, H)
        assert freqs == [440.0]
        assert effects == {}
        assert pos is None


class TestComputePolySoundBasic:
    def _square(self, cx, cy, size=50):
        half = size // 2
        return [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]

    def test_returns_frequencies(self):
        pts              = self._square(320, 240)
        freqs, effects, pos = compute_poly_sound(pts, [8], [8], W, H)
        assert isinstance(freqs, list)
        assert len(freqs) >= 1
        assert all(f > 0 for f in freqs)

    def test_note_pos_is_centroid(self):
        pts = self._square(320, 240)
        _, _, pos = compute_poly_sound(pts, [8], [8], W, H)
        assert pos == (320, 240)

    def test_filter_effect_present(self):
        pts = self._square(320, 240)
        _, effects, _ = compute_poly_sound(pts, [8], [8], W, H)
        assert "filter" in effects
        assert 0.0 <= effects["filter"] <= 1.0

    def test_filter_top_is_open(self):
        pts = self._square(320, 10)   # near top of frame
        _, effects, _ = compute_poly_sound(pts, [8], [8], W, H)
        assert effects["filter"] > 0.9

    def test_filter_bottom_is_dark(self):
        pts = self._square(320, 470)  # near bottom of frame
        _, effects, _ = compute_poly_sound(pts, [8], [8], W, H)
        assert effects["filter"] < 0.1

    def test_left_pitch_lower_than_right(self):
        pts_left  = self._square(50, 240)
        pts_right = self._square(590, 240)
        freqs_l, _, _ = compute_poly_sound(pts_left,  [8], [8], W, H)
        freqs_r, _, _ = compute_poly_sound(pts_right, [8], [8], W, H)
        assert freqs_l[0] < freqs_r[0]


class TestEffectUnlocks:
    def _polygon(self, n_verts, cx=320, cy=240):
        import math
        pts = []
        for i in range(n_verts):
            angle = 2 * math.pi * i / n_verts
            pts.append((int(cx + 80 * math.cos(angle)),
                        int(cy + 80 * math.sin(angle))))
        return pts

    def test_reverb_not_present_below_6_verts(self):
        pts = self._polygon(4)
        _, effects, _ = compute_poly_sound(pts, [8, 12], [8, 12], W, H)
        assert "reverb" not in effects

    def test_reverb_present_at_6_verts(self):
        pts = self._polygon(6)
        _, effects, _ = compute_poly_sound(pts, [8, 12, 16], [8, 12, 16], W, H)
        assert "reverb" in effects

    def test_tremolo_not_present_below_8_verts(self):
        pts = self._polygon(6)
        _, effects, _ = compute_poly_sound(pts, [8, 12, 16], [8, 12, 16], W, H)
        assert "tremolo" not in effects

    def test_tremolo_present_at_8_verts(self):
        pts = self._polygon(8)
        active = [8, 12, 16, 20]
        _, effects, _ = compute_poly_sound(pts, active, active, W, H)
        assert "tremolo" in effects

    def test_all_effects_in_range(self):
        pts = self._polygon(8)
        active = [8, 12, 16, 20]
        _, effects, _ = compute_poly_sound(pts, active, active, W, H)
        for key in ("filter", "reverb", "tremolo"):
            assert key in effects
            assert 0.0 <= effects[key] <= 1.0, f"{key}={effects[key]} out of range"
