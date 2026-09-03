"""Unit tests for knob.py — pure geometry, no camera required."""
import math

import pytest

from aetheric.knob import (
    INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP, PINKY_TIP,
    RING_TIP, THUMB_TIP, WRIST,
    KnobPoseDetector, RotaryKnob, hand_rotation, measure_hand, palm_width,
    wrap_to_pi,
)


# ── Angle helpers ─────────────────────────────────────────────────────────────
class TestWrapToPi:
    def test_identity_inside_range(self):
        for a in (-3.0, -1.0, 0.0, 1.0, 3.0):
            assert wrap_to_pi(a) == pytest.approx(a)

    def test_folds_above_pi(self):
        assert wrap_to_pi(math.pi + 0.5) == pytest.approx(-math.pi + 0.5)

    def test_folds_below_minus_pi(self):
        assert wrap_to_pi(-math.pi - 0.5) == pytest.approx(math.pi - 0.5)

    def test_full_turn_is_zero(self):
        assert wrap_to_pi(2.0 * math.pi) == pytest.approx(0.0, abs=1e-9)


class TestHandRotation:
    def test_pointing_right_is_zero(self):
        assert hand_rotation((0.0, 0.0), (10.0, 0.0)) == pytest.approx(0.0)

    def test_pointing_down_is_half_pi(self):
        # Image y grows downwards, so "down the screen" is +pi/2.
        assert hand_rotation((0.0, 0.0), (0.0, 10.0)) == pytest.approx(math.pi / 2)

    def test_palm_width_is_euclidean(self):
        assert palm_width((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


# ── Rotary knob ───────────────────────────────────────────────────────────────
class TestRotaryKnob:
    def test_first_grip_frame_does_not_move_the_value(self):
        k = RotaryKnob(initial=0.5)
        assert k.update(1.234, True) == pytest.approx(0.5)

    def test_quarter_turn_covers_half_the_range(self):
        k = RotaryKnob(turns_for_full_range=0.5, initial=0.0)
        k.update(0.0, True)
        assert k.update(math.pi / 2, True) == pytest.approx(0.5)

    def test_clamps_at_both_ends(self):
        # Stepped in quarter turns: a single delta of exactly pi is ambiguous
        # in direction (see RotaryKnob.update), so a real gesture is sampled
        # far more finely than that.
        k = RotaryKnob(turns_for_full_range=0.5, initial=0.5)
        angle = 0.0
        k.update(angle, True)
        for _ in range(6):
            angle = wrap_to_pi(angle + math.pi / 2)
            k.update(angle, True)
        assert k.value == pytest.approx(1.0)
        for _ in range(12):
            angle = wrap_to_pi(angle - math.pi / 2)
            k.update(angle, True)
        assert k.value == pytest.approx(0.0)

    def test_half_turn_in_one_frame_is_ambiguous(self):
        """Nyquist for angle: +pi and -pi are the same observation."""
        k = RotaryKnob(turns_for_full_range=0.5, initial=0.5)
        k.update(0.0, True)
        k.update(math.pi, True)
        assert k.value in (pytest.approx(0.0), pytest.approx(1.0))

    def test_release_holds_value_and_regrips_without_jumping(self):
        k = RotaryKnob(turns_for_full_range=0.5, initial=0.0)
        k.update(0.0, True)
        k.update(math.pi / 4, True)
        held = k.value
        k.update(math.pi / 4, False)          # let go
        assert k.value == pytest.approx(held)
        k.update(-2.5, True)                  # re-grip somewhere else entirely
        assert k.value == pytest.approx(held)  # ...must not jump

    def test_dead_zone_rejects_tremor(self):
        k = RotaryKnob(dead_zone_radians=0.05, initial=0.3)
        k.update(0.0, True)
        k.update(0.01, True)
        assert k.value == pytest.approx(0.3)

    def test_wrap_does_not_cause_a_full_turn(self):
        k = RotaryKnob(turns_for_full_range=0.5, initial=0.5)
        k.update(math.pi - 0.05, True)
        k.update(-math.pi + 0.05, True)       # 0.1 rad forwards across the seam
        assert k.value == pytest.approx(0.5 + 0.1 / math.pi, abs=1e-6)

    def test_accumulates_past_one_full_turn(self):
        k = RotaryKnob(turns_for_full_range=4.0, initial=0.0)
        angle = 0.0
        k.update(angle, True)
        for _ in range(16):                   # 16 x quarter turn = 4 turns
            angle = wrap_to_pi(angle + math.pi / 2)
            k.update(angle, True)
        assert k.value == pytest.approx(1.0)

    def test_set_and_reset(self):
        k = RotaryKnob(initial=0.2)
        k.set(0.9)
        assert k.value == pytest.approx(0.9)
        k.reset()
        assert k.value == pytest.approx(0.2)
        assert k.engaged is False


# ── Pose detection ────────────────────────────────────────────────────────────
class TestKnobPoseDetector:
    @staticmethod
    def _det():
        return KnobPoseDetector(close_below=0.6, open_above=0.9,
                                require_extended=2, extended_above=1.6,
                                folded_below=1.3)

    def test_rejects_inconsistent_thresholds(self):
        with pytest.raises(ValueError):
            KnobPoseDetector(close_below=0.9, open_above=0.6)
        with pytest.raises(ValueError):
            KnobPoseDetector(extended_above=1.0, folded_below=1.5)

    def test_grips_on_pinch_with_fingers_out(self):
        det = self._det()
        assert det.push(0.4, [1.8, 1.8, 1.7]) is True

    def test_fist_does_not_grip(self):
        """A fist also brings thumb and index together — the extra condition
        is what keeps a relaxed hand from dragging the value."""
        det = self._det()
        assert det.push(0.3, [0.8, 0.7, 0.6]) is False

    def test_pinch_hysteresis(self):
        det = self._det()
        det.push(0.4, [1.8, 1.8, 1.8])
        det.push(0.75, [1.8, 1.8, 1.8])       # between the two thresholds
        assert det.is_pinching is True        # ...still gripped
        det.push(0.95, [1.8, 1.8, 1.8])
        assert det.is_pinching is False

    def test_extension_hysteresis(self):
        det = self._det()
        det.push(0.4, [1.8, 1.8, 1.8])
        assert det.push(0.4, [1.4, 1.4, 1.0]) is True   # sags but stays out
        assert det.push(0.4, [1.2, 1.2, 1.0]) is False  # ...now folded

    def test_reset(self):
        det = self._det()
        det.push(0.4, [1.8, 1.8, 1.8])
        det.reset()
        assert det.is_holding is False


# ── Landmark reduction ────────────────────────────────────────────────────────
class TestMeasureHand:
    @staticmethod
    def _hand(scale=1.0, dx=0.0, dy=0.0):
        """Upright hand: wrist at origin, knuckles 40 px wide, fingers up."""
        base = {
            WRIST: (0.0, 0.0),
            INDEX_MCP: (-20.0, -60.0),
            PINKY_MCP: (20.0, -60.0),
            MIDDLE_MCP: (0.0, -60.0),
            THUMB_TIP: (-35.0, -40.0),
            INDEX_TIP: (-20.0, -120.0),
            MIDDLE_TIP: (0.0, -125.0),
            RING_TIP: (15.0, -118.0),
            PINKY_TIP: (30.0, -100.0),
        }
        return {k: (x * scale + dx, y * scale + dy) for k, (x, y) in base.items()}

    def test_scale_invariance(self):
        """Moving towards the camera scales every pixel distance; normalising
        by palm width has to cancel that out exactly."""
        near = measure_hand(self._hand(scale=2.0))
        far = measure_hand(self._hand(scale=0.5))
        assert near["pinch"] == pytest.approx(far["pinch"])
        for a, b in zip(near["extensions"], far["extensions"]):
            assert a == pytest.approx(b)

    def test_translation_invariance(self):
        a = measure_hand(self._hand())
        b = measure_hand(self._hand(dx=300.0, dy=-120.0))
        assert a["pinch"] == pytest.approx(b["pinch"])
        assert a["angle"] == pytest.approx(b["angle"])

    def test_rotation_leaves_extensions_alone(self):
        """The reason extension is radial, not vertical: this gesture rotates
        the hand, so a tip-above-knuckle test would flip mid-turn."""
        upright = measure_hand(self._hand())
        theta = math.pi / 2
        rotated_pts = {}
        for lid, (x, y) in self._hand().items():
            rotated_pts[lid] = (x * math.cos(theta) - y * math.sin(theta),
                                x * math.sin(theta) + y * math.cos(theta))
        rotated = measure_hand(rotated_pts)
        for a, b in zip(upright["extensions"], rotated["extensions"]):
            assert a == pytest.approx(b)
        assert wrap_to_pi(rotated["angle"] - upright["angle"]) == pytest.approx(theta)

    def test_upright_hand_points_up(self):
        m = measure_hand(self._hand())
        assert m["angle"] == pytest.approx(-math.pi / 2)

    def test_palm_width_reported(self):
        assert measure_hand(self._hand())["palm_width"] == pytest.approx(40.0)
