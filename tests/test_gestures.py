"""Unit tests for gestures.py — geometry helpers and gesture detectors."""
import sys
import types

import numpy as np
import pytest

# ── Minimal MediaPipe stub (no camera / GPU required) ────────────────────────
def _make_lm(x: float, y: float):
    lm = types.SimpleNamespace(x=x, y=y)
    return lm


def _make_hand(coords: dict) -> types.SimpleNamespace:
    """coords: {landmark_id: (x_norm, y_norm)}"""
    max_id   = max(coords) + 1
    landmark = [_make_lm(0.5, 0.5)] * max_id
    for lid, (x, y) in coords.items():
        landmark[lid] = _make_lm(x, y)
    hand = types.SimpleNamespace(landmark=landmark)
    return hand


# ── Coordinate helpers ────────────────────────────────────────────────────────
import gestures as G

W, H = 640, 480


class TestLmPx:
    def test_center(self):
        hand = _make_hand({0: (0.5, 0.5)})
        assert G.lm_px(hand, 0, W, H) == (320, 240)

    def test_origin(self):
        hand = _make_hand({0: (0.0, 0.0)})
        assert G.lm_px(hand, 0, W, H) == (0, 0)

    def test_bottom_right(self):
        hand = _make_hand({0: (1.0, 1.0)})
        assert G.lm_px(hand, 0, W, H) == (640, 480)


class TestDist:
    def test_zero(self):
        assert G.dist((0, 0), (0, 0)) == pytest.approx(0.0)

    def test_horizontal(self):
        assert G.dist((0, 0), (3, 0)) == pytest.approx(3.0)

    def test_diagonal(self):
        assert G.dist((0, 0), (3, 4)) == pytest.approx(5.0)


class TestMid:
    def test_basic(self):
        assert G.mid((0, 0), (10, 10)) == (5, 5)

    def test_odd(self):
        assert G.mid((0, 0), (3, 3)) == (1, 1)


class TestPolygonArea:
    def test_unit_square(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert G.polygon_area(pts) == pytest.approx(1.0)

    def test_right_triangle(self):
        pts = np.array([[0, 0], [4, 0], [0, 3]])
        assert G.polygon_area(pts) == pytest.approx(6.0)


# ── fingers_up ────────────────────────────────────────────────────────────────
class TestFingersUp:
    def _hand_all_up(self):
        # All fingertips higher (smaller y) than their knuckles
        coords = {
            G.INDEX:  (0.5, 0.1), 6:  (0.5, 0.5),
            G.MIDDLE: (0.5, 0.1), 10: (0.5, 0.5),
            G.RING:   (0.5, 0.1), 14: (0.5, 0.5),
            G.PINKY:  (0.5, 0.1), 18: (0.5, 0.5),
        }
        return _make_hand(coords)

    def _hand_all_down(self):
        coords = {
            G.INDEX:  (0.5, 0.9), 6:  (0.5, 0.5),
            G.MIDDLE: (0.5, 0.9), 10: (0.5, 0.5),
            G.RING:   (0.5, 0.9), 14: (0.5, 0.5),
            G.PINKY:  (0.5, 0.9), 18: (0.5, 0.5),
        }
        return _make_hand(coords)

    def test_all_up(self):
        assert G.fingers_up(self._hand_all_up()) == 4

    def test_all_down(self):
        assert G.fingers_up(self._hand_all_down()) == 0

    def test_two_up(self):
        coords = {
            G.INDEX:  (0.5, 0.1), 6:  (0.5, 0.5),   # up
            G.MIDDLE: (0.5, 0.1), 10: (0.5, 0.5),   # up
            G.RING:   (0.5, 0.9), 14: (0.5, 0.5),   # down
            G.PINKY:  (0.5, 0.9), 18: (0.5, 0.5),   # down
        }
        assert G.fingers_up(_make_hand(coords)) == 2


# ── is_open_palm ──────────────────────────────────────────────────────────────
class TestIsOpenPalm:
    def test_wide_spread(self):
        hand = _make_hand({
            G.THUMB: (0.0, 0.5),
            G.PINKY: (1.0, 0.5),
        })
        assert G.is_open_palm(hand, W, H) is True

    def test_closed(self):
        hand = _make_hand({
            G.THUMB: (0.5, 0.5),
            G.PINKY: (0.5, 0.5),
        })
        assert G.is_open_palm(hand, W, H) is False


# ── is_cross_pose ─────────────────────────────────────────────────────────────
# ── assign_hands ──────────────────────────────────────────────────────────────
class TestAssignHands:
    def _make_results(self, hands_data):
        """hands_data: list of (landmark_coords_dict, handedness_label_or_None)"""
        landmarks  = []
        handedness = []
        for coords, label in hands_data:
            landmarks.append(_make_hand(coords))
            if label is not None:
                cls   = types.SimpleNamespace(label=label)
                hness = types.SimpleNamespace(classification=[cls])
                handedness.append(hness)

        results = types.SimpleNamespace(
            multi_hand_landmarks=landmarks if landmarks else None,
            multi_handedness=handedness if handedness else None,
        )
        return results

    def test_no_hands(self):
        results = types.SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
        lh, rh  = G.assign_hands(results)
        assert lh is None and rh is None

    def test_mediapipe_handedness(self):
        coords = {G.THUMB: (0.5, 0.5), 17: (0.5, 0.5)}
        results = self._make_results([(coords, "Left"), (coords, "Right")])
        lh, rh  = G.assign_hands(results)
        assert lh is not None and rh is not None

    def test_position_fallback_right_thumb_left(self):
        # thumb.x < landmark[17].x → rh
        coords = {G.THUMB: (0.3, 0.5), 17: (0.7, 0.5)}
        results = types.SimpleNamespace(
            multi_hand_landmarks=[_make_hand(coords)],
            multi_handedness=None,
        )
        lh, rh = G.assign_hands(results)
        assert rh is not None and lh is None


# The finger-count waveform selector is gone: waveform is chosen by voice
# ("cuadrada", "square") or by the z/x/c/v keys, and shaped continuously by the
# shape knob. Counting fingers in IDLE competed with those and fired by
# accident whenever a hand happened to be visible.
