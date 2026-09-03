"""Rotary hand knob — grip, turn, let go.

Ported from the AirSculpt gesture layer. Pinch thumb to index and rotate the
hand as if turning a knob that faces the camera; release the pinch and the
value stays where it was, so you can bring your arm back to a comfortable angle
and keep turning. Rotation accumulates, which means unlimited travel: you can
turn as many times as you like instead of running out of frame.

Two design decisions worth stating, because both were arrived at by discarding
something that did not survive contact with a webcam:

**Rotation in the camera plane, not palm tilt.** An earlier design used a flat
palm facing the ceiling moved up and down. A palm facing the ceiling is nearly
edge-on to the lens, so its landmarks are badly conditioned and the grip
flickers. Rotation about the camera axis is the opposite: it is the most
visible thing a hand can do, and the wrist-to-middle-knuckle vector that
measures it is long, well separated from the noise floor and visible at any
hand orientation.

**Extension measured radially, not vertically.** The rest of this project
decides whether a finger is extended by comparing the tip's :math:`y` against
its knuckle's. That test silently assumes the hand is upright — and this
gesture rotates the hand, which is precisely when it stops being true. Here
extension is the tip-to-wrist distance divided by palm width: invariant to both
rotation and distance from the camera.
"""
from __future__ import annotations

import math

# MediaPipe hand landmark indices used here.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

_OTHER_TIPS = (MIDDLE_TIP, RING_TIP, PINKY_TIP)


def wrap_to_pi(radians: float) -> float:
    """Fold an angle difference into :math:`[-\\pi, \\pi]`.

    Without this, a hand crossing the :math:`+\\pi / -\\pi` boundary reads as a
    full turn backwards in a single frame.
    """
    return (float(radians) + math.pi) % (2.0 * math.pi) - math.pi


def hand_rotation(wrist: tuple, middle_mcp: tuple) -> float:
    """Hand angle in the camera plane, in radians."""
    return math.atan2(middle_mcp[1] - wrist[1], middle_mcp[0] - wrist[0])


def palm_width(pt_index_mcp: tuple, pt_pinky_mcp: tuple) -> float:
    """Knuckle-to-knuckle width, the scale everything else is measured in."""
    return math.hypot(pt_index_mcp[0] - pt_pinky_mcp[0],
                      pt_index_mcp[1] - pt_pinky_mcp[1])


class RotaryKnob:
    """Accumulating rotary control producing a value in ``[0, 1]``."""

    def __init__(self, turns_for_full_range: float = 0.5, initial: float = 0.0,
                 dead_zone_radians: float = 0.012):
        # Half a turn per full range by default. A wrist rotates comfortably
        # through roughly a quarter turn before the arm has to follow, so
        # anything near a full turn per range means a natural gesture barely
        # moves the value.
        self._per_radian = 1.0 / (float(turns_for_full_range) * 2.0 * math.pi)
        self._dead_zone = float(dead_zone_radians)
        self._initial = _clamp01(initial)
        self._value = self._initial
        self._prev_angle: float | None = None
        self._engaged = False

    @property
    def value(self) -> float:
        return self._value

    @property
    def engaged(self) -> bool:
        return self._engaged

    def set(self, value: float) -> None:
        """Force the value, e.g. when adopting an effect's current setting."""
        self._value = _clamp01(value)

    def release(self) -> None:
        self._engaged = False
        self._prev_angle = None

    def reset(self) -> None:
        self._value = self._initial
        self.release()

    def update(self, angle: float, gripping: bool) -> float:
        """Advance the knob. ``angle`` may be in any range; wrap is handled.

        There is a sampling limit here worth naming: rotation is only observed
        once per video frame, and :func:`wrap_to_pi` maps every delta into
        :math:`[-\\pi, \\pi]`, so a turn of more than half a revolution between
        two frames is indistinguishable from a smaller turn the other way. This
        is Nyquist applied to angle. At 30 fps it allows 15 revolutions per
        second, far beyond what a wrist does, so it never binds in practice —
        but a delta of exactly :math:`\\pi` is genuinely ambiguous and resolves
        arbitrarily.
        """
        if not gripping:
            self.release()
            return self._value

        if not self._engaged or self._prev_angle is None:
            # First frame of a grip only captures the reference angle, so
            # closing the hand never jumps the value.
            self._engaged = True
            self._prev_angle = float(angle)
            return self._value

        delta = wrap_to_pi(float(angle) - self._prev_angle)
        if abs(delta) < self._dead_zone:
            return self._value

        # Screen-clockwise raises the value, as on a real knob. Image y points
        # down, so a positive atan2 delta is already clockwise on screen.
        self._value = _clamp01(self._value + delta * self._per_radian)
        self._prev_angle = float(angle)
        return self._value


class KnobPoseDetector:
    """Is the hand holding the knob? Pinch **and** other fingers out.

    Both halves are required. A closed fist also puts the thumb next to the
    index, so pinch distance alone would keep the knob gripped once the hand
    relaxes — and then simply lowering the arm would keep turning the value
    that was just settled on. Requiring middle/ring/pinky to stay out makes
    letting go unambiguous.

    Both tests have hysteresis so a pose resting on a threshold does not
    chatter between gripped and free.
    """

    def __init__(self, close_below: float = 0.60, open_above: float = 0.90,
                 require_extended: int = 2, extended_above: float = 1.60,
                 folded_below: float = 1.30):
        if open_above <= close_below:
            raise ValueError("open_above must exceed close_below")
        if folded_below >= extended_above:
            raise ValueError("folded_below must be less than extended_above")
        self._close_below = float(close_below)
        self._open_above = float(open_above)
        self._require = int(require_extended)
        self._extended_above = float(extended_above)
        self._folded_below = float(folded_below)
        self._pinching = False
        self._fingers_out = False

    @property
    def is_pinching(self) -> bool:
        return self._pinching

    @property
    def is_holding(self) -> bool:
        return self._pinching and self._fingers_out

    def reset(self) -> None:
        self._pinching = False
        self._fingers_out = False

    def push(self, pinch_distance: float, other_extensions) -> bool:
        """Feed one frame. Distances are in palm widths."""
        if not self._pinching and pinch_distance < self._close_below:
            self._pinching = True
        elif self._pinching and pinch_distance > self._open_above:
            self._pinching = False

        threshold = self._folded_below if self._fingers_out else self._extended_above
        self._fingers_out = sum(1 for e in other_extensions if e > threshold) >= self._require

        return self.is_holding


def measure_hand(landmarks_px) -> dict:
    """Reduce a hand to the scalars the knob needs.

    ``landmarks_px`` maps a landmark index to an ``(x, y)`` pixel tuple. Keeping
    this separate from MediaPipe types is what lets the whole gesture path be
    unit tested with plain tuples and no camera.
    """
    wrist = landmarks_px[WRIST]
    width = palm_width(landmarks_px[INDEX_MCP], landmarks_px[PINKY_MCP])
    width = max(width, 1e-6)

    pinch = math.hypot(landmarks_px[THUMB_TIP][0] - landmarks_px[INDEX_TIP][0],
                       landmarks_px[THUMB_TIP][1] - landmarks_px[INDEX_TIP][1]) / width

    extensions = [
        math.hypot(landmarks_px[tip][0] - wrist[0],
                   landmarks_px[tip][1] - wrist[1]) / width
        for tip in _OTHER_TIPS
    ]

    return {
        "angle": hand_rotation(wrist, landmarks_px[MIDDLE_MCP]),
        "pinch": pinch,
        "extensions": extensions,
        "palm_width": width,
    }


class EffectKnobController:
    """Binds the rotary knob to one selected effect at a time.

    Selecting an effect adopts its current value, so the knob always starts
    from where the sound already is rather than snapping it to whatever the
    previous effect was left on.
    """

    def __init__(self, engine, effect_names, **knob_kwargs):
        self._engine = engine
        self._names = tuple(effect_names)
        self._index = 0
        self._knob = RotaryKnob(**knob_kwargs)
        self._knob.set(self._engine.get_effect(self._names[0]))

    @property
    def selected(self) -> str:
        return self._names[self._index]

    @property
    def value(self) -> float:
        return self._knob.value

    @property
    def engaged(self) -> bool:
        return self._knob.engaged

    def select(self, name: str) -> bool:
        if name not in self._names:
            return False
        self._index = self._names.index(name)
        self._knob.set(self._engine.get_effect(name))
        self._knob.release()
        return True

    def cycle(self, step: int = 1) -> str:
        self._index = (self._index + step) % len(self._names)
        self._knob.set(self._engine.get_effect(self.selected))
        self._knob.release()
        return self.selected

    def nudge(self, delta: float) -> float:
        """Step the selected effect, e.g. from a spoken "up" or "down".

        The knob's own value is moved with it so that gripping afterwards
        continues from where the voice left off rather than snapping back to
        whatever the hand last set.
        """
        value = _clamp01(self._engine.get_effect(self.selected) + delta)
        self._engine.set_effect(self.selected, value)
        self._knob.set(value)
        return value

    def update(self, angle: float, gripping: bool) -> float:
        value = self._knob.update(angle, gripping)
        if gripping:
            self._engine.set_effect(self.selected, value)
        return value

    def release(self) -> None:
        self._knob.release()


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))
