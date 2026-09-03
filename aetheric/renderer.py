"""Rendering for Aetheric Geometry.

Two layers with different jobs, drawn with different tools:

* **The gesture layer** — skeleton, thread, polygon — stays in OpenCV. It is
  geometry over live video, it changes every frame, and ``LINE_AA`` is
  perfectly good at drawing a line.

* **The HUD** — status bar, hints, hold panel — goes through :mod:`ui`, which
  renders TrueType text and antialiased shapes via Pillow.

The HUD is laid out as a hardware panel rather than a web dashboard: modules
separated by hairline rules, a label above its value in every one, condensed
uppercase for the labels and a monospaced face for anything numeric. The
reference is a channel strip, and the reason is that a strip is *scanned* —
values are compared down a column at a glance while both hands are busy — which
is exactly the reading task here.

Colour carries meaning and nothing else: amber marks the one control being
touched, green means sound is present, red means recording. The effect meters
are deliberately all one quiet steel, because giving each its own hue would
make six things shout equally and leave nothing to mark the one that matters.

Layout constants come from ``ui.metrics(frame_height)`` so the HUD holds its
proportions from a small window up to fullscreen.
"""
import time

import cv2
import numpy as np

from . import ui
from .config import COOLDOWN_FRAMES, PINCH_THRESH
from .gestures import INDEX, PINKY, THUMB, dist, lm_px

# ── Gesture-layer colours (BGR, for OpenCV) ───────────────────────────────────
NEAR_WHITE = (243, 236, 233)
TEXT_DIM = (170, 147, 139)
NEON_GREEN = (128, 222, 74)
NEON_CYAN = (220, 205, 56)
NEON_PINK = (255, 128, 206)
WARM_RED = (96, 96, 255)

STATE_COLOR = {"IDLE": TEXT_DIM, "HILO": NEON_GREEN, "POLIGONO": NEON_CYAN}

_STATE_LABEL = {"IDLE": "IDLE", "HILO": "THREAD", "POLIGONO": "POLYGON"}

_HINTS = {
    "IDLE": ("Double pinch to start a thread", ""),
    "HILO": ("Kiss or pyramid for a polygon", "Double pinch to stop"),
    "POLIGONO": ("X pitch  ·  Y filter", "Pinch a finger to add a point"),
}

#: Keybinds, as (key, action) so the key can be set brighter than its label.
_KEYBINDS = (("H", "hold"), ("G", "release"), ("Tab", "next"),
             ("1-6", "effect"), ("Z X C V", "wave"), ("R", "rec"),
             ("Q", "quit"))

# Panel geometry from the design: 320 wide, inset 20 from the top and right,
# stopping 56 above the video area's bottom edge.
_PANEL_W = 320
_PANEL_INSET = 20
_PANEL_BOTTOM_GAP = 56
_BAR_H = 66


# ══════════════════════════════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════════════════════════════
_bar_cache = ui.TileCache()
_hints_cache = ui.TileCache()
_fps_cache = ui.TileCache()
_hold_cache = ui.TileCache()
_note_cache = ui.TileCache()


def draw_status_bar(img, state: str, waveform: str, n_verts: int,
                    note: str, effects: dict, W: int, H: int):
    """Bottom bar: MODE, NOTE and WAVE spread across a 66 px strip.

    The effect meters that used to live here now belong to the hold panel,
    which is where the design puts every value that can be turned. One place
    to look for a number beats two.
    """
    m = ui.metrics(H)
    bar_h = m.px(_BAR_H)
    y0 = H - bar_h

    ui.scrim(img, 0, y0, W, bar_h, top=0.85)

    key = (W, H, state, waveform, note)
    cached = _bar_cache.get(key)
    if cached is not None:
        ui.blit_array(img, cached, 0, y0)
        return

    o = ui.Overlay(W, bar_h, ss=1)
    o.line(0, 0, W, 0, fill=ui.BORDER, alpha=18)

    pad = m.px(24)
    label_y = bar_h // 2 - m.px(10)
    value_y = bar_h // 2 + m.px(11)

    def module(at_x, label, value, *, colour=ui.TEXT, kind="heading",
               anchor="l"):
        o.text(at_x, label_y, label, size=m.px(9), kind="label",
               fill=ui.FAINT, alpha=ui.A_FAINT, anchor=anchor + "m",
               tracking=m.px(1.1))
        o.text(at_x, value_y, value, size=m.px(16), kind=kind, fill=colour,
               anchor=anchor + "m", tracking=m.px(0.3))

    # -- mode ----------------------------------------------------------------
    module(pad, "MODE", _STATE_LABEL.get(state, state),
           colour=ui.LIVE if state != "IDLE" else ui.MUTED)

    # -- note, centred like the design's space-between middle child ----------
    if note:
        module(W // 2 - m.px(120), "NOTE", note, kind="mono_bold")
    if state == "POLIGONO":
        module(W // 2 - m.px(30), "POINTS", str(n_verts), kind="mono_bold")

    # -- wave ----------------------------------------------------------------
    card_w, card_h = m.px(72), m.px(34)
    card_x = W - card_w - pad
    _waveform_path(o, waveform, card_x, (bar_h - card_h) // 2 + m.px(2),
                   card_w, card_h - m.px(4), ui.ACCENT)
    module(card_x - m.px(16), "WAVE", waveform, anchor="r")

    ui.blit_array(img, _bar_cache.put(key, o.rasterize()), 0, y0)


def _waveform_path(o: "ui.Overlay", shape: str, x, y, w, h, colour,
                   cycles: int = 2):
    """The selected waveform, as a polyline.

    Two cycles by default: one cycle of a sawtooth is a single diagonal stroke,
    which reads as a stray line rather than as a waveform. The repeat is what
    makes the shape legible.
    """
    n = max(16, int(w))
    t = (np.linspace(0.0, float(cycles), n, endpoint=False)) % 1.0
    if shape == "SIN":
        wave = np.sin(2 * np.pi * t)
    elif shape == "SAW":
        wave = 2.0 * t - 1.0
    elif shape == "SQUARE":
        wave = np.where(t < 0.5, 1.0, -1.0)
    else:
        wave = 4.0 * np.abs(t - 0.5) - 1.0
    ys = y + h / 2.0 - wave * (h / 2.0 - 1)
    xs = x + np.arange(n) * (w / n)
    o.polyline(list(zip(xs, ys)), fill=colour, width=2)


def draw_hints(img, state: str, recording: bool = False):
    """Gesture hint bottom-left, keybinds bottom-right, over the video.

    Both sit inside the video area rather than in the bar, so the bar stays a
    readout and this stays guidance - the design keeps those two jobs apart.
    """
    H, W = img.shape[:2]
    m = ui.metrics(H)
    strip_h = m.px(26)
    y0 = H - m.px(_BAR_H) - strip_h - m.px(10)

    key = (W, H, state, recording)
    cached = _hints_cache.get(key)
    if cached is None:
        o = ui.Overlay(W, strip_h, ss=1)
        pad = m.px(20)
        cy = strip_h // 2

        # "GESTURE:" and "STOP:" label what each half is, so the line reads as
        # two named facts rather than one run-on sentence.
        left, right_hint = _HINTS.get(state, ("", ""))
        x = pad
        o.text(x, cy, "GESTURE:", size=m.px(10), kind="label", fill=ui.TEXT,
               alpha=ui.A_FAINT, anchor="lm", tracking=m.px(0.9))
        x += o.text_width("GESTURE:", size=m.px(10), kind="label",
                          tracking=m.px(0.9)) + m.px(7)
        o.text(x, cy, left, size=m.px(11), kind="body", fill=ui.TEXT,
               alpha=ui.A_MUTED, anchor="lm")
        if right_hint:
            x += o.text_width(left, size=m.px(11), kind="body") + m.px(12)
            o.text(x, cy, "·", size=m.px(11), kind="body", fill=ui.TEXT,
                   alpha=60, anchor="lm")
            x += m.px(11)
            o.text(x, cy, "STOP:", size=m.px(10), kind="label", fill=ui.TEXT,
                   alpha=ui.A_FAINT, anchor="lm", tracking=m.px(0.9))
            x += o.text_width("STOP:", size=m.px(10), kind="label",
                              tracking=m.px(0.9)) + m.px(7)
            o.text(x, cy, right_hint, size=m.px(11), kind="body",
                   fill=ui.TEXT, alpha=ui.A_MUTED, anchor="lm")

        # Keybinds run right to left so the row ends flush with the margin.
        if recording:
            o.text(W - pad, cy, "R  stop recording", size=m.px(11),
                   kind="label", fill=ui.DANGER, anchor="rm")
        else:
            rx = W - pad
            for keyname, action in reversed(_KEYBINDS):
                aw = o.text_width(action, size=m.px(11), kind="body")
                o.text(rx, cy, action, size=m.px(11), kind="body",
                       fill=ui.TEXT, alpha=ui.A_FAINT, anchor="rm")
                rx -= aw + m.px(5)
                kw = o.text_width(keyname, size=m.px(11), kind="label")
                o.text(rx, cy, keyname, size=m.px(11), kind="label",
                       fill=ui.TEXT, alpha=155, anchor="rm")
                rx -= kw + m.px(13)
        cached = _hints_cache.put(key, o.rasterize())
    ui.blit_array(img, cached, 0, y0)


def draw_fps(img, fps: int, W: int):
    """Frame rate, top-left beside the voice pill.

    Not top-right, where the design puts the effects panel: a badge there
    lands on the panel header.
    """
    H = img.shape[0]
    m = ui.metrics(H)
    w, h = m.px(64), m.px(32)

    key = (W, H, fps)
    cached = _fps_cache.get(key)
    if cached is None:
        o = ui.Overlay(w, h)
        o.panel(0, 0, w, h, radius=h // 2, fill=ui.SURFACE, alpha=158,
                border=ui.BORDER, border_alpha=ui.A_HAIR)
        o.text(w // 2, h // 2, f"{fps} FPS", size=m.px(10), kind="label",
               fill=ui.TEXT, alpha=ui.A_FAINT, anchor="mm", tracking=m.px(0.9))
        cached = _fps_cache.put(key, o.rasterize())
    ui.blit_array(img, cached, m.px(20) + m.px(150) + m.px(8), m.px(20))


def draw_recording_indicator(img, W: int):
    """Pulsing REC pill, right of the FPS badge."""
    H = img.shape[0]
    m = ui.metrics(H)
    w, h = m.px(74), m.px(32)
    # A sine pulse rather than a hard blink: it reads as "live" instead of
    # "something is broken".
    pulse = 0.55 + 0.45 * float(np.sin(time.time() * 4.0))
    o = ui.Overlay(w, h)
    o.panel(0, 0, w, h, radius=h // 2, fill=ui.SURFACE, alpha=158,
            border=ui.DANGER, border_alpha=int(50 + 80 * pulse))
    o.dot(m.px(16), h // 2, m.px(4), fill=ui.DANGER,
          alpha=int(150 + 105 * pulse))
    o.text(m.px(29), h // 2, "REC", size=m.px(10), kind="label",
           fill=ui.DANGER, anchor="lm", tracking=m.px(0.9))
    o.blit(img, m.px(20) + m.px(150) + m.px(8) + m.px(64) + m.px(8), m.px(20))


_voice_cache = ui.TileCache()


def draw_voice_badge(img, wake_word, awake: bool, W: int, H: int):
    """Whether the microphone is currently accepting commands.

    Without this the wake word is invisible state: you say a command, nothing
    happens, and there is no way to tell whether it was misheard or simply not
    being listened for.
    """
    if wake_word is None:
        return
    m = ui.metrics(H)
    label = "LISTENING" if awake else f'SAY "{wake_word.upper()}"'
    w, h = m.px(150), m.px(32)

    key = (H, label, awake)
    cached = _voice_cache.get(key)
    if cached is None:
        colour = ui.ACCENT if awake else ui.TEXT
        o = ui.Overlay(w, h)
        o.panel(0, 0, w, h, radius=h // 2, fill=ui.SURFACE, alpha=158,
                border=ui.BORDER, border_alpha=ui.A_HAIR)
        o.text(m.px(16), h // 2, label, size=m.px(11), kind="label",
               fill=ui.TEXT, alpha=192, anchor="lm", tracking=m.px(1.1))
        o.mic_glyph(w - m.px(19), h // 2, m.px(15), fill=colour,
                    alpha=255 if awake else 110, width=2)
        cached = _voice_cache.put(key, o.rasterize())
    ui.blit_array(img, cached, m.px(20), m.px(20))


def draw_note_label(img, note: str, cx: int, cy: int, color: tuple):
    """Floating pill that follows the hands."""
    if not note:
        return
    H, W = img.shape[:2]
    m = ui.metrics(H)
    w, h = m.px(64), m.px(30)
    x = int(np.clip(cx - w // 2, m.px(8), W - w - m.px(8)))
    y = int(np.clip(cy - h - m.px(18), m.px(8), H - h - m.px(8)))

    accent = ui.LIVE if color == NEON_GREEN else ui.CYAN
    key = (H, note, accent)
    cached = _note_cache.get(key)
    if cached is None:
        o = ui.Overlay(w, h)
        o.panel(0, 0, w, h, radius=h // 2, fill=ui.BG, alpha=205,
                border=accent, border_alpha=120)
        o.text(w // 2, h // 2, note, size=m.px(15), kind="mono_bold",
               fill=ui.TEXT, anchor="mm")
        cached = _note_cache.put(key, o.rasterize())
    ui.blit_array(img, cached, x, y)


# ══════════════════════════════════════════════════════════════════════════════
#  Hold panel
# ══════════════════════════════════════════════════════════════════════════════
_hold_chrome_cache = ui.TileCache()
_hold_dial_cache = ui.TileCache()
_hold_grip_cache = ui.TileCache()

#: Short forms, so a blend caption fits inside the dial.
_WAVE_SHORT = {"SIN": "SINE", "TRIANGLE": "TRI", "SAW": "SAW", "SQUARE": "SQR"}


def _shape_caption(value: float) -> str:
    """Name the shape, or the two it is currently between.

    A bare "SHAPE 40" says nothing about what you are listening to. Naming the
    pair is what makes a morph knob playable: you can see you are three
    quarters of the way from triangle into sawtooth without stopping to listen.
    """
    from synth import WAVE_RAMP

    pos = max(0.0, min(1.0, float(value))) * (len(WAVE_RAMP) - 1)
    low = int(pos)
    blend = pos - low
    if blend < 0.02:
        return _WAVE_SHORT[WAVE_RAMP[low]]
    if blend > 0.98:
        return _WAVE_SHORT[WAVE_RAMP[min(low + 1, len(WAVE_RAMP) - 1)]]
    high = min(low + 1, len(WAVE_RAMP) - 1)
    return f"{_WAVE_SHORT[WAVE_RAMP[low]]} {_WAVE_SHORT[WAVE_RAMP[high]]}"


def draw_hold_panel(img, effects: dict, order, selected: str,
                    gripping: bool, angle, W: int, H: int):
    """The held-sound surface: one dial for the selected effect, all values listed.

    Sits on the right so it never lands under the hands, which are busy.

    Split across three cached tiles rather than one, because they change at
    completely different rates. The card, its header and the effect names only
    move when the selection or the grip does; the dial and the value rows move
    on every frame of a turn. Rebuilding the whole panel for a knob sweep cost
    ~21 ms a frame, which is most of a frame's budget spent redrawing pixels
    that did not change. The dial keeps 2x supersampling because it is the only
    curved element; the rest is text and straight rules and does not need it.
    """
    m = ui.metrics(H)
    inset = m.px(_PANEL_INSET)
    w = m.px(_PANEL_W)
    row_h = m.px(28)
    head_h = m.px(50)
    dial_block = m.px(192)
    x, y = W - w - inset, inset
    h = H - m.px(_BAR_H) - m.px(_PANEL_BOTTOM_GAP) - inset
    px = m.px(20)
    accent = ui.ACCENT

    # -- chrome: card, header, row labels ------------------------------------
    chrome_key = (W, H, selected, gripping, tuple(order))
    chrome = _hold_chrome_cache.get(chrome_key)
    if chrome is None:
        o = ui.Overlay(w, h)
        o.panel(0, 0, w, h, radius=m.px(16), fill=ui.SURFACE, alpha=158,
                border=ui.BORDER, border_alpha=ui.A_HAIR)
        o.dot(px + m.px(4), m.px(22), m.px(4), fill=accent)
        o.text(px + m.px(16), m.px(22), "HOLD", size=m.px(12), kind="heading",
               fill=ui.TEXT, anchor="lm", tracking=m.px(1.0))
        o.text(w - px, m.px(22), "AETHERIC MODULATION", size=m.px(9),
               kind="label", fill=ui.TEXT, alpha=ui.A_FAINT, anchor="rm",
               tracking=m.px(0.9))
        o.line(px, m.px(40), w - px, m.px(40), fill=ui.BORDER, alpha=20)

        ty = head_h + dial_block + m.px(10)
        for name in order:
            on = name == selected
            if on:
                o.rect(m.px(12), ty - m.px(3), w - m.px(24), row_h + m.px(8),
                       radius=m.px(6), fill=accent, alpha=31)
            o.text(px, ty + row_h // 2 - m.px(2), name.upper(), size=m.px(10),
                   kind="label", fill=accent if on else ui.TEXT,
                   alpha=255 if on else 140, anchor="lm", tracking=m.px(0.9))
            ty += row_h + m.px(6)
        chrome = _hold_chrome_cache.put(chrome_key, o.rasterize())
    ui.blit_array(img, chrome, x, y)

    # -- dial ----------------------------------------------------------------
    value = float(effects.get(selected, 0.0))
    dial_r = m.px(68)
    dial_size = dial_r * 2 + m.px(36)
    needle = (round(float(np.degrees(angle)) / 3.0) * 3.0
              if (gripping and angle is not None) else None)
    caption = _shape_caption(value) if selected == "shape" else selected.upper()
    dial_key = (H, selected, caption, round(value, 2), needle)
    dial = _hold_dial_cache.get(dial_key)
    if dial is None:
        o = ui.Overlay(dial_size, dial_size)
        c = dial_size // 2
        o.dial_segmented(c, c, dial_r, value, fill=accent, segments=44,
                         width=m.px(9), needle_deg=needle)
        o.text(c, c - m.px(6), f"{value * 100:.0f}", size=m.px(38),
               kind="mono_bold", fill=ui.TEXT, anchor="mm")
        o.text(c, c + m.px(24), caption, size=m.px(10), kind="label",
               fill=accent, anchor="mm", tracking=m.px(1.2))
        dial = _hold_dial_cache.put(dial_key, o.rasterize())
    ui.blit_array(img, dial, x + (w - dial_size) // 2,
                  y + head_h + (dial_block - dial_size) // 2)

    # Directly under the ring, not at the bottom of the block: it labels the
    # dial, and a caption floating in open space stops looking like one.
    grip_y = y + head_h + dial_block // 2 + dial_r + m.px(6)
    grip_key = (H, gripping)
    grip = _hold_grip_cache.get(grip_key)
    if grip is None:
        o = ui.Overlay(w, m.px(18), ss=1)
        o.text(w // 2, m.px(9), "TURN" if gripping else "PINCH TO GRIP",
               size=m.px(9), kind="label",
               fill=accent if gripping else ui.TEXT,
               alpha=255 if gripping else 90, anchor="mm", tracking=m.px(1.0))
        grip = _hold_grip_cache.put(grip_key, o.rasterize())
    ui.blit_array(img, grip, x, grip_y)

    # -- value rows ----------------------------------------------------------
    # The selected row carries a 0/value/100 scale under its track. Only the
    # selected one: putting a scale under all seven turns the panel into a wall
    # of small numbers, and the scale is only wanted where a value is moving.
    values = tuple(round(float(effects.get(n, 0.0)), 2) for n in order)
    row_pitch = row_h + m.px(6)
    rows_h = row_pitch * len(order)
    rows_key = (W, H, selected, tuple(order), values)
    rows = _hold_cache.get(rows_key)
    if rows is None:
        o = ui.Overlay(w, rows_h, ss=1)
        ty = 0
        for name, val in zip(order, values):
            on = name == selected
            colour = accent if on else ui.TEXT
            alpha = 255 if on else 140
            bar_x = px + m.px(64)
            bar_w = w - bar_x - px - m.px(40)
            cy = ty + row_h // 2 - m.px(2)

            o.meter(bar_x, cy, bar_w, m.px(4), val, fill=colour,
                    track_alpha=26)
            o.text(w - px, cy + m.px(2), f"{val * 100:.0f}", size=m.px(11),
                   kind="mono", fill=colour, alpha=alpha, anchor="rm")

            if on:
                scale_y = cy + m.px(12)
                o.text(bar_x, scale_y, "0", size=m.px(8), kind="mono",
                       fill=ui.TEXT, alpha=80, anchor="lm")
                o.text(bar_x + bar_w, scale_y, "100", size=m.px(8),
                       kind="mono", fill=ui.TEXT, alpha=80, anchor="rm")
                if 0.12 < val < 0.88:
                    o.text(bar_x + bar_w * val, scale_y, f"{val * 100:.0f}",
                           size=m.px(8), kind="mono", fill=accent, anchor="mm")
            ty += row_pitch
        rows = _hold_cache.put(rows_key, o.rasterize())
    ui.blit_array(img, rows, x, y + head_h + dial_block + m.px(10))


def draw_cooldown_border(img, n: int, W: int, H: int):
    """Vignette that closes in as the hands stay lost."""
    ratio = min(1.0, n / COOLDOWN_FRAMES)
    thickness = max(2, int(ui.metrics(H).px(14) * ratio))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W - 1, H - 1), WARM_RED, thickness)
    alpha = 0.12 + 0.33 * ratio
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ══════════════════════════════════════════════════════════════════════════════
#  Gesture layer (OpenCV)
# ══════════════════════════════════════════════════════════════════════════════
def draw_skeleton(img, results):
    if not results.multi_hand_landmarks:
        return
    import mediapipe as mp
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    lm_s = mp_draw.DrawingSpec(color=(96, 82, 68), thickness=1, circle_radius=2)
    con_s = mp_draw.DrawingSpec(color=(74, 63, 52), thickness=1)
    for hand in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS, lm_s, con_s)


def draw_pinch_guides(img, hand, W: int, H: int):
    """In IDLE: a threshold ring around each fingertip.

    The ring fades toward white as the finger closes on the thumb, so the pinch
    distance is visible before it triggers rather than after.
    """
    thumb_pt = lm_px(hand, THUMB, W, H)
    for lid, col in ((INDEX, NEON_CYAN), (12, NEON_GREEN),
                     (16, NEON_PINK), (PINKY, TEXT_DIM)):
        tip = lm_px(hand, lid, W, H)
        ratio = min(1.0, dist(thumb_pt, tip) / PINCH_THRESH)
        ring = tuple(int(col[c] * (1.0 - ratio) + NEAR_WHITE[c] * ratio)
                     for c in range(3))
        cv2.circle(img, tip, int(PINCH_THRESH / 2), ring, 1, cv2.LINE_AA)


def draw_hilo(img, lh, rh, W: int, H: int, color: tuple):
    a = lm_px(lh, INDEX, W, H)
    b = lm_px(rh, INDEX, W, H)
    overlay = img.copy()
    cv2.line(overlay, a, b, color, 9, cv2.LINE_AA)
    for pt in (a, b):
        cv2.circle(overlay, pt, 20, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

    cv2.line(img, a, b, color, 2, cv2.LINE_AA)
    for pt in (a, b):
        cv2.circle(img, pt, 8, color, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 8, NEAR_WHITE, 1, cv2.LINE_AA)


def draw_polygon(img, pts: list, color: tuple):
    if len(pts) < 3:
        return
    arr = np.array(pts, np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [arr], color)
    cv2.addWeighted(overlay, 0.16, img, 0.84, 0, img)
    cv2.polylines(img, [arr], True, color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(img, pt, 5, NEAR_WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 5, color, 2, cv2.LINE_AA)
