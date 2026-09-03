"""A small UI toolkit for drawing the HUD over the camera frame.

Why this exists
---------------
The HUD used to be drawn with ``cv2.putText`` and the built-in Hershey fonts.
Those are single-stroke vector fonts from 1967 with no hinting, no kerning and
one weight; OpenCV's ``LINE_AA`` antialiases the strokes but cannot make the
letterforms anything other than what they are. No amount of tuning colours or
positions gets past that, because the typeface itself is the problem.

Everything here therefore renders through Pillow with real TrueType faces, and
draws into an RGBA tile at 2x before downscaling it onto the frame. The 2x pass
matters for more than text: Pillow's ``arc`` and ``rounded_rectangle`` are not
antialiased at all, so the dial and the panel corners would otherwise stair-step
badly. Supersampling gives them clean edges for a couple of milliseconds per
panel, and the panels are small.

Layout is derived from frame height rather than hard-coded, so the HUD keeps its
proportions from a windowed 720p preview up to a fullscreen projector, instead
of shrinking into a corner.

Colour convention: everything in this module is RGB, because that is what Pillow
speaks. :func:`Overlay.blit` converts to the frame's BGR on the way out, so a
caller never has to think about channel order.
"""
from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Palette ───────────────────────────────────────────────────────────────────
# The reference is studio hardware, not a web dashboard: an anodized rack panel,
# silkscreened labels, one amber lamp. Two decisions follow from that.
#
# The neutrals are biased warm rather than being pure grey. A neutral #808080
# next to an amber accent reads as unconsidered; pulling a few points of red
# and yellow into the whole ramp makes the panel and the lamp belong to each
# other.
#
# There is exactly one accent, and it is spent on the thing being touched.
# Green and red are not accents here, they are signal semantics borrowed from
# every mixing desk ever built - green means it is sounding, red means it is
# recording - so they carry meaning rather than decoration.
GROUND      = (5, 5, 5)         # #050505 - deepest, behind everything
SURFACE     = (10, 10, 11)      # #0a0a0b - panel face, used translucent
SURFACE_HI  = (26, 26, 29)      # #1a1a1d - raised module
BORDER      = (255, 255, 255)   # used at low alpha as a hairline
TEXT        = (244, 242, 238)   # #f4f2ee
MUTED       = (244, 242, 238)   # same ink at ~45% alpha
FAINT       = (244, 242, 238)   # same ink at ~35% alpha

ACCENT      = (255, 140, 59)    # #FF8C3B - selection, the control being turned
LIVE        = (138, 224, 92)    # #8AE05C - mode / signal present
DANGER      = (229, 72, 77)     # #e5484d - recording
STEEL       = (255, 140, 59)    # waveform readout follows the accent

#: Alphas that go with MUTED and FAINT, since both are the same ink.
A_TEXT   = 255
A_MUTED  = 115   # ~0.45
A_FAINT  = 90    # ~0.35
A_HAIR   = 23    # ~0.09 borders

# Kept as names the renderer already uses, mapped onto the new ramp.
BG          = GROUND
AMBER       = ACCENT
CYAN        = STEEL
MAGENTA     = STEEL

#: Face roles, mapped onto what Windows actually ships.
#:
#: The design specifies Inter and IBM Plex Mono. Neither is installed by
#: default and the CSP-free desktop build has nowhere to fetch a webfont from,
#: so each role resolves to the closest thing present: Segoe UI Variable is a
#: neutral humanist grotesque cut from the same intent as Inter, and Cascadia
#: Mono is a modern monospace with the same upright, even-width figures as IBM
#: Plex Mono. Anything numeric stays monospaced, because proportional digits
#: change width as a value counts and the panel appears to twitch.
_FONT_ROLES = {
    "label":   ("SegUIVar.ttf", 600, None),
    "heading": ("SegUIVar.ttf", 700, None),
    "body":    ("SegUIVar.ttf", 400, None),
    "mono":    ("CascadiaMono.ttf", None, None),
    "mono_bold": ("CascadiaMono.ttf", 600, None),
}

_FALLBACKS = ("segoeui.ttf", "seguisb.ttf", "calibri.ttf", "arial.ttf",
              "DejaVuSans.ttf")

_FONT_DIRS = (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"),
              Path("/Library/Fonts"), Path("/System/Library/Fonts"))

# Names the renderer used before the type system was reworked.
_ALIASES = {"regular": "body", "medium": "label", "bold": "heading"}


def _find(name: str) -> Path | None:
    for directory in _FONT_DIRS:
        path = directory / name
        if path.exists():
            return path
    return None


@functools.lru_cache(maxsize=256)
def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a cached face for a role, applying its variable-font axes."""
    role = _ALIASES.get(kind, kind)
    filename, weight, width = _FONT_ROLES.get(role, _FONT_ROLES["body"])

    path = _find(filename)
    if path is not None:
        try:
            face = ImageFont.truetype(str(path), size)
            if weight is not None or width is not None:
                try:
                    axes = face.get_variation_axes()
                    values = []
                    for axis in axes:
                        axis_name = axis["name"]
                        if isinstance(axis_name, bytes):
                            axis_name = axis_name.decode("ascii", "ignore")
                        target = weight if axis_name.lower() == "weight" else width
                        values.append(target if target is not None
                                      else axis["default"])
                    face.set_variation_by_axes(values)
                except (OSError, AttributeError):
                    pass    # static face; the default instance is fine
            return face
        except OSError:
            pass

    for name in _FALLBACKS:
        found = _find(name)
        if found is not None:
            try:
                return ImageFont.truetype(str(found), size)
            except OSError:
                continue
    return ImageFont.load_default()


class Metrics:
    """Sizes derived from frame height, so the HUD scales with the window."""

    def __init__(self, frame_height: int):
        # 720p is the reference design size; clamp so a tiny or enormous window
        # still produces something usable.
        self.k = max(0.72, min(2.2, frame_height / 720.0))

    def px(self, value: float) -> int:
        return max(1, int(round(value * self.k)))


@functools.lru_cache(maxsize=8)
def metrics(frame_height: int) -> Metrics:
    return Metrics(frame_height)


class Overlay:
    """An RGBA tile drawn at 2x and composited down onto a BGR frame.

    All coordinates passed to the helpers are in logical (1x) pixels.
    """

    def __init__(self, width: int, height: int, ss: int = 2):
        """``ss`` is the supersampling factor.

        Use 2 for anything with a curve in it — dials, rings, large corner
        radii — where Pillow's own rasteriser does not antialias and the edges
        would otherwise stair-step. Use 1 for wide, mostly-text panels: Pillow
        already antialiases TrueType glyphs properly, so the second pass buys
        nothing there and a full-width tile is exactly where the resize costs
        the most.
        """
        self.SS = max(1, int(ss))
        self.w = max(1, int(width))
        self.h = max(1, int(height))
        self._img = Image.new("RGBA", (self.w * self.SS, self.h * self.SS), (0, 0, 0, 0))
        self._d = ImageDraw.Draw(self._img)

    # -- primitives ----------------------------------------------------------
    def panel(self, x, y, w, h, radius=14, fill=SURFACE, alpha=225,
              border=BORDER, border_alpha=28, border_width=1):
        """A translucent rounded card."""
        s = self.SS
        box = (x * s, y * s, (x + w) * s - 1, (y + h) * s - 1)
        self._d.rounded_rectangle(
            box, radius=radius * s,
            fill=(*fill, alpha),
            outline=(*border, border_alpha) if border else None,
            width=border_width * s)

    def rect(self, x, y, w, h, radius=0, fill=SURFACE, alpha=255):
        s = self.SS
        box = (x * s, y * s, (x + w) * s - 1, (y + h) * s - 1)
        if radius:
            self._d.rounded_rectangle(box, radius=radius * s, fill=(*fill, alpha))
        else:
            self._d.rectangle(box, fill=(*fill, alpha))

    def line(self, x0, y0, x1, y1, fill=BORDER, alpha=40, width=1):
        s = self.SS
        self._d.line((x0 * s, y0 * s, x1 * s, y1 * s),
                     fill=(*fill, alpha), width=width * s)

    def polyline(self, points, fill=TEXT, alpha=255, width=2):
        """Draw a path through logical-space ``(x, y)`` points."""
        s = self.SS
        if len(points) < 2:
            return
        self._d.line([(float(px) * s, float(py) * s) for px, py in points],
                     fill=(*fill, alpha), width=max(1, int(width * s)),
                     joint="curve")

    def dot(self, cx, cy, r, fill=TEXT, alpha=255):
        s = self.SS
        self._d.ellipse(((cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s),
                        fill=(*fill, alpha))

    def ring(self, cx, cy, r, width=2, fill=BORDER, alpha=40):
        s = self.SS
        self._d.ellipse(((cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s),
                        outline=(*fill, alpha), width=width * s)

    def arc(self, cx, cy, r, start_deg, end_deg, width=3, fill=ACCENT, alpha=255):
        s = self.SS
        if end_deg <= start_deg:
            return
        self._d.arc(((cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s),
                    start_deg, end_deg, fill=(*fill, alpha), width=width * s)

    # -- text ----------------------------------------------------------------
    def text(self, x, y, string, size=14, kind="regular", fill=TEXT, alpha=255,
             anchor="la", tracking=0.0):
        """Draw text. ``anchor`` follows Pillow's two-letter convention."""
        s = self.SS
        f = font(kind, int(round(size * s)))
        if not tracking:
            self._d.text((x * s, y * s), string, font=f,
                         fill=(*fill, alpha), anchor=anchor)
            return

        # Letter-spacing, for small uppercase labels where it buys legibility.
        gap = tracking * s
        widths = [self._d.textlength(ch, font=f) for ch in string]
        total = sum(widths) + gap * max(0, len(string) - 1)
        cx = x * s
        if anchor[0] == "m":
            cx -= total / 2
        elif anchor[0] == "r":
            cx -= total
        for ch, wch in zip(string, widths):
            self._d.text((cx, y * s), ch, font=f, fill=(*fill, alpha),
                         anchor="l" + anchor[1])
            cx += wch + gap

    def text_width(self, string, size=14, kind="regular", tracking=0.0) -> float:
        f = font(kind, int(round(size * self.SS)))
        w = self._d.textlength(string, font=f)
        if tracking:
            w += tracking * self.SS * max(0, len(string) - 1)
        return w / self.SS

    # -- components ----------------------------------------------------------
    def meter(self, x, y, w, h, value, fill=ACCENT, track_alpha=34):
        """Horizontal bar with rounded caps."""
        r = h / 2.0
        self.rect(x, y, w, h, radius=r, fill=BORDER, alpha=track_alpha)
        filled = int(round(w * max(0.0, min(1.0, value))))
        if filled >= h:
            self.rect(x, y, filled, h, radius=r, fill=fill, alpha=255)
        elif filled > 0:
            # Too short for the rounded cap geometry; a dot keeps it honest.
            self.dot(x + r, y + r, r, fill=fill, alpha=255)

    def dial_segmented(self, cx, cy, r, value, fill=ACCENT, segments=44,
                       width=9, gap_deg=1.6, needle_deg=None):
        """Ring of discrete lit segments, as on a hardware encoder collar.

        A solid arc tells you the value only if you read the number in the
        middle. A segmented one is countable: the eye picks up "about two
        thirds lit" from across a room, which is the whole point of a ring on
        a control you operate without looking at it.
        """
        start, sweep = 135.0, 270.0
        step = sweep / segments
        lit = value * segments

        for i in range(segments):
            a0 = start + i * step
            a1 = a0 + step - gap_deg
            if i + 1 <= lit:                      # fully lit
                colour, alpha = fill, 255
            elif i < lit:                         # the partial segment
                colour, alpha = fill, int(90 + 165 * (lit - i))
            else:
                colour, alpha = BORDER, 28
            self.arc(cx, cy, r, a0, a1, width=width, fill=colour, alpha=alpha)

        if needle_deg is not None:
            rad = np.radians(needle_deg)
            inner = r * 0.70
            self._d.line(
                ((cx + inner * np.cos(rad)) * self.SS,
                 (cy + inner * np.sin(rad)) * self.SS,
                 (cx + (r + width * 0.9) * np.cos(rad)) * self.SS,
                 (cy + (r + width * 0.9) * np.sin(rad)) * self.SS),
                fill=(*TEXT, 220), width=max(1, int(1.6 * self.SS)))

    def mic_glyph(self, cx, cy, h, fill=TEXT, alpha=200, width=2):
        """A microphone outline, for the voice pill.

        The design's plain dot says "something is on"; a microphone says what.
        On a control surface with several status dots already, the difference
        is whether the badge has to be learned or is simply read.
        """
        s = self.SS
        cap_w = h * 0.38
        cap_h = h * 0.52
        top = cy - h / 2.0
        self._d.rounded_rectangle(
            ((cx - cap_w / 2) * s, top * s, (cx + cap_w / 2) * s,
             (top + cap_h) * s),
            radius=(cap_w / 2) * s, outline=(*fill, alpha),
            width=max(1, int(width * s / 2)))
        # Cradle arc under the capsule, then the stem and base.
        cradle_r = h * 0.34
        self._d.arc(((cx - cradle_r) * s, (cy - cradle_r * 0.55) * s,
                     (cx + cradle_r) * s, (cy + cradle_r * 1.45) * s),
                    0, 180, fill=(*fill, alpha), width=max(1, int(width * s / 2)))
        self._d.line((cx * s, (cy + cradle_r * 0.9) * s,
                      cx * s, (cy + h / 2.0) * s),
                     fill=(*fill, alpha), width=max(1, int(width * s / 2)))

    def dial(self, cx, cy, r, value, fill=ACCENT, needle_deg=None, width=4,
             ticks=11):
        """270-degree gauge, opening at the bottom like a hardware knob.

        The tick ring is not decoration: an arc alone gives no sense of *how
        far* 60 % is without reading the number, which is the whole reason a
        physical knob has a scale silkscreened around it.
        """
        start, sweep = 135.0, 270.0

        if ticks:
            for i in range(ticks):
                angle = np.radians(start + sweep * i / (ticks - 1))
                major = i % ((ticks - 1) // 2 or 1) == 0
                r0 = r + width * 0.7 + (1 if major else 2)
                r1 = r0 + (width * 1.3 if major else width * 0.7)
                self._d.line(
                    ((cx + r0 * np.cos(angle)) * self.SS,
                     (cy + r0 * np.sin(angle)) * self.SS,
                     (cx + r1 * np.cos(angle)) * self.SS,
                     (cy + r1 * np.sin(angle)) * self.SS),
                    fill=(*BORDER, 90 if major else 45),
                    width=max(1, int(self.SS)))

        self.arc(cx, cy, r, start, start + sweep, width=width,
                 fill=BORDER, alpha=38)
        v = max(0.0, min(1.0, value))
        if v > 0.001:
            self.arc(cx, cy, r, start, start + sweep * v, width=width, fill=fill)

        # Where the hand actually is, when it is gripping. Starts well outside
        # the centre so it never crosses the value printed there.
        if needle_deg is not None:
            rad = np.radians(needle_deg)
            inner = r * 0.66
            self._d.line(
                ((cx + inner * np.cos(rad)) * self.SS,
                 (cy + inner * np.sin(rad)) * self.SS,
                 (cx + (r - width - 2) * np.cos(rad)) * self.SS,
                 (cy + (r - width - 2) * np.sin(rad)) * self.SS),
                fill=(*TEXT, 210), width=max(1, int(1.5 * self.SS)))

    # -- output --------------------------------------------------------------
    def rasterize(self) -> "Tile":
        """Downsample to logical size and return a composite-ready tile.

        ``BOX`` rather than ``LANCZOS``: for an exact 2x reduction a box filter
        *is* the correct supersampling resolve — it averages each 2x2 block —
        and it costs a fraction of what a windowed-sinc kernel does. LANCZOS
        would additionally ring on the hard edges that panels are made of.
        """
        img = (self._img if self.SS == 1
               else self._img.resize((self.w, self.h), Image.BOX))
        return Tile(np.asarray(img, dtype=np.uint8))

    def blit(self, frame: np.ndarray, x: int, y: int) -> None:
        """Alpha-composite this tile onto a BGR frame at (x, y)."""
        blit_array(frame, self.rasterize(), x, y)


class Tile:
    """A rasterised overlay, pre-chewed into the form compositing wants.

    Channel-order conversion and the float promotion used to happen on every
    frame inside the blit. Reversing RGB to BGR with a negative-stride slice
    produces a non-contiguous view, and promoting *that* to float32 is a
    strided element-by-element copy: it measured 5.9 ms for a 1280x88 tile,
    which is an entire frame's budget spent rearranging bytes that never
    change. Since the tile is cached anyway, all of it is done once here.

    What remains per frame is one multiply-add over the destination region.
    """

    __slots__ = ("h", "w", "premultiplied", "inv_alpha", "_scratch")

    def __init__(self, rgba: np.ndarray):
        self.h, self.w = rgba.shape[:2]
        alpha = rgba[:, :, 3].astype(np.float32) * (1.0 / 255.0)
        bgr = np.ascontiguousarray(rgba[:, :, 2::-1]).astype(np.float32)
        # Both stored 3-channel so OpenCV can consume them without broadcasting.
        self.premultiplied = np.ascontiguousarray(bgr * alpha[:, :, None])
        self.inv_alpha = np.ascontiguousarray(
            np.repeat((1.0 - alpha)[:, :, None], 3, axis=2))
        self._scratch = np.empty((self.h, self.w, 3), np.float32)


def blit_array(frame: np.ndarray, tile: Tile, x: int, y: int) -> None:
    """Alpha-composite a cached tile onto a BGR frame.

    Routed through OpenCV rather than numpy expressions: ``roi * inv + pm``
    allocates two full-size float32 temporaries per call and runs single
    threaded, where the equivalent OpenCV calls are SIMD, multithreaded and
    write into a buffer the tile already owns.
    """
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + tile.w), min(fh, y + tile.h)
    if x1 <= x0 or y1 <= y0:
        return

    sy0, sy1 = y0 - y, y1 - y
    sx0, sx1 = x0 - x, x1 - x
    roi = frame[y0:y1, x0:x1]
    scratch = tile._scratch[sy0:sy1, sx0:sx1]

    cv2.multiply(roi, tile.inv_alpha[sy0:sy1, sx0:sx1],
                 dst=scratch, dtype=cv2.CV_32F)
    cv2.add(scratch, tile.premultiplied[sy0:sy1, sx0:sx1], dst=scratch)
    cv2.convertScaleAbs(scratch, dst=roi)


class TileCache:
    """One-slot cache of a rasterised overlay.

    The HUD is redrawn every frame but its *content* changes rarely — a note
    name, a vertex count, a value being turned. Rasterising Pillow text and
    arcs is the expensive half of the work and compositing is the cheap half,
    so keeping the last tile and re-blitting it takes the steady-state cost of
    the whole HUD from tens of milliseconds down to about one. A single slot is
    enough because consecutive frames almost always share the same key.
    """

    __slots__ = ("_key", "_tile")

    def __init__(self):
        self._key = None
        self._tile = None

    def get(self, key):
        return self._tile if key == self._key else None

    def put(self, key, tile: np.ndarray) -> np.ndarray:
        self._key = key
        self._tile = tile
        return tile


def scrim(frame: np.ndarray, x: int, y: int, w: int, h: int,
          top: float = 0.0, bottom: float | None = None) -> None:
    """Darken a region in place so HUD text survives a bright camera image.

    ``top`` and ``bottom`` are the darkening strength at the top and bottom
    edges; when ``bottom`` is omitted the region is darkened uniformly. The
    gradient form matters more than it sounds: a flat scrim ends in a hard
    horizontal seam straight across the picture, which looks like a rendering
    bug. Ramping it in makes the bar sit on the image instead of being stamped
    onto it.
    """
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1]
    if bottom is None or bottom == top:
        # cv2 does this with SIMD and writes in place; the numpy round-trip
        # through float32 costs several times more for the same result.
        cv2.convertScaleAbs(roi, dst=roi, alpha=1.0 - top)
        return

    # Build the ramp across the requested band, then take the visible slice, so
    # clipping at a frame edge does not rescale the gradient.
    ramp = np.linspace(top, bottom, h, dtype=np.float32)[y0 - y:y1 - y]
    np.copyto(roi, (roi.astype(np.float32)
                    * (1.0 - ramp)[:, None, None]).astype(np.uint8))
