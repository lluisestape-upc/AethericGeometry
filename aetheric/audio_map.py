"""Maps polygon geometry to synthesis parameters."""
import numpy as np

from .gestures import polygon_area
from .tuning import chord_frequencies, pitch_from_norm, quantize_pentatonic


def compute_poly_sound(poly_pts: list, active_l: list, active_r: list,
                       W: int, H: int) -> tuple:
    """Map polygon geometry to synthesis parameters.

    Axis mapping:
      X centroid   → pitch        (left = low, right = high)
      Y centroid   → filter       (top = open/bright, bottom = dark)
      Polygon area → reverb wet   (unlocked at 6 vertices)
      Aspect ratio → tremolo depth (unlocked at 8 vertices)

    Returns (frequencies, effects_dict, note_pos).
    """
    if not poly_pts:
        return [440.0], {}, None

    pts     = np.array(poly_pts)
    cx      = float(np.mean(pts[:, 0]))
    cy      = float(np.mean(pts[:, 1]))
    n_verts = len(poly_pts)

    base     = quantize_pentatonic(pitch_from_norm(cx / W, lo=110.0, hi=880.0))
    n_voices = max(2, min(len(active_l) + len(active_r), 8))
    freqs    = chord_frequencies(base, n_voices)

    effects: dict = {"filter": float(np.clip(1.0 - cy / H, 0.0, 1.0))}

    if n_verts >= 6:
        area     = polygon_area(pts)
        max_area = W * H * 0.20
        effects["reverb"] = float(np.clip(area / max_area, 0.0, 1.0))

    if n_verts >= 8:
        x_span  = float(pts[:, 0].max() - pts[:, 0].min())
        y_span  = float(pts[:, 1].max() - pts[:, 1].min()) + 1.0
        effects["tremolo"] = float(np.clip((x_span / y_span - 0.8) / 1.4, 0.0, 1.0))

    return freqs, effects, (int(cx), int(cy))
