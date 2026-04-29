import cv2
import mediapipe as mp
import numpy as np
import time

from synth import (
    SynthEngine, CHORD_RATIOS,
    pitch_from_norm, quantize_pentatonic, freq_to_note,
)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# ── Landmark IDs ──────────────────────────────────────────────────────────────
THUMB    = 4
INDEX    = 8
MIDDLE   = 12
RING     = 16
PINKY    = 20
ALL_TIPS = (THUMB, INDEX, MIDDLE, RING, PINKY)

# ── Thresholds (pixels) ───────────────────────────────────────────────────────
PINCH_THRESH        = 60
DOUBLE_PINCH_THRESH = 90
PYRAMID_THRESH      = 70
COOLDOWN_FRAMES     = 45   # ~1.5 s at 30 fps before auto-reset

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
PANEL_BG     = (18, 12, 28)
PANEL_BORDER = (65, 55, 95)
NEON_GREEN   = (70, 230, 90)
NEON_CYAN    = (240, 210, 40)
NEON_PINK    = (215, 65, 230)
TEXT_DIM     = (145, 135, 165)
NEAR_WHITE   = (230, 225, 242)
WARM_RED     = (40, 40, 220)

WAVES       = ("SIN", "SAW", "SQUARE", "TRIANGLE")
STATE_COLOR = {"IDLE": TEXT_DIM, "HILO": NEON_GREEN, "POLIGONO": NEON_CYAN}

# Effect definitions: (param-key, min-vertices-to-unlock, short-label, color)
EFFECT_DEFS = [
    ("filter",  4, "FILTER",  NEON_CYAN),
    ("reverb",  6, "REVERB",  NEON_PINK),
    ("tremolo", 8, "TREMOLO", NEON_GREEN),
]


# ── Geometry helpers ──────────────────────────────────────────────────────────
def lm_px(hand, lid, W, H):
    m = hand.landmark[lid]
    return int(m.x * W), int(m.y * H)

def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def mid(a, b):
    return (a[0] + b[0]) // 2, (a[1] + b[1]) // 2

def fingers_up(hand):
    return sum(
        1 for tip, knuckle in ((INDEX, 6), (MIDDLE, 10), (RING, 14), (PINKY, 18))
        if hand.landmark[tip].y < hand.landmark[knuckle].y
    )

def polygon_area(pts: np.ndarray) -> float:
    x = pts[:, 0].astype(float)
    y = pts[:, 1].astype(float)
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


# ── Polygon → sound mapping ───────────────────────────────────────────────────
def compute_poly_sound(poly_pts, active_l, active_r, W, H):
    """
    Return (frequencies, effects_dict, note_pos) from the current polygon.

    Axis mapping:
      X centroid  → pitch  (left = low, right = high)
      Y centroid  → filter brightness  (top = open, bottom = dark)
      Poly area   → reverb wet  (unlocked at 6 vertices)
      Aspect ratio→ tremolo depth  (unlocked at 8 vertices)
    """
    if not poly_pts:
        return [440.0], {}, None

    pts     = np.array(poly_pts)
    cx      = float(np.mean(pts[:, 0]))
    cy      = float(np.mean(pts[:, 1]))
    n_verts = len(poly_pts)

    # X → pitch
    base     = quantize_pentatonic(pitch_from_norm(cx / W, lo=110.0, hi=880.0))
    n_voices = max(2, min(len(active_l) + len(active_r), 8))
    ratios   = CHORD_RATIOS.get(n_voices, CHORD_RATIOS[8])
    freqs    = [base * r for r in ratios[:n_voices]]

    # Y → filter (top=open=1, bottom=dark=0)
    effects = {"filter": float(np.clip(1.0 - cy / H, 0.0, 1.0))}

    # Area → reverb  (active at ≥6 vertices)
    if n_verts >= 6:
        area     = polygon_area(pts)
        max_area = W * H * 0.20          # 20 % of frame as reference
        effects["reverb"] = float(np.clip(area / max_area, 0.0, 1.0))

    # Aspect ratio → tremolo  (active at ≥8 vertices)
    if n_verts >= 8:
        x_span  = float(pts[:, 0].max() - pts[:, 0].min())
        y_span  = float(pts[:, 1].max() - pts[:, 1].min()) + 1.0
        aspect  = x_span / y_span
        # neutral at 1:1, increases as shape gets wider
        effects["tremolo"] = float(np.clip((aspect - 0.8) / 1.4, 0.0, 1.0))

    note_pos = (int(cx), int(cy))
    return freqs, effects, note_pos


# ── Drawing helpers ───────────────────────────────────────────────────────────
def blend_fill(img, x, y, w, h, color, alpha):
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def draw_panel(img, x, y, w, h, alpha=0.68):
    blend_fill(img, x, y, w, h, PANEL_BG, alpha)
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_BORDER, 1)

def draw_waveform_graph(img, shape, x, y, w, h, color):
    if w < 4:
        return
    t = np.linspace(0, 2 * np.pi, w, endpoint=False)
    if   shape == "SIN":    wave = np.sin(t)
    elif shape == "SAW":    wave = (t / np.pi) % 2 - 1.0
    elif shape == "SQUARE": wave = np.sign(np.sin(t)).astype(float)
    else:                   wave = 2 * np.abs((t / np.pi) % 2 - 1) - 1
    margin = 5
    xs  = (x + np.arange(w)).astype(np.int32)
    ys  = np.clip(y + h // 2 - wave * (h // 2 - margin), y, y + h - 1).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)

def draw_note_label(img, note, cx, cy, color):
    scale = 0.80
    (tw, th), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    pad = 6
    x0  = cx - tw // 2 - pad
    y0  = cy - th - pad
    blend_fill(img, x0, y0, tw + pad * 2, th + pad * 2, PANEL_BG, 0.72)
    cv2.rectangle(img, (x0, y0), (x0 + tw + pad * 2, y0 + th + pad * 2), color, 1)
    cv2.putText(img, note, (x0 + pad, cy),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

def draw_effect_bars(img, n_verts, effects, x, cy):
    """Draw compact inline effect bars starting at (x, cy-centred)."""
    BAR_W = 32
    for key, min_v, label, col in EFFECT_DEFS:
        active = n_verts >= min_v
        lc     = col if active else TEXT_DIM
        # short letter label
        cv2.putText(img, label[0], (x, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, lc, 1, cv2.LINE_AA)
        # bar outline
        bx, by, bh = x + 12, cy - 5, 10
        cv2.rectangle(img, (bx, by), (bx + BAR_W, by + bh), PANEL_BORDER, 1)
        # bar fill
        if active:
            fill = int(BAR_W * float(effects.get(key, 0.0)))
            if fill > 0:
                cv2.rectangle(img, (bx, by), (bx + fill, by + bh), col, -1)
        x += BAR_W + 20

def draw_status_bar(img, state, waveform, n_verts, note, effects, W, H):
    bar_h = 72
    blend_fill(img, 0, H - bar_h, W, bar_h, PANEL_BG, 0.75)
    cv2.line(img, (0, H - bar_h), (W, H - bar_h), PANEL_BORDER, 1)

    cy  = H - bar_h // 2
    col = STATE_COLOR.get(state, NEAR_WHITE)

    # State dot + label
    cv2.circle(img, (22, cy), 7, col, -1, cv2.LINE_AA)
    cv2.putText(img, state, (38, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, col, 2, cv2.LINE_AA)

    if state == "HILO":
        # Show current note in centre
        if note:
            cv2.putText(img, note, (W // 2 - 18, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, NEAR_WHITE, 2, cv2.LINE_AA)

    elif state == "POLIGONO":
        cv2.putText(img, f"{n_verts} pts", (175, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEON_CYAN, 1, cv2.LINE_AA)
        # Live effect bars (note shown as floating label on polygon)
        draw_effect_bars(img, n_verts, effects, x=240, cy=cy)

    # Waveform mini-panel (right-anchored)
    ww, wh = 100, 44
    wx     = W - ww - 14
    wy     = H - bar_h + (bar_h - wh) // 2
    draw_panel(img, wx, wy, ww, wh, alpha=0.55)
    draw_waveform_graph(img, waveform, wx + 5, wy + 2, ww - 10, wh - 4, NEON_PINK)
    (tw, _), _ = cv2.getTextSize(waveform, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, waveform, (wx - tw - 8, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEON_PINK, 1, cv2.LINE_AA)

def draw_hints(img, state):
    hints = {
        "IDLE":     "Doble pinza -> activar HILO",
        "HILO":     "Beso/Piramide -> POLIGONO   |   Doble pinza -> IDLE",
        "POLIGONO": "X=nota  Y=filtro  [6pts: +reverb]  [8pts: +tremolo]",
    }
    cv2.putText(img, hints.get(state, ""), (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_DIM, 1, cv2.LINE_AA)

def draw_fps(img, fps, W):
    cv2.putText(img, f"{fps} fps", (W - 72, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv2.LINE_AA)

def draw_cooldown_border(img, n, W, H):
    ratio = n / COOLDOWN_FRAMES
    ov    = img.copy()
    cv2.rectangle(ov, (0, 0), (W, H), WARM_RED, max(2, int(12 * ratio)))
    alpha = 0.15 + 0.35 * ratio
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def draw_skeleton(img, results):
    if not results.multi_hand_landmarks:
        return
    lm_s  = mp_draw.DrawingSpec(color=(70, 60, 85),  thickness=1, circle_radius=2)
    con_s = mp_draw.DrawingSpec(color=(55, 48, 70), thickness=1)
    for hand in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS, lm_s, con_s)

def draw_hilo(img, lh, rh, W, H, color):
    a = lm_px(lh, INDEX, W, H)
    b = lm_px(rh, INDEX, W, H)
    cv2.line(img, a, b, color, 3, cv2.LINE_AA)
    for pt in (a, b):
        cv2.circle(img, pt, 10, color, -1, cv2.LINE_AA)
        ov = img.copy()
        cv2.circle(ov, pt, 20, color, 2, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)

def draw_polygon(img, pts, color):
    if len(pts) < 3:
        return
    arr = np.array(pts, np.int32)
    ov  = img.copy()
    cv2.fillPoly(ov, [arr], color)
    cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)
    cv2.polylines(img, [arr], True, color, 3, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(img, pt, 6, NEAR_WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 6, color, 2, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    hands_model = mp_hands.Hands(
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
        max_num_hands=2,
    )
    cam   = cv2.VideoCapture(0)
    synth = SynthEngine()
    if not synth.start():
        print("[audio] sounddevice not available — running visual-only.")

    state        = "IDLE"
    waveform     = "SIN"
    active_l     = [INDEX]
    active_r     = [INDEX]
    prev_dpinch  = False
    pending      = None
    missing      = 0
    current_note = ""
    current_fx   = {}

    fps_count   = 0
    fps_display = 0
    fps_t       = time.time()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            frame   = cv2.flip(frame, 1)
            H, W    = frame.shape[:2]
            results = hands_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            fps_count += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps_display = fps_count
                fps_count   = 0
                fps_t       = now

            # Assign hands by thumb-x vs pinky-base-x
            lh = rh = None
            for m in (results.multi_hand_landmarks or []):
                if m.landmark[THUMB].x < m.landmark[17].x:
                    rh = m
                else:
                    lh = m

            # Waveform selector: left hand finger count, IDLE only
            if state == "IDLE" and lh:
                n = fingers_up(lh)
                if 1 <= n <= 4:
                    waveform = WAVES[n - 1]

            poly_pts = []

            if lh and rh:
                missing = 0

                pl = lm_px(lh, THUMB, W, H);  il = lm_px(lh, INDEX, W, H)
                pr = lm_px(rh, THUMB, W, H);  ir = lm_px(rh, INDEX, W, H)

                pinch_l = dist(pl, il) < PINCH_THRESH
                pinch_r = dist(pr, ir) < PINCH_THRESH
                d_pinch = pinch_l and pinch_r
                rise    = d_pinch and not prev_dpinch
                fall    = not d_pinch and prev_dpinch
                kiss    = d_pinch and dist(mid(pl, il), mid(pr, ir)) < DOUBLE_PINCH_THRESH
                pyramid = (
                    sum(
                        1 for lid in ALL_TIPS
                        if dist(lm_px(lh, lid, W, H), lm_px(rh, lid, W, H)) < PYRAMID_THRESH
                    ) >= 4
                )

                # State machine
                if state == "IDLE":
                    if rise:
                        state   = "HILO"
                        pending = None

                elif state == "HILO":
                    if kiss:
                        state    = "POLIGONO"
                        active_l = [INDEX]
                        active_r = [INDEX]
                        pending  = None
                    elif pyramid:
                        state    = "POLIGONO"
                        active_l = list(ALL_TIPS[1:])
                        active_r = list(ALL_TIPS[1:])
                        pending  = None
                    elif rise:
                        pending = "RESET"
                    elif fall and pending == "RESET":
                        state    = "IDLE"
                        active_l = [INDEX]
                        active_r = [INDEX]
                        pending  = None

                elif state == "POLIGONO":
                    if kiss:
                        state = "HILO"
                    for lid in (MIDDLE, RING, PINKY):
                        if dist(pl, lm_px(lh, lid, W, H)) < PINCH_THRESH and lid not in active_l:
                            active_l.append(lid)
                        if dist(pr, lm_px(rh, lid, W, H)) < PINCH_THRESH and lid not in active_r:
                            active_r.append(lid)

                prev_dpinch = d_pinch

                if state == "POLIGONO":
                    for d in (THUMB, INDEX, MIDDLE, RING, PINKY):
                        if d == THUMB or d in active_l:
                            poly_pts.append(lm_px(lh, d, W, H))
                    for d in (PINKY, RING, MIDDLE, INDEX, THUMB):
                        if d == THUMB or d in active_r:
                            poly_pts.append(lm_px(rh, d, W, H))

            elif state != "IDLE":
                missing += 1
                if missing > COOLDOWN_FRAMES:
                    state       = "IDLE"
                    prev_dpinch = False
                    active_l    = [INDEX]
                    active_r    = [INDEX]

            # ── Sound ─────────────────────────────────────────────────────────
            note_pos = None

            if state == "IDLE":
                synth.set_params(False, [], waveform)
                current_note = ""
                current_fx   = {}

            elif state == "HILO" and lh and rh:
                il_pos = lm_px(lh, INDEX, W, H)
                ir_pos = lm_px(rh, INDEX, W, H)
                freq   = quantize_pentatonic(
                    pitch_from_norm(dist(il_pos, ir_pos) / W, lo=880.0, hi=110.0)
                )
                synth.set_params(True, [freq], waveform)
                current_note = freq_to_note(freq)
                current_fx   = {}
                note_pos     = mid(il_pos, ir_pos)

            elif state == "POLIGONO" and poly_pts:
                freqs, effects, note_pos = compute_poly_sound(
                    poly_pts, active_l, active_r, W, H
                )
                synth.set_params(True, freqs, waveform, amplitude=0.22, effects=effects)
                current_note = freq_to_note(freqs[0])
                current_fx   = effects

            # ── Render ────────────────────────────────────────────────────────
            draw_skeleton(frame, results)
            col = STATE_COLOR.get(state, NEAR_WHITE)

            if state == "HILO" and lh and rh:
                draw_hilo(frame, lh, rh, W, H, col)
            elif state == "POLIGONO":
                draw_polygon(frame, poly_pts, col)

            if note_pos:
                draw_note_label(frame, current_note, note_pos[0], note_pos[1], col)

            if state != "IDLE" and missing > 0:
                draw_cooldown_border(frame, missing, W, H)

            draw_status_bar(frame, state, waveform, len(poly_pts),
                            current_note, current_fx, W, H)
            draw_hints(frame, state)
            draw_fps(frame, fps_display, W)

            cv2.imshow("Aetheric Geometry", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        synth.stop()
        cam.release()
        hands_model.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
